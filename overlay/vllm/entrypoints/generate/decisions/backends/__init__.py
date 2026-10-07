# SPDX-License-Identifier: Apache-2.0
"""Decision backends: pluggable readout strategies behind one wire.

A backend turns one CompiledQuestion into a score per option. Three ship:

- `logit`   any generative checkpoint: restricted-logprob gather over the
            option markers (the default).
- `encoder` token_classify pooling models (Laya): one pooling pass.
- `canvas`  DiffusionGemma: a read-only canvas denoise.

A backend is a class with:

    name: str                       # key for selection, e.g. "logit"
    architectures: tuple[str, ...]  # model architectures it serves by
                                    # default (auto-selection); may be ()
    options_model: type | None      # pydantic model for per-request
                                    # `backend_options`; None = takes none
    def __init__(self, host, **kwargs)   # host: BackendHost (host.py)
    async def read(self, question, request_id) -> BackendResult

and optionally

    async def read_many(self, questions, request_id)
        -> list[BackendResult | BackendError | None]

which answers several questions of one request (same state) from shared
engine requests. The server calls it once for a multi-question request;
a None entry is read on its own with `read`. See README.md.

Register it with `register_backend`, or from a package advertising the
`vllm.decision_backends` entry-point group. Selection at startup:
`VLLM_TYPED_DECISIONS_BACKEND=name[:k=v,...]`, else the backend claiming
the served model's architecture, else `logit`. Per request: `backend`.

Whatever a backend puts in `BackendResult.meta` is returned under the
answer's `extra.backend`, except the standard keys the server lifts into
`extra.audit`: input_tokens, cached_input_tokens, option_mass, readout,
label_layout.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "BackendError", "BackendResult", "DecisionBackend", "register_backend",
    "get_backend", "select_backend_name", "validate_backend_options",
    "restricted_softmax", "option_mass", "cached_tokens", "add_known",
]

# meta keys the server lifts into extra.audit (same meaning on every
# backend); everything else stays in extra.backend
AUDIT_META_KEYS = ("input_tokens", "cached_input_tokens", "option_mass",
                   "readout", "label_layout")


@runtime_checkable
class DecisionBackend(Protocol):
    name: str
    # Startup calibration: whether this backend calibrates
    # by default (CALIBRATION unset or "jevbench"). Only the logit
    # backend has measured evidence that it needs a T.
    calibrate_by_default: bool

    async def read(self, question, request_id: str) -> "BackendResult": ...


class BackendError(Exception):
    """A read the caller can act on failed; the question is reported in
    partial_failures with this message."""

    def __init__(self, message: str, option_ids: list[str] | None = None):
        super().__init__(message)
        self.option_ids = option_ids or []


class BackendResult:
    """One question's read. `option_logits`: a score per option id on a
    log scale (logprobs, logits, or log of averaged probabilities), so
    softmax(option_logits / T) is the calibrated distribution.
    `probabilities`: restricted_softmax(option_logits). `meta`: what the
    backend did (see the module docstring)."""

    __slots__ = ("option_logits", "probabilities", "forward_passes", "meta")

    def __init__(self, option_logits: dict[str, float],
                 probabilities: dict[str, float],
                 forward_passes: int, meta: dict | None = None):
        self.option_logits = option_logits
        self.probabilities = probabilities
        self.forward_passes = forward_passes
        self.meta = meta or {}


def restricted_softmax(logits: dict, temperature: float = 1.0) -> dict:
    """Softmax over the option scores only; never the full vocabulary."""
    m = max(logits.values())
    exps = {k: math.exp((v - m) / temperature) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: e / total for k, e in exps.items()}


def option_mass(host, logprobs: dict) -> float | None:
    """Share of the model's full next-token probability that landed on
    ANY option: sum(exp(logprob)) over the option tokens. Near 1: the
    model wanted to answer with an option. Low: the options don't fit or
    the prompt confuses it. Only meaningful when the engine reports
    full-vocabulary log-probabilities (logprobs_mode raw_logprobs, the
    default); None otherwise."""
    mode = getattr(host.model_config, "logprobs_mode", "raw_logprobs")
    if mode != "raw_logprobs":
        return None
    return sum(math.exp(v) for v in logprobs.values())


def cached_tokens(result) -> int | None:
    """Prompt tokens the engine served from its prefix cache for one
    engine request (RequestOutput.num_cached_tokens); None if unknown."""
    n = getattr(result, "num_cached_tokens", None)
    return n if isinstance(n, int) else None


def add_known(*values: int | None) -> int | None:
    """Sum of the known values; None when none is known."""
    known = [v for v in values if v is not None]
    return sum(known) if known else None


# ---------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}
_BUILTINS = ("logit", "encoder", "canvas")


def register_backend(name: str, cls) -> None:
    """Register a backend class under `name`. Re-registering a name is
    refused; don't name a plugin after a built-in."""
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__qualname__} needs a `name` attribute")
    if name in _REGISTRY:
        raise ValueError(f"decision backend {name!r} is already registered")
    _REGISTRY[name] = cls


def _ensure_builtins() -> None:
    from .canvas_backend import CanvasBackend
    from .encoder_backend import EncoderBackend
    from .logit_backend import LogitBackend
    for cls in (LogitBackend, EncoderBackend, CanvasBackend):
        _REGISTRY.setdefault(cls.name, cls)


def load_decision_backend_plugins() -> None:
    """Load backends (`vllm.decision_backends`) and question types
    (`vllm.decision_question_types`) advertised by installed packages.
    Each entry resolves to a callable that registers itself. A failing
    plugin is logged and skipped. Call once at server start."""
    from importlib.metadata import entry_points
    for ep in entry_points(group="vllm.decision_backends"):
        try:
            ep.load()()
        except Exception:  # noqa: BLE001 - plugin isolation
            logger.exception(
                "decision backend plugin %r failed to load; selecting it "
                "by name will fail with an unknown-backend error", ep.name)
    from ..question_types import load_question_type_plugins
    load_question_type_plugins()


def select_backend_name(architectures: list[str]) -> str:
    """The backend that claims one of the served model's architectures;
    a plugin's claim wins over a built-in's. `logit` when none claims."""
    _ensure_builtins()
    archs = set(architectures)
    claims = [n for n, cls in _REGISTRY.items()
              if archs & set(getattr(cls, "architectures", ()) or ())]
    if not claims:
        return "logit"
    plugins = [n for n in claims if n not in _BUILTINS]
    chosen = (plugins or claims)[0]
    if len(claims) > 1:
        logger.warning("several decision backends claim %s: %s; using %r",
                       sorted(archs), claims, chosen)
    return chosen


def get_backend(name: str, host, **kwargs):
    """Construct the backend registered as `name` over `host`."""
    _ensure_builtins()
    if name in _REGISTRY:
        return _REGISTRY[name](host, **kwargs)
    raise ValueError(f"unknown decision backend '{name}' (loaded: "
                     f"{', '.join(sorted(_REGISTRY))})")


def validate_backend_options(backend, options: dict | None) -> Any:
    """Validate a request's `backend_options` against the backend's
    `options_model`. Returns the validated model (or None when none was
    sent). Raises ValueError with a caller-facing message."""
    if not options:
        return None
    model = getattr(backend, "options_model", None)
    if model is None:
        raise ValueError(f"backend {backend.name!r} takes no "
                         f"backend_options (got {sorted(options)})")
    from pydantic import ValidationError
    try:
        return model.model_validate(options)
    except ValidationError as e:
        raise ValueError(f"backend_options for {backend.name!r}: {e}") \
            from None


def parse_backend_spec(spec: str) -> tuple[str, dict]:
    """Parse `name[:kwarg=val[,kwarg=val...]]` into (name, kwargs) for
    VLLM_TYPED_DECISIONS_BACKEND. Values: int, float, bool or string."""
    name, sep, kv = spec.partition(":")
    name = name.strip()
    kwargs: dict = {}
    if not sep:
        return name, kwargs
    import ast
    for pair in kv.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, eq, v = pair.partition("=")
        if not k or not eq:
            raise ValueError(f"invalid backend spec {spec!r}: expected "
                             f"kwarg=value in {pair!r}")
        v = v.strip()
        if v.lower() in ("true", "false"):
            kwargs[k.strip()] = v.lower() == "true"
        else:
            try:
                kwargs[k.strip()] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                kwargs[k.strip()] = v
    return name, kwargs


def validation_error(message: str, code: int = 422):
    """An ErrorResponse-shaped request-validation failure. The routes
    answer body-validation failures with it: /v1/systemone with 422
    (Jev), /v1/decisions with 400 (OpenAI)."""
    try:
        from vllm.entrypoints.serve.engine.protocol import (
            ErrorInfo, ErrorResponse)
    except ImportError:
        from vllm.entrypoints.openai.engine.protocol import (
            ErrorInfo, ErrorResponse)
    return ErrorResponse(error=ErrorInfo(
        message=message, type="invalid_request_error", param=None,
        code=code))
