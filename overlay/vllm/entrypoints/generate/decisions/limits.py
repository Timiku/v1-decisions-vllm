# SPDX-License-Identifier: Apache-2.0
"""Startup settings for the decisions endpoints.

Every cap is a startup setting, shared by both endpoints and both request
forms: A-Z markers, 255 options per choice (Jev's limit), 10 score
levels, 64 questions. The server's default calibration temperature and
the answer-slot self-check threshold live here too. Loaded once via `load_limits()`; `get_limits()` returns the
process-wide instance; tests can install one with
`set_limits_for_tests()`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

MARKERS_ENV = "VLLM_TYPED_DECISIONS_MARKERS"
MAX_OPTIONS_ENV = "VLLM_TYPED_DECISIONS_MAX_OPTIONS"
SHORTLIST_ENV = "VLLM_TYPED_DECISIONS_SHORTLIST"
MAX_SCORE_LEVELS_ENV = "VLLM_TYPED_DECISIONS_MAX_SCORE_LEVELS"
MAX_QUESTIONS_ENV = "VLLM_TYPED_DECISIONS_MAX_QUESTIONS"
TEMPERATURE_ENV = "VLLM_TYPED_DECISIONS_TEMPERATURE"
MIN_OPTION_MASS_ENV = "VLLM_TYPED_DECISIONS_MIN_OPTION_MASS"
CALIBRATION_ENV = "VLLM_TYPED_DECISIONS_CALIBRATION"
CALIBRATION_DIR_ENV = "VLLM_TYPED_DECISIONS_CALIBRATION_DIR"


def engine_read_limit(model_config) -> tuple[int | None, int | None]:
    """The engine's restricted-read limit as (limit, token_id_cap).

    A restricted gather asks for `logprobs=len(token_ids)`, so the read
    is bounded by the engine's `--max-logprobs` AND by vLLM's
    MAX_LOGPROB_TOKEN_IDS cap on the number of token ids per request
    (when the running vLLM defines it). Either bound may be absent:
    vLLM's `max_logprobs -1` means uncapped, and the constant may not
    exist. Returns (None, cap) when nothing caps the read, so callers
    can say "uncapped"; the cap is None when this vLLM has no such
    constant, so messages can name which setting to raise."""
    max_logprobs = getattr(model_config, "max_logprobs", None)
    cap = None
    try:
        from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS
        cap = int(MAX_LOGPROB_TOKEN_IDS)
    except (ImportError, AttributeError):
        pass
    limit: int | None = None
    if max_logprobs is not None and max_logprobs >= 0:
        limit = max_logprobs
    if cap is not None:
        limit = cap if limit is None else min(limit, cap)
    return limit, cap


def expand_char_spec(spec: str) -> list[str]:
    """Parse a comma-separated char spec: `A-Z,a-z` style ranges and
    single characters, in order. Raises ValueError on malformed input."""
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if len(part) == 3 and part[1] == "-":
            lo, hi = part[0], part[2]
            if ord(lo) > ord(hi):
                raise ValueError(f"invalid range {part!r} in spec {spec!r}")
            out.extend(chr(c) for c in range(ord(lo), ord(hi) + 1))
        elif len(part) == 1:
            out.append(part)
        else:
            raise ValueError(
                f"invalid entry {part!r} in {spec!r}: expected a single "
                "character or a X-Y range")
    if not out:
        raise ValueError(f"empty character spec {spec!r}")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate characters in spec {spec!r}")
    return out


@dataclass(frozen=True)
class DecisionLimits:
    markers: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    max_options: int = 255
    shortlist: int = 16
    max_score_levels: int = 10
    max_questions: int = 64
    # Server-default calibration temperature: probabilities =
    # softmax(logits / T). A request's calibration_temperature overrides
    # it. 1.0 = raw restricted softmax.
    temperature: float = 1.0
    # Answer-slot self-check: on an easy probe question the model must put
    # at least this share of its next-token probability on the option
    # letters, or the logit backend refuses to serve. 0 turns it off.
    min_option_mass: float = 0.5
    # Startup calibration: what to calibrate on at
    # startup - "jevbench" (default, logit only unless CALIBRATION=on),
    # "on" (any backend with a question set), a path to an operator
    # question file (any backend), or "off".
    calibration: str | None = None
    # Where saved calibration results go.
    calibration_dir: str | None = None

    def __post_init__(self):
        if self.shortlist > len(self.markers):
            raise ValueError(
                f"{SHORTLIST_ENV}={self.shortlist} exceeds the marker "
                f"count ({len(self.markers)})")
        if self.max_score_levels > len(self.markers):
            raise ValueError(
                f"{MAX_SCORE_LEVELS_ENV}={self.max_score_levels} exceeds "
                f"the marker count ({len(self.markers)})")
        if self.shortlist < 2:
            raise ValueError(f"{SHORTLIST_ENV} must be >= 2")
        if not self.temperature > 0:
            raise ValueError(f"{TEMPERATURE_ENV} must be > 0")
        if not 0 <= self.min_option_mass <= 1:
            raise ValueError(f"{MIN_OPTION_MASS_ENV} must be in [0, 1]")


def load_limits() -> DecisionLimits:
    kwargs: dict = {}
    if v := os.environ.get(MARKERS_ENV):
        kwargs["markers"] = tuple(expand_char_spec(v))
    if v := os.environ.get(MAX_OPTIONS_ENV):
        kwargs["max_options"] = int(v)
    if v := os.environ.get(SHORTLIST_ENV):
        kwargs["shortlist"] = int(v)
    if v := os.environ.get(MAX_SCORE_LEVELS_ENV):
        kwargs["max_score_levels"] = int(v)
    if v := os.environ.get(MAX_QUESTIONS_ENV):
        kwargs["max_questions"] = int(v)
    if v := os.environ.get(TEMPERATURE_ENV):
        kwargs["temperature"] = float(v)
    if v := os.environ.get(MIN_OPTION_MASS_ENV):
        kwargs["min_option_mass"] = float(v)
    if v := os.environ.get(CALIBRATION_ENV):
        kwargs["calibration"] = v
    if v := os.environ.get(CALIBRATION_DIR_ENV):
        kwargs["calibration_dir"] = v
    return DecisionLimits(**kwargs)


def calibration_mode(limits: DecisionLimits | None = None) -> str:
    """Normalized calibration setting: "jevbench", "on", "off", or
    "file". Anything that is not an existing file is a file path, which
    is validated at startup by the caller (a missing file falls back to
    T=1.0 there)."""
    limits = limits or get_limits()
    v = limits.calibration
    if not v or v == "jevbench":
        return "jevbench"
    if v == "on":
        return "on"
    if v == "off":
        return "off"
    return "file"


def operator_temperature_set() -> bool:
    """True when VLLM_TYPED_DECISIONS_TEMPERATURE was set by the
    operator: present and non-empty in the environment. A limits object
    with a temperature != 1.0 counts too (that is how tests install an
    operator value; load_limits only puts a non-default value there when
    the env had one)."""
    if os.environ.get(TEMPERATURE_ENV):
        return True
    try:
        return get_limits().temperature != 1.0
    except Exception:
        return False


def default_calibration_dir() -> str:
    return os.path.join(
        os.environ.get("HF_HOME")
        or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        "decisions-calibration")


_LIMITS: DecisionLimits | None = None


def get_limits() -> DecisionLimits:
    global _LIMITS
    if _LIMITS is None:
        _LIMITS = load_limits()
    return _LIMITS


def set_limits_for_tests(limits: DecisionLimits) -> None:
    global _LIMITS
    _LIMITS = limits
