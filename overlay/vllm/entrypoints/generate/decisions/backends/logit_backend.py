# SPDX-License-Identifier: Apache-2.0
"""LogitBackend: restricted-logprob gather on generative checkpoints.

The reference backend. The compiled question is rendered as the standard
decision prompt and one forward pass gathers the next-token logprobs of
exactly the option markers; those are the option scores. No
full-vocabulary materialization; sampler settings don't touch the result.

Per-request options (`backend_options`): `readout` picks how options are
read: `direct` (one marker per option, up to the marker count),
`wide-direct` (markers plus single-token letter pairs, one pass),
`two-stage` (a yes/no read per option, then a direct read of the
finalists), or `auto` (direct within the marker count, else two-stage).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..protocol import CompiledQuestion
from . import (BackendError, BackendResult, cached_tokens, option_mass,
               restricted_softmax)


class LogitOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readout: Literal["auto", "direct", "wide-direct", "two-stage"] = "auto"


class LogitBackend:
    calibrate_by_default = True
    """Reads decisions from a served generative model's next-token logits."""

    name = "logit"
    architectures: tuple[str, ...] = ()   # the fallback for every model
    options_model = LogitOptions

    def __init__(self, host):
        self.host = host

    def _options(self, question: CompiledQuestion) -> LogitOptions:
        opts = question.backend_options
        if opts is None:
            return LogitOptions()
        if isinstance(opts, dict):
            return LogitOptions.model_validate(opts)
        return opts

    async def read(self, question: CompiledQuestion,
                   request_id: str) -> BackendResult:
        limits = self.host.limits
        k = len(question.options)
        readout = self._options(question).readout
        if readout == "auto":
            # direct within the marker count; then wide-direct up to its
            # capacity on this tokenizer, never past the engine's read
            # limit (a stock vLLM caps --max-logprobs and the token-id
            # count); else two-stage. In paired A/B runs, wide-direct
            # was as accurate as two-stage or better, with one pass
            # instead of k+1: more accurate on the 4B, level on the 27B
            # at 32/64 options, about 2x faster on the 4B.
            if k <= len(limits.markers):
                readout = "direct"
            else:
                readout = ("wide-direct"
                           if k <= self.host.wide_direct_capacity()
                           else "two-stage")
        if readout == "two-stage":
            from .large_choice import two_stage_read
            return await two_stage_read(self, question, request_id, limits)
        if readout == "wide-direct":
            from .large_choice import wide_direct_markers
            markers = wide_direct_markers(self.host.tokenizer,
                                          limits.markers)
            if k > len(markers):
                raise BackendError(
                    f"wide-direct capacity on this tokenizer is "
                    f"{len(markers)} single-token markers; k={k}")
            limit = self._read_limit()
            if limit is not None and k > limit:
                raise BackendError(self._over_limit_message(k, "wide-direct"))
            return await self._direct_read(
                question, request_id, labels=markers[:k],
                readout_name="wide-direct", layout_name="merged-pairs")
        if k > len(limits.markers):
            raise BackendError(
                f"direct read supports at most {len(limits.markers)} "
                f"options (the marker count); k={k}. Use readout "
                "'wide-direct' or 'two-stage' for larger questions.")
        return await self._direct_read(question, request_id)

    def _read_limit(self) -> int | None:
        """The engine's restricted-read limit (None = uncapped, e.g. a
        hand-built host in tests)."""
        limit = getattr(self.host, "read_limit", None)
        return limit

    def _over_limit_message(self, k: int, readout: str) -> str:
        limit = self._read_limit()
        max_lp = getattr(self.host.model_config, "max_logprobs", None)
        cap = getattr(self.host, "token_id_cap", None)
        settings = []
        if max_lp is not None and 0 <= max_lp <= k:
            settings.append("--max-logprobs (currently %s)" % max_lp)
        if cap is not None and cap <= k:
            settings.append("the MAX_LOGPROB_TOKEN_IDS cap in "
                            "vllm/sampling_params.py (currently %s)" % cap)
        raise_setting = " and ".join(settings) if settings else \
            "the engine's read limit"
        return (f"{readout} read needs {k} one-pass logprobs but the "
                f"engine reads at most {limit} "
                f"(max_logprobs={max_lp}, token-id cap={cap}); raise "
                f"{raise_setting} or use readout 'two-stage'.")

    async def _direct_read(self, question: CompiledQuestion,
                           request_id: str, labels: list[str] | None = None,
                           readout_name: str = "direct",
                           layout_name: str = "direct") -> BackendResult:
        """One forward pass; restricted gather over one token per option."""
        prompt = await self.host.render(question, labels=labels)
        found, result = await self.host.restricted_read(
            prompt.engine_input, prompt.slot_ids, request_id)
        option_logits: dict[str, float] = {}
        missing: list[str] = []
        for option, tok in zip(question.options, prompt.slot_ids):
            if tok in found:
                option_logits[option.id] = found[tok]
            else:
                missing.append(option.id)
        if missing:
            raise BackendError(
                f"option tokens missing from the gathered logprobs: "
                f"{missing}; cannot score", missing)
        return BackendResult(
            option_logits, restricted_softmax(option_logits),
            forward_passes=1,
            meta={"input_tokens": prompt.input_tokens,
                  "cached_input_tokens": cached_tokens(result),
                  "option_mass": option_mass(self.host, option_logits),
                  "readout": readout_name,
                  "label_layout": layout_name})
