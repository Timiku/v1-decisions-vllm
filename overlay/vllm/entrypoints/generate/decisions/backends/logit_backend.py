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
finalists), or `auto` (direct within the marker count, then
wide-direct, then two-stage; see `read`).

`logprobs` picks how a read gets the markers' logprobs: `exact` (the
engine's logprob_token_ids, every marker exact) or `top-k` (the engine's
plain top-k window, k = --max-logprobs; a marker outside the window
scores the window's lowest logprob, an upper bound). top-k needs no
per-label engine limit and works under speculative decoding, where
logprob_token_ids reads come back incomplete. The server default is the
constructor's `logprobs` (VLLM_TYPED_DECISIONS_BACKEND=logit:logprobs=top-k),
`exact` when unset.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..protocol import CompiledQuestion
from . import (BackendError, BackendResult, cached_tokens, option_mass,
               restricted_softmax)

LOGPROBS_MODES = ("exact", "top-k")


class LogitOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    readout: Literal["auto", "direct", "wide-direct", "two-stage"] = "auto"
    # None: the server's default (the backend's constructor `logprobs`)
    logprobs: Literal["exact", "top-k"] | None = None


class LogitBackend:
    calibrate_by_default = True
    """Reads decisions from a served generative model's next-token logits."""

    name = "logit"
    architectures: tuple[str, ...] = ()   # the fallback for every model
    options_model = LogitOptions

    def __init__(self, host, logprobs: str = "exact"):
        if logprobs not in LOGPROBS_MODES:
            raise ValueError(
                f"logit backend: logprobs must be one of {LOGPROBS_MODES}; "
                f"got {logprobs!r}")
        self.host = host
        self.logprobs = logprobs

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
        options = self._options(question)
        readout = options.readout
        mode = options.logprobs or self.logprobs
        exact = mode == "exact"
        if readout == "auto":
            # direct within the marker count; then wide-direct up to its
            # capacity on this tokenizer. An exact read can't pass the
            # engine's read limit (a stock vLLM caps --max-logprobs and
            # the token-id count), so past it an exact wide-direct read
            # switches to top-k, which has no per-label limit; two-stage
            # only past the tokenizer's capacity. In paired A/B runs,
            # wide-direct was as accurate as two-stage or better, with
            # one pass instead of k+1: more accurate on the 4B, level on
            # the 27B at 32/64 options, about 2x faster on the 4B. On a
            # stock 27B server (window 20), top-k wide-direct beat exact
            # two-stage at 64-255 options (95/90/84 vs 85/83/62 of 100)
            # and was about 10x faster.
            if k <= len(limits.markers):
                readout = "direct"
            elif k <= self.host.wide_direct_capacity(capped=exact):
                readout = "wide-direct"
            elif k <= self.host.wide_direct_capacity(capped=False):
                readout = "wide-direct"
                mode, exact = "top-k", False
            else:
                readout = "two-stage"
        if readout == "two-stage":
            from .large_choice import two_stage_read
            return await two_stage_read(self, question, request_id, limits,
                                        logprobs=mode)
        if readout == "wide-direct":
            from .large_choice import wide_direct_markers
            markers = wide_direct_markers(self.host.tokenizer,
                                          limits.markers)
            if k > len(markers):
                raise BackendError(
                    f"wide-direct capacity on this tokenizer is "
                    f"{len(markers)} single-token markers; k={k}")
            limit = self._read_limit()
            if exact and limit is not None and k > limit:
                raise BackendError(self._over_limit_message(k, "wide-direct"))
            return await self._direct_read(
                question, request_id, labels=markers[:k],
                readout_name="wide-direct", layout_name="merged-pairs",
                logprobs=mode)
        if k > len(limits.markers):
            raise BackendError(
                f"direct read supports at most {len(limits.markers)} "
                f"options (the marker count); k={k}. Use readout "
                "'wide-direct' or 'two-stage' for larger questions.")
        return await self._direct_read(question, request_id, logprobs=mode)

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
                f"{raise_setting}, use logprobs 'top-k' or readout "
                f"'two-stage'.")

    async def _direct_read(self, question: CompiledQuestion,
                           request_id: str, labels: list[str] | None = None,
                           readout_name: str = "direct",
                           layout_name: str = "direct",
                           logprobs: str = "exact") -> BackendResult:
        """One forward pass over one token per option. logprobs `exact`:
        restricted gather; when it exhausts the host's retries with
        option ids still missing (the engine's transient gather defect,
        or speculative decoding), one top-k read scores the rest: an
        honest degraded answer, marked in meta. logprobs `top-k`: one
        top-k read."""
        prompt = await self.host.render(question, labels=labels)
        if logprobs == "top-k":
            return await self._window_read(
                question, prompt, {}, readout_name, layout_name,
                request_id, prior_passes=0, degraded_missing=None)
        found, result, attempts = await self.host.restricted_read(
            prompt.engine_input, prompt.slot_ids, request_id)
        option_logits: dict[str, float] = {}
        missing: list[str] = []
        for option, tok in zip(question.options, prompt.slot_ids):
            if tok in found:
                option_logits[option.id] = found[tok]
            else:
                missing.append(option.id)
        if missing:
            return await self._window_read(
                question, prompt, option_logits, readout_name, layout_name,
                f"{request_id}-degraded", prior_passes=attempts,
                degraded_missing=missing)
        return BackendResult(
            option_logits, restricted_softmax(option_logits),
            forward_passes=attempts,
            meta={"input_tokens": prompt.input_tokens,
                  "cached_input_tokens": cached_tokens(result),
                  "option_mass": option_mass(self.host, option_logits),
                  "readout": readout_name,
                  "label_layout": layout_name})

    async def _window_read(
            self, question: CompiledQuestion, prompt,
            option_logits: dict[str, float], readout_name: str,
            layout_name: str, request_id: str, prior_passes: int,
            degraded_missing: list[str] | None) -> BackendResult:
        """One top-k read on the prompt. Options not already in
        `option_logits` take their marker's logprob from the window; a
        marker outside the window takes the window's lowest logprob (its
        true logprob is at most that), listed in meta `floored`.
        `option_mass` counts only markers read from a pass, so it is a
        lower bound when any option is floored. `degraded_missing` set:
        the fallback after an incomplete exact gather."""
        try:
            found, floor, result = await self.host.topk_read(
                prompt.engine_input, prompt.slot_ids, request_id)
        except BackendError as e:
            if degraded_missing is None:
                raise
            raise BackendError(
                f"option tokens missing from the gathered logprobs: "
                f"{degraded_missing}; cannot score ({e})",
                degraded_missing) from e
        option_logits = dict(option_logits)
        floored: list[str] = []
        for option, tok in zip(question.options, prompt.slot_ids):
            if option.id in option_logits:
                continue
            if tok in found:
                option_logits[option.id] = found[tok]
            else:
                floored.append(option.id)
        # every marker outside the window: no answer to read. An empty
        # window has no floor (-inf), so nothing can be floored either.
        if len(floored) == len(question.options) or (
                floored and floor == float("-inf")):
            raise BackendError(
                f"none of the option markers is in the top-k window "
                f"(k={self.host.topk_window()}): {floored}", floored)
        read = dict(option_logits)
        for oid in floored:
            option_logits[oid] = floor
        meta = {"input_tokens": prompt.input_tokens,
                "cached_input_tokens": cached_tokens(result),
                "option_mass": option_mass(self.host, read),
                "readout": readout_name,
                "label_layout": layout_name,
                "logprobs": "top-k",
                "topk_window": self.host.topk_window()}
        if floored:
            meta["floored"] = floored
        if degraded_missing is not None:
            meta["readout"] = "direct-degraded"
            meta["degraded"] = True
            meta["degraded_reason"] = (
                f"restricted gather missed {degraded_missing}; scored "
                f"from the top-k window")
        return BackendResult(
            option_logits, restricted_softmax(option_logits),
            forward_passes=prior_passes + 1, meta=meta)
