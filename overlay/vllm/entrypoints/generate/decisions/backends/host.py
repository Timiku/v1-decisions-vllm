# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BackendHost: everything a decision backend may use from the server.

A backend is constructed as `Backend(host, **kwargs)` and talks to the
engine only through this object. The built-in backends (logit, encoder,
canvas) use nothing else, which is what keeps the surface sufficient and
lets the serving internals change without breaking plugins.

Surface:
- `model_config`, `tokenizer`, `limits`, `architectures`
- `await render(question, labels=None)` -> RenderedPrompt: the standard
  decision prompt (chat template, thinking off, markers checked)
- `await restricted_read(engine_input, token_ids, request_id)` ->
  (logprob by token id, RequestOutput): one pass, logprobs for exactly
  those token ids
- `await generate(engine_input, sampling_params, request_id)` -> the
  final RequestOutput, for reads with custom sampling params
- `await pool(prompt_ids, pooling_params, request_id)` -> the final
  pooling output, for encoder (pooling) models
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import BackendError


@dataclass
class RenderedPrompt:
    engine_input: Any        # pass to restricted_read / generate
    input_tokens: int        # prompt length in tokens
    slot_ids: list[int]      # token id of each option's marker, in order
    prompt_ids: list[int]    # the prompt's token ids


class BackendHost:

    def __init__(self, serving):
        self._serving = serving
        # set once at startup (serving.py); None until then (tests that
        # build a host by hand). Backends treat None as "no engine cap".
        self.read_limit: int | None = None
        self.token_id_cap: int | None = None

    def wide_direct_capacity(self) -> int:
        """Options wide-direct can read in one pass on this tokenizer,
        never more than the engine's read limit."""
        from .large_choice import wide_direct_markers
        wd = len(wide_direct_markers(self.tokenizer, self.limits.markers))
        if self.read_limit is not None:
            wd = min(wd, self.read_limit)
        return wd

    # ---- facts ------------------------------------------------------
    @property
    def model_config(self):
        return self._serving.model_config

    @property
    def architectures(self) -> list[str]:
        return list(getattr(self.model_config, "architectures", None) or [])

    @property
    def tokenizer(self):
        return self._serving.base_renderer.get_tokenizer()

    @property
    def limits(self):
        from ..limits import get_limits
        return get_limits()

    # ---- prompts ----------------------------------------------------
    async def render(self, question, labels: list[str] | None = None
                     ) -> RenderedPrompt:
        """The standard decision prompt for `question`. `labels` replaces
        the default A, B, C... markers (each must be one token). Raises
        BackendError when the prompt can't be read reliably (too many
        options, a marker that isn't one token, over the context)."""
        if labels is None:
            build = await self._serving._build_prompt(question)
        else:
            build = await self._serving._build_prompt(question,
                                                      labels=labels)
        if not isinstance(build, tuple):
            err = getattr(build, "error", None)
            raise BackendError(getattr(err, "message", None) or str(build))
        engine_input, input_tokens, slot_ids = build
        prompt_ids = (engine_input.get("prompt_token_ids")
                      if isinstance(engine_input, dict)
                      else getattr(engine_input, "prompt_token_ids", None))
        return RenderedPrompt(engine_input, input_tokens, list(slot_ids),
                              list(prompt_ids or []))

    # ---- engine -----------------------------------------------------
    async def generate(self, engine_input, sampling_params,
                       request_id: str):
        """Run one engine request; return its final RequestOutput."""
        log = getattr(self._serving, "_log_inputs", None)
        if log is not None:
            log(request_id, engine_input, params=sampling_params,
                lora_request=None)
        result = None
        async for res in self._serving.engine_client.generate(
                engine_input, sampling_params, request_id):
            result = res
        if result is None or not result.outputs:
            raise BackendError("the engine returned no output")
        if getattr(result.outputs[0], "finish_reason", None) == "error":
            raise BackendError("the engine reported a generation error")
        return result

    async def restricted_read(self, engine_input, token_ids: list[int],
                              request_id: str
                              ) -> tuple[dict[int, float], Any]:
        """One forward pass returning the next-token logprob of exactly
        `token_ids` (however small; never a top-k window). Returns
        (logprob by token id, the RequestOutput). Token ids the engine
        didn't report are absent from the map; the caller decides."""
        from vllm.sampling_params import SamplingParams
        params = SamplingParams(
            max_tokens=1, temperature=0.0, n=1,
            logprobs=len(token_ids),
            logprob_token_ids=list(token_ids))
        result = await self.generate(engine_input, params, request_id)
        logprobs = result.outputs[0].logprobs
        if not logprobs:
            raise BackendError(
                "no logprobs returned: the engine rejected "
                "logprob_token_ids (speculative decoding enabled?)")
        pos0 = logprobs[0]
        return ({t: pos0[t].logprob for t in token_ids if t in pos0},
                result)

    async def pool(self, prompt_ids: list[int], pooling_params,
                   request_id: str):
        """Run one pooling request (token_classify models); return its
        final output."""
        encode = getattr(self._serving.engine_client, "encode", None)
        if encode is None:
            raise BackendError(
                "this server exposes no pooling (encode) path; serve the "
                "checkpoint as a pooling model")
        last = None
        async for out in encode({"prompt_token_ids": list(prompt_ids)},
                                pooling_params, request_id):
            last = out
        if last is None or not getattr(last, "finished", True):
            raise BackendError("the pooling read did not complete")
        return last
