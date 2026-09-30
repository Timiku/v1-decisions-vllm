# SPDX-License-Identifier: Apache-2.0
"""EncoderBackend: token-classification pooling models (Laya and kin).

Serves decision reads from checkpoints registered as `token_classify`
pooling models (PR #58429's Laya). Each question is one deterministic
pooling pass. Prompts come from the checkpoint's own format
(`laya_prompts.build_prompt` for Laya), because the encoder was trained
on it; for other checkpoints the standard decision prompt is used, with
a warning.

Claims `LayaForDecision` checkpoints, so a Laya model is served with this
backend without configuration.
"""

from __future__ import annotations

import json
import os

from vllm.logger import init_logger

from ..protocol import CompiledQuestion
from . import BackendError, BackendResult, cached_tokens, restricted_softmax

logger = init_logger(__name__)

LAYA_ARCHITECTURE = "LayaForDecision"
# the question types Laya's head was trained on
LAYA_TYPES = ("noul", "choice", "score")

# Warned at most once per process: a startup misconfiguration, not a
# per-request event.
_no_prompt_builder_warned = False


def laya_prompt_builder(host, truncate_state: bool = False):
    """Prompt builder for LayaForDecision checkpoints, reading max_len and
    head_max_len from the checkpoint's laya_config (hf_config first, then
    laya_config.json next to the weights). None for non-Laya models."""
    if LAYA_ARCHITECTURE not in host.architectures:
        return None
    from . import laya_prompts

    mc = host.model_config
    laya_cfg = getattr(getattr(mc, "hf_config", None), "laya_config",
                       None) or {}
    if not laya_cfg:
        cfg_path = os.path.join(mc.model, "laya_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                laya_cfg = json.load(f)
    if "max_len" not in laya_cfg or "head_max_len" not in laya_cfg:
        raise ValueError(
            f"checkpoint {mc.model!r} carries no laya_config with max_len "
            "and head_max_len; the encoder backend refuses to guess them")
    tokenizer = host.tokenizer

    def build(question: CompiledQuestion) -> tuple[list[int], int]:
        if question.qtype not in LAYA_TYPES:
            raise BackendError(
                f"the Laya encoder answers {', '.join(LAYA_TYPES)} "
                f"questions, not {question.qtype!r}")
        q = {
            "type": question.qtype,
            "instructions": question.question,
            "criteria": (
                {o.id: o.description for o in question.options}
                if question.qtype != "score" else
                [o.description for o in question.options]),
        }
        laya_prompts.check_question(question.request_id, q)
        q = laya_prompts.normalize_question(q)
        # Build unbounded first and compare with max_len: Laya's builder
        # silently truncates; the default here refuses instead.
        full = laya_prompts.build_prompt(tokenizer, question.state, q,
                                         10**9, laya_cfg["head_max_len"])
        if len(full) > laya_cfg["max_len"] and not truncate_state:
            raise BackendError(
                f"state exceeds the encoder's max_len={laya_cfg['max_len']} "
                "tokens; refused, never truncated (set truncate_state=true "
                "for the upstream behaviour)")
        ids = full[: laya_cfg["max_len"]]
        return ids, len(ids)

    return build


class EncoderBackend:
    calibrate_by_default = False
    """Reads decisions from a token_classify pooling model."""

    name = "encoder"
    architectures = (LAYA_ARCHITECTURE,)
    options_model = None

    def __init__(self, host, truncate_state: bool = False,
                 prompt_builder=None):
        self.host = host
        self.truncate_state = truncate_state
        self.prompt_builder = (prompt_builder
                               or laya_prompt_builder(host, truncate_state))
        global _no_prompt_builder_warned
        if self.prompt_builder is None and not _no_prompt_builder_warned:
            _no_prompt_builder_warned = True
            logger.warning(
                "EncoderBackend without a checkpoint prompt builder: reads "
                "will use the standard decision prompt. If the encoder was "
                "trained on its own prompt format, accuracy will suffer.")

    async def read(self, question: CompiledQuestion,
                   request_id: str) -> BackendResult:
        if self.prompt_builder is not None:
            prompt_ids, input_tokens = self.prompt_builder(question)
        else:
            prompt = await self.host.render(question)
            prompt_ids, input_tokens = prompt.prompt_ids, prompt.input_tokens

        from vllm.pooling_params import PoolingParams
        out = await self.host.pool(
            prompt_ids, PoolingParams(task="token_classify",
                                      use_activation=False), request_id)
        rows = self._extract_rows(out)
        option_ids = [o.id for o in question.options]
        if len(rows) < len(option_ids):
            raise BackendError(
                f"pooler returned {len(rows)} rows for {len(option_ids)} "
                "options", option_ids[len(rows):])
        # Laya orders noul rows false-then-true; compiled noul options are
        # true-then-false.
        if question.qtype == "noul" and option_ids[0] == "true":
            rows = rows[::-1]
        logits_map = {oid: float(row[0])
                      for oid, row in zip(option_ids, rows)}
        return BackendResult(
            logits_map, restricted_softmax(logits_map), forward_passes=1,
            meta={"input_tokens": input_tokens,
                  "cached_input_tokens": cached_tokens(out),
                  "prompt_source": ("checkpoint" if self.prompt_builder
                                    else "standard"),
                  "truncate_state": self.truncate_state})

    @staticmethod
    def _extract_rows(out) -> list[list[float]]:
        """PoolingRequestOutput -> list of per-row score lists."""
        data = getattr(out, "outputs", None)
        if data is None:
            raise BackendError("pooling output carried no data")
        for attr in ("probs", "data"):
            v = getattr(data, attr, None)
            if v is not None:
                return [list(map(float, row)) for row in v]
        try:
            return [list(map(float, row)) for row in data]
        except (TypeError, ValueError) as e:
            raise BackendError(f"unrecognized pooling output shape: {e!r}")
