# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""/v1/systemone: a projection of /v1/decisions.

The Jev request is answered by `create_decisions` unchanged, then each
answer is reduced to its question type's `jev_fields`. No readout,
calibration or failure handling of its own: whatever /v1/decisions
returns for the same questions, this returns minus `extra` and the
decisions-only envelope fields. tests/test_unify.py holds that as the
contract.
"""
from __future__ import annotations

from vllm.entrypoints.generate.decisions.protocol import (
    EXTRA_BLOCKS, DecisionsRequest)
from vllm.entrypoints.generate.decisions.question_types import (
    get_question_type)
from vllm.entrypoints.generate.decisions.serving import ServingDecisions
from vllm.entrypoints.generate.decisions.systemone_protocol import (
    SystemOneRequest,
    SystemOneResponse,
    SystemOneUsage,
)

try:
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse
except ImportError:  # locked fork moved engine protocol to openai
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse


def project_answer(answer: dict) -> dict:
    """A /v1/decisions answer -> the Jev answer: `type` plus the type's
    Jev fields."""
    qt = get_question_type(answer["type"])
    return {"type": answer["type"],
            **{k: answer[k] for k in qt.jev_fields}}


def project_response(decisions: dict) -> SystemOneResponse:
    """A /v1/decisions response dict -> the Jev response."""
    return SystemOneResponse(
        id="systemone-" + decisions["id"].removeprefix("decisions-"),
        created=decisions["created"],
        model=decisions["model"],
        answers={qid: project_answer(a)
                 for qid, a in decisions["answers"].items()},
        usage=SystemOneUsage(
            input_tokens=decisions["usage"]["input_tokens"]),
        partial_failures=decisions.get("partial_failures"),
    )


class ServingSystemOne(ServingDecisions):
    """Jev-compatible endpoint over the decisions path."""

    async def create_systemone(
        self,
        request: SystemOneRequest,
        raw_request=None,
    ) -> SystemOneResponse | ErrorResponse:
        # Already validated as a Jev body; the calibration temperature is
        # the server default because the Jev wire cannot carry one, and
        # the extra blocks are never built into the response.
        decisions_request = DecisionsRequest.model_construct(
            model=request.model, state=request.state,
            questions=request.questions, backend=request.backend,
            backend_options=None, calibration_temperature=None, seed=None,
            extra={b: "none" for b in EXTRA_BLOCKS})
        result = await self.create_decisions(decisions_request, raw_request)
        if isinstance(result, ErrorResponse):
            return result
        return project_response(result)
