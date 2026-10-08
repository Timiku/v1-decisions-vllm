# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""/v1/systemone: a projection of /v1/decisions.

The Jev request is answered by `answer_query` unchanged, then each Jev
answer is reduced to its question type's `jev_fields` (a plugin type's
answer is returned whole). No readout,
calibration or failure handling of its own: whatever /v1/decisions
returns for the same questions, this returns minus `extra` and the
decisions-only envelope fields. tests/test_unify.py holds that as the
contract.
"""
from __future__ import annotations

from vllm.entrypoints.generate.decisions.protocol import (
    DecisionsQuery, normalize_extra)
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
    """An internal answer -> the Jev answer: `type`, the type's Jev
    fields, and `extra` when the request's detail kept any. A type that
    is not one of Jev's has no Jev shape, so its answer is returned
    whole."""
    qt = get_question_type(answer["type"])
    if not qt.jev:
        return dict(answer)
    out = {"type": answer["type"],
           **{k: answer[k] for k in qt.jev_fields}}
    if "extra" in answer:
        out["extra"] = answer["extra"]
    return out


def project_response(decisions: dict) -> SystemOneResponse:
    """An `answer_query` result -> the Jev response."""
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
        # Already validated as a Jev body; the settings come from `extra`.
        fields = request.extra.query_fields()
        fields["backend"] = request.backend or fields["backend"]
        fields["extra"] = normalize_extra(fields["extra"])
        query = DecisionsQuery.model_construct(
            model=request.model, state=request.state,
            questions=request.questions, **fields)
        result = await self.answer_query(query, raw_request)
        if isinstance(result, ErrorResponse):
            return result
        return project_response(result)
