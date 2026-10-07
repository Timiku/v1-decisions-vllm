# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The internal decisions query, and the `extra` settings both wires share.

`DecisionsQuery` is what the serving path answers: a state plus a map of
typed questions (Jev's noul / choice / score are the built-in types from
`question_types.py`; plugins add more) and the request-level settings.
It is not a wire model: `/v1/decisions` (OpenAI's format,
`openai_protocol.py`) and `/v1/systemone` (Jev's format,
`systemone_protocol.py`) each turn their request into one.

Every limit comes from `limits.py`, so both wires share one set of caps.

`CompiledQuestion` is the internal, non-wire unit a backend reads: one
question compiled against the request's state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import Field, field_validator, model_validator

try:
    from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel
except ImportError:  # locked fork moved engine protocol to openai
    from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel
from vllm.utils import random_uuid

from .question_types import parse_questions

# `model` aliases: matching ids resolve to the versioned Jev id in the
# response; any other id is echoed unchanged. The wire never rejects a
# model name.
MODEL_ALIASES: dict[str, str] = {
    "jev-latest": "jev-1.13.0",
    "jev-preview": "jev-1.13.0",
}

# The blocks of an answer's `extra`, and how much of each to return.
EXTRA_BLOCKS = ("audit", "backend")
EXTRA_LEVELS = ("full", "basic", "none")


class DecisionOption(OpenAIBaseModel):
    id: str = Field(..., description="Stable option identifier, returned "
                    "verbatim.")
    description: str = Field(..., min_length=1,
                             description="What this option means.")


def check_question_count(v: dict | None) -> dict | None:
    from .limits import MAX_QUESTIONS_ENV, get_limits
    if v is None:
        return v
    cap = get_limits().max_questions
    if not 1 <= len(v) <= cap:
        raise ValueError(f"questions supports 1-{cap} entries (got "
                         f"{len(v)}; {MAX_QUESTIONS_ENV})")
    return v


def normalize_extra(value: Any) -> dict[str, str]:
    """`extra` as one level for every block ("basic") or a per-block map
    ({"audit": "full", "backend": "none"}); blocks left out of a map get
    "full". Returns the per-block map."""
    if isinstance(value, str):
        if value not in EXTRA_LEVELS:
            raise ValueError(f"extra must be one of {EXTRA_LEVELS} or a "
                             f"map of block -> level (got {value!r})")
        return {b: value for b in EXTRA_BLOCKS}
    if isinstance(value, dict):
        unknown = set(value) - set(EXTRA_BLOCKS)
        if unknown:
            raise ValueError(f"unknown extra block(s) {sorted(unknown)} "
                             f"(blocks: {', '.join(EXTRA_BLOCKS)})")
        bad = {k: v for k, v in value.items() if v not in EXTRA_LEVELS}
        if bad:
            raise ValueError(f"extra levels must be one of {EXTRA_LEVELS} "
                             f"(got {bad})")
        return {b: value.get(b, "full") for b in EXTRA_BLOCKS}
    raise ValueError("extra must be a level string or a map of block -> "
                     "level")


# ---------------------------------------------------------------------
# the internal query
# ---------------------------------------------------------------------

class DecisionsQuery(OpenAIBaseModel):
    """One query for the serving path, built by either wire. Unknown
    fields inside a question are rejected."""

    model: str | None = Field(
        default=None, description="Echoed in the response, aliases "
        "resolved. Omitted: the served model name.")
    state: str | dict | list = Field(
        ..., description="The evidence; a string or JSON rendered "
        "verbatim.")
    questions: dict[str, Any] = Field(
        ..., description="Caller-chosen id -> typed question (any "
        "registered question type); answers return under the same ids.")

    calibration_temperature: float | None = Field(
        default=None, gt=0.0,
        description="T for this request: probabilities = softmax(logits "
        "/ T). Omitted: the server default "
        "(VLLM_TYPED_DECISIONS_TEMPERATURE, default 1.0).")
    backend: str | None = Field(
        default=None, description="Readout backend: logit | encoder | "
        "canvas | <plugin>. Omitted: the server's startup backend.")
    backend_options: dict[str, Any] | None = Field(
        default=None, description="Per-request settings for the backend, "
        "validated by that backend (logit: readout; canvas: samples, "
        "max_steps). A backend that takes none refuses any.")
    seed: int | None = Field(
        default=None, ge=0, description="Seeds backends that sample "
        "(canvas); ignored by the others.")
    extra: str | dict[str, str] = Field(
        default="full", validate_default=True, description="How much of each answer's `extra` "
        "to return: one level for every block (full | basic | none) or a "
        "map per block, e.g. {\"audit\": \"full\", \"backend\": "
        "\"none\"}. full: everything. basic: without per-option lists. "
        "none: leave the block out.")

    @field_validator("questions", mode="before")
    @classmethod
    def _parse_questions(cls, v):
        return parse_questions(v)

    @field_validator("questions")
    @classmethod
    def _question_count(cls, v):
        return check_question_count(v)

    @field_validator("extra")
    @classmethod
    def _extra(cls, v):
        return normalize_extra(v)


# ---------------------------------------------------------------------
# internal: one compiled question
# ---------------------------------------------------------------------

@dataclass
class CompiledQuestion:
    """One question compiled against the request state: what a backend
    reads. Not a wire model; built by compile.compile_question (and by
    backends for their own sub-reads, e.g. two-stage).

    `backend_options` is whatever the request sent under that name,
    already validated by the backend's `options_model` when it has one."""

    state: str | dict | list
    question: str
    options: list[DecisionOption]
    qtype: str | None = None
    model: str | None = None
    seed: int | None = None
    backend: str | None = None
    backend_options: Any = None
    request_id: str = field(default_factory=random_uuid)
