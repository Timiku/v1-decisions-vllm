# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SystemOne (Jev-compatible) wire: the Jev request and response shapes.

`/v1/systemone` is a projection of `/v1/decisions`: the request is
validated here (Jev's fields, and only the question types marked `jev`
in the registry), answered by the decisions path, and each answer is
reduced to its type's `jev_fields`. Nothing is computed differently for
this wire.

Differences from the hosted Jev API are listed in the README under
"Differences from the Jev API".
"""
from __future__ import annotations

import time
from typing import Any

from pydantic import Field, field_validator, model_validator

from vllm.utils import random_uuid

from .protocol import OpenAIBaseModel, check_question_count
from .question_types import parse_questions

# /v1/decisions request fields that are not part of the Jev wire. Sending
# one here is an error rather than a silently ignored field.
DECISIONS_ONLY_FIELDS = ("calibration_temperature", "seed",
                         "backend_options", "extra", "question", "options",
                         "qtype")


class SystemOneRequest(OpenAIBaseModel):
    model: str = Field(..., description='Any model id; "jev-latest" and '
                       "other aliases resolve to a versioned Jev id, any "
                       "other string is echoed.")
    state: str | dict | list = Field(
        ..., description="The content to evaluate.")
    questions: dict[str, Any] = Field(
        ..., description="Caller-chosen id -> typed question (Jev's types "
        "only).")
    backend: str | None = Field(
        default=None, description="Readout backend override (logit | "
        "encoder | canvas | <plugin>) for every question in the request.")

    @field_validator("questions", mode="before")
    @classmethod
    def _parse_questions(cls, v):
        return parse_questions(v, jev_only=True)

    @field_validator("questions")
    @classmethod
    def _question_count(cls, v):
        return check_question_count(v)

    @model_validator(mode="before")
    @classmethod
    def _reject_decisions_fields(cls, data):
        if isinstance(data, dict):
            for name in DECISIONS_ONLY_FIELDS:
                if name in data:
                    raise ValueError(
                        f"{name} is not part of the Jev wire; use "
                        "/v1/decisions")
        return data


class SystemOneUsage(OpenAIBaseModel):
    input_tokens: int
    output_tokens: int = 0


class SystemOneResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"systemone-{random_uuid()}")
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    answers: dict[str, dict[str, Any]] = Field(
        ..., description="Each answer: `type` plus its type's Jev fields "
        "(noul: noul; choice: choice, probabilities, confidence; score: "
        "score, legend, probabilities, confidence).")
    usage: SystemOneUsage
    partial_failures: dict[str, str] | None = Field(
        default=None, description="qid -> error for questions that failed; "
        "absent when every question succeeded.")
