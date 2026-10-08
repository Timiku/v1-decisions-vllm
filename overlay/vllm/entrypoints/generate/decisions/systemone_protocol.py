# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SystemOne (Jev-compatible) wire: the Jev request and response shapes.

`/v1/systemone` answers through the same path as `/v1/decisions`: the
request is validated here (Jev's fields, any registered question type),
plus the same `extra` block as `/v1/decisions`. A Jev type's answer is
reduced to its `jev_fields` and its `extra`; a plugin type's answer comes
back whole. Nothing is computed differently for this wire.

Differences from the hosted Jev API are listed in the README under
"Differences from the Jev API".
"""
from __future__ import annotations

import time
from typing import Any

from pydantic import Field, field_validator, model_validator

from vllm.utils import random_uuid

from .openai_protocol import DecisionsExtra
from .protocol import OpenAIBaseModel, check_question_count
from .question_types import parse_questions

# Settings that are not Jev fields: on this wire they go under `extra`.
# Sending one at the top level is an error rather than a silently ignored
# field.
EXTRA_ONLY_FIELDS = ("calibration_temperature", "seed", "backend_options")


class SystemOneRequest(OpenAIBaseModel):
    model: str = Field(..., description='Any model id; "jev-latest" and '
                       "other aliases resolve to a versioned Jev id, any "
                       "other string is echoed.")
    state: str | dict | list = Field(
        ..., description="The content to evaluate.")
    questions: dict[str, Any] = Field(
        ..., description="Caller-chosen id -> typed question: Jev's types "
        "or any registered plugin type.")
    backend: str | None = Field(
        default=None, description="Readout backend override (logit | "
        "encoder | canvas | <plugin>) for every question in the request.")
    extra: DecisionsExtra = Field(
        default_factory=DecisionsExtra,
        description="Settings that are not Jev fields, as on "
        "/v1/decisions; answers carry `extra` at its `detail`.")

    @field_validator("questions", mode="before")
    @classmethod
    def _parse_questions(cls, v):
        return parse_questions(v)

    @field_validator("questions")
    @classmethod
    def _question_count(cls, v):
        return check_question_count(v)

    @model_validator(mode="before")
    @classmethod
    def _reject_top_level_settings(cls, data):
        if isinstance(data, dict):
            for name in EXTRA_ONLY_FIELDS:
                if name in data:
                    raise ValueError(
                        f"{name} is not a Jev field; send it under "
                        "`extra`")
        return data

    @model_validator(mode="after")
    def _backend_in_one_place(self):
        if self.backend is not None and self.extra.backend is not None:
            raise ValueError("send backend at the top level or under "
                             "`extra`, not both")
        return self


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
        "score, legend, probabilities, confidence), plus `extra` at the "
        "request's detail. A plugin type's answer comes back whole.")
    usage: SystemOneUsage
    partial_failures: dict[str, str] | None = Field(
        default=None, description="qid -> error for questions that failed; "
        "absent when every question succeeded.")
