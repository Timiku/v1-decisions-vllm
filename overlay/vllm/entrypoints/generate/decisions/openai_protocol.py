# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""/v1/decisions request: OpenAI's Decisions format, plus `extra`.

The fields and their checks follow OpenAI's Decisions API, and where
OpenAI leaves something open (how input is joined, what is refused),
vLLM's own draft of it (vllm-project/vllm#60465). Differences, all
additive, are listed in the README under "Differences from OpenAI and
from upstream":

- `extra` carries the settings OpenAI has no field for (calibration
  temperature, backend, backend options, seed, detail);
- a choice question takes up to 255 choices (the server's
  VLLM_TYPED_DECISIONS_MAX_OPTIONS), not 26.

Each OpenAI question becomes one of Jev's question types (predicate ->
noul without criteria, choice -> choice, score -> score), so it renders
exactly as the same Jev question does and the saved calibration applies.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import (ConfigDict, Field, PrivateAttr, StrictBool, StrictStr,
                      field_validator, model_validator)

from .protocol import DecisionsQuery, OpenAIBaseModel, normalize_extra

ShortText = Annotated[StrictStr, Field(max_length=1048576)]
InputText = Annotated[StrictStr, Field(max_length=10485760)]
ChoiceValue = StrictStr | StrictBool


class _Strict(OpenAIBaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------
# input
# ---------------------------------------------------------------------

class InputTextPart(_Strict):
    type: Literal["input_text"]
    text: InputText


class InputMessage(_Strict):
    role: Literal["user"]
    content: InputText | Annotated[list[InputTextPart],
                                   Field(max_length=16384)]
    type: Literal["message"] = "message"

    @model_validator(mode="before")
    @classmethod
    def _reject_images(cls, data: Any) -> Any:
        if (isinstance(data, dict)
                and isinstance(data.get("content"), list)
                and any(isinstance(p, dict)
                        and p.get("type") == "input_image"
                        for p in data["content"])):
            raise ValueError("text input only; images are not supported "
                             "yet")
        return data


def input_text(value: str | list[InputMessage]) -> str:
    """Parts joined with "\\n", messages with "\\n\\n" (as #60465)."""
    if isinstance(value, str):
        return value
    return "\n\n".join(
        m.content if isinstance(m.content, str)
        else "\n".join(p.text for p in m.content)
        for m in value)


# ---------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------

class _QuestionBase(_Strict):
    instructions: ShortText
    name: ShortText | None = None

    @model_validator(mode="after")
    def _name_not_null(self):
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("name must be a string when provided")
        return self


class PredicateQuestion(_QuestionBase):
    type: Literal["predicate"]


class ChoiceOption(_Strict):
    value: ChoiceValue
    description: ShortText = ""


class ChoiceQuestion(_QuestionBase):
    type: Literal["choice"]
    choices: list[ChoiceOption] = Field(min_length=2, max_length=255)

    @model_validator(mode="after")
    def _distinct(self):
        values = [(type(c.value), c.value) for c in self.choices]
        if len(set(values)) != len(values):
            raise ValueError("choice values must be distinct")
        return self


class ScoreLevel(_Strict):
    label: ShortText
    description: ShortText = ""


class ScoreQuestion(_QuestionBase):
    type: Literal["score"]
    levels: list[ScoreLevel] = Field(min_length=2, max_length=10)


Question = Annotated[PredicateQuestion | ChoiceQuestion | ScoreQuestion,
                     Field(discriminator="type")]


def option_id(value: str | bool) -> str:
    """A choice's option id: its JSON text, so "true" and true differ."""
    return json.dumps(value, ensure_ascii=False)


def value_text(value: str | bool) -> str:
    """What a choice without a description renders as."""
    return value if isinstance(value, str) else option_id(value)


def to_jev_question(q: PredicateQuestion | ChoiceQuestion | ScoreQuestion
                    ) -> dict:
    """The Jev question that renders this one."""
    if isinstance(q, PredicateQuestion):
        return {"type": "noul", "instructions": q.instructions}
    if isinstance(q, ChoiceQuestion):
        return {"type": "choice", "instructions": q.instructions,
                "criteria": {option_id(c.value):
                             c.description or value_text(c.value)
                             for c in q.choices}}
    return {"type": "score", "instructions": q.instructions,
            "criteria": [f"{lv.label}: {lv.description}" if lv.description
                         else lv.label for lv in q.levels]}


# ---------------------------------------------------------------------
# extra and the request
# ---------------------------------------------------------------------

class DecisionsExtra(_Strict):
    """Settings OpenAI's format has no field for. Shared by both wires."""

    calibration_temperature: float | None = Field(
        default=None, gt=0.0,
        description="T for this request: probabilities = softmax(logits "
        "/ T). Omitted: the server default.")
    backend: str | None = Field(
        default=None, description="Readout backend: logit | encoder | "
        "canvas | <plugin>. Omitted: the server's startup backend.")
    backend_options: dict[str, Any] | None = Field(
        default=None, description="Per-request settings for the backend, "
        "validated by that backend.")
    seed: int | None = Field(
        default=None, ge=0, description="Seeds backends that sample "
        "(canvas); ignored by the others.")
    detail: str | dict[str, str] = Field(
        default="full", description="How much of each answer's `extra` "
        "to return: full | basic | none, or a map per block, e.g. "
        "{\"audit\": \"full\", \"backend\": \"none\"}.")

    @field_validator("detail")
    @classmethod
    def _detail(cls, v):
        normalize_extra(v)
        return v

    def query_fields(self) -> dict:
        """These settings as DecisionsQuery fields."""
        return {"calibration_temperature": self.calibration_temperature,
                "backend": self.backend,
                "backend_options": self.backend_options,
                "seed": self.seed,
                "extra": self.detail}


class DecisionsRequest(_Strict):
    """POST /v1/decisions. Unknown fields are refused, as in #60465;
    everything that is ours goes under `extra`."""

    model: ShortText
    input: InputText | Annotated[list[InputMessage],
                                 Field(max_length=131072)]
    questions: list[Question] = Field(min_length=1, max_length=200)
    safety_identifier: Annotated[StrictStr,
                                 Field(max_length=128)] | None = None
    extra: DecisionsExtra = Field(default_factory=DecisionsExtra)

    _query: DecisionsQuery = PrivateAttr()

    @model_validator(mode="after")
    def _build_query(self):
        # Built here so the server's limits (question count, options per
        # question) refuse the request as a validation error.
        self._query = DecisionsQuery(
            model=self.model, state=input_text(self.input),
            questions={str(i): to_jev_question(q)
                       for i, q in enumerate(self.questions)},
            **self.extra.query_fields())
        return self

    def to_query(self) -> DecisionsQuery:
        """The internal query; question i is under id str(i)."""
        return self._query

    def label(self, qid: str) -> str:
        """How an error names question `qid`: its name, else its index."""
        name = self.questions[int(qid)].name
        return repr(name) if name is not None else qid
