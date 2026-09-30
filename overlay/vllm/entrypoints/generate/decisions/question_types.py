# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Question types: the pluggable definition of what a question is.

A question type owns four things:

- its **wire shape** (`model`): the JSON a caller sends under `questions`,
  a pydantic model with `type: Literal[<name>]` and `instructions`;
- its **options** (`options(question)`): the closed set of answers the
  model chooses among. Every type reduces to a probability distribution
  over a closed option set; that is the one thing backends rely on;
- its **answer** (`answer(probabilities, options)`): the type-specific
  answer fields built from that distribution (`noul`, `choice`, ...);
- its **Jev projection** (`jev`, `jev_fields`): whether `/v1/systemone`
  accepts it, and which answer fields that wire returns.

Jev's three types (noul, choice, score) are the built-in entries. A new
type registers itself with `register_question_type` (directly, or from a
package advertising the `vllm.decision_question_types` entry-point group)
and is then accepted by `/v1/decisions` with no change to the core.
"""
from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field, ValidationError, field_validator

from vllm.logger import init_logger

try:
    from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel
except ImportError:  # locked fork moved engine protocol to openai
    from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel

logger = init_logger(__name__)


# ---------------------------------------------------------------------
# the base every question type builds on
# ---------------------------------------------------------------------

class QuestionModel(OpenAIBaseModel):
    """Wire shape shared by every question type. A misspelled key inside
    a question changes the prompt, so unknown fields are refused."""

    model_config = {"extra": "forbid"}

    type: str
    instructions: str | dict | list = Field(
        ..., description="What the model should decide. Structured objects "
        "are rendered verbatim as JSON.")


class QuestionType:
    """Subclass, set the class attributes, implement `options` and
    `answer`, then `register_question_type(MyType())`."""

    name: ClassVar[str]
    model: ClassVar[type[QuestionModel]]
    jev: ClassVar[bool] = False
    # answer keys /v1/systemone returns besides `type` (jev types only)
    jev_fields: ClassVar[tuple[str, ...]] = ()

    def options(self, question: QuestionModel) -> list:
        """The closed option set: a list of protocol.DecisionOption."""
        raise NotImplementedError

    def answer(self, probabilities: dict[str, float],
               options: list) -> dict[str, Any]:
        """Type-specific answer fields, from the calibrated distribution
        over option ids (in option order)."""
        raise NotImplementedError


def flatten(value: Any) -> str:
    """Criteria/instructions values may be string | object | array | null;
    structured values are JSON-serialized verbatim, None becomes ""."""
    import json
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _option(id_: str, description: str):
    from .protocol import DecisionOption
    return DecisionOption(id=id_, description=description)


# ---------------------------------------------------------------------
# Jev's three types
# ---------------------------------------------------------------------

class NoulCriteria(OpenAIBaseModel):
    model_config = {"extra": "forbid"}

    true: str | dict | list | None = None
    false: str | dict | list | None = None


class NoulQuestion(QuestionModel):
    type: Literal["noul"]
    criteria: NoulCriteria | None = None


class Noul(QuestionType):
    """Yes/no. Answer: `noul` = P(true). True always renders first."""

    name = "noul"
    model = NoulQuestion
    jev = True
    jev_fields = ("noul",)

    def options(self, question):
        c = question.criteria
        return [_option("true", (flatten(c.true) if c else "") or "Yes"),
                _option("false", (flatten(c.false) if c else "") or "No")]

    def answer(self, probabilities, options):
        return {"noul": probabilities.get("true", 0.0)}


class ChoiceQuestion(QuestionModel):
    type: Literal["choice"]
    criteria: dict[str, str | dict | list | None] = Field(
        ..., description="Option id -> description; null when the id "
        "speaks for itself.")

    @field_validator("criteria")
    @classmethod
    def _option_count(cls, v: dict) -> dict:
        from .limits import MAX_OPTIONS_ENV, get_limits
        cap = get_limits().max_options
        if not 2 <= len(v) <= cap:
            raise ValueError(
                f"choice supports 2-{cap} options (got {len(v)}; "
                f"{MAX_OPTIONS_ENV})")
        return v


class Choice(QuestionType):
    """One of N caller-named options. Answer: `choice` = most likely id."""

    name = "choice"
    model = ChoiceQuestion
    jev = True
    jev_fields = ("choice", "probabilities", "confidence")

    def options(self, question):
        return [_option(key, flatten(desc) or key)
                for key, desc in question.criteria.items()]

    def answer(self, probabilities, options):
        return {"choice": max(probabilities, key=probabilities.get)}


class ScoreQuestion(QuestionModel):
    type: Literal["score"]
    criteria: list[str | dict | list] = Field(
        ..., description="Ordered level descriptions; level ids are "
        "'0'..'k-1'.")

    @field_validator("criteria")
    @classmethod
    def _level_count(cls, v: list) -> list:
        from .limits import MAX_SCORE_LEVELS_ENV, get_limits
        cap = get_limits().max_score_levels
        if not 2 <= len(v) <= cap:
            raise ValueError(
                f"score supports 2-{cap} levels (got {len(v)}; "
                f"{MAX_SCORE_LEVELS_ENV})")
        return v


class Score(QuestionType):
    """Ordered levels. Answer: `score` = sum(level * p), which can fall
    between levels, plus `legend` (level id -> description)."""

    name = "score"
    model = ScoreQuestion
    jev = True
    jev_fields = ("score", "legend", "probabilities", "confidence")

    def options(self, question):
        # level indices as ids give Jev's score wire shape directly
        return [_option(str(i), flatten(desc))
                for i, desc in enumerate(question.criteria)]

    def answer(self, probabilities, options):
        return {
            "score": sum(i * probabilities.get(str(i), 0.0)
                         for i in range(len(options))),
            "legend": {o.id: o.description for o in options},
        }


# ---------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------

_TYPES: dict[str, QuestionType] = {}


def register_question_type(qt: QuestionType) -> None:
    """Make a question type available by its `name`. Re-registering an
    existing name is refused."""
    name = getattr(qt, "name", None)
    if not name:
        raise ValueError(f"{type(qt).__qualname__} needs a `name`")
    if name in _TYPES:
        raise ValueError(f"question type {name!r} is already registered")
    if not issubclass(qt.model, QuestionModel):
        raise ValueError(f"question type {name!r}: `model` must subclass "
                         "QuestionModel")
    _TYPES[name] = qt


for _builtin in (Noul(), Choice(), Score()):
    register_question_type(_builtin)


def get_question_type(name: str) -> QuestionType:
    try:
        return _TYPES[name]
    except KeyError:
        raise ValueError(f"unknown question type {name!r} (registered: "
                         f"{', '.join(sorted(_TYPES))})") from None


def question_type_names(jev_only: bool = False) -> list[str]:
    return [n for n, qt in _TYPES.items() if qt.jev or not jev_only]


def parse_questions(value: Any, jev_only: bool = False) -> dict:
    """Validate a `questions` map: each entry by its registered type.
    Already-parsed QuestionModel instances pass through."""
    if not isinstance(value, dict):
        raise ValueError("questions must be an object of id -> question")
    allowed = question_type_names(jev_only)
    out = {}
    for qid, q in value.items():
        if isinstance(q, QuestionModel):
            out[qid] = q
            continue
        if not isinstance(q, dict):
            raise ValueError(f"question {qid!r} must be an object")
        t = q.get("type")
        if t not in allowed:
            where = " on the Jev wire" if jev_only else ""
            raise ValueError(f"question {qid!r}: unknown type {t!r}{where} "
                             f"(one of: {', '.join(allowed)})")
        try:
            out[qid] = _TYPES[t].model.model_validate(q)
        except ValidationError as e:
            raise ValueError(f"question {qid!r}: {e}") from None
    return out


def load_question_type_plugins() -> None:
    """Load question types advertised via the `vllm.decision_question_types`
    entry-point group. Each entry resolves to a callable that calls
    `register_question_type`. A failing plugin is logged and skipped."""
    from importlib.metadata import entry_points
    for ep in entry_points(group="vllm.decision_question_types"):
        try:
            ep.load()()
        except Exception:  # noqa: BLE001 - plugin isolation
            logger.exception("question type plugin %r failed to load",
                             ep.name)
