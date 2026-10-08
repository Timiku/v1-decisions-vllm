"""The API is open for extension without core edits (offline).

Each test adds something the way a third-party package would, through
the public registries only, and checks the whole request path picks it
up:

- a new question type (answered on /v1/decisions and /v1/systemone);
- a new backend that claims a model architecture, takes per-request
  options, and reports its own facts under extra.backend;
- per-block `extra` control;
- canvas per-request options.
"""
from __future__ import annotations

import asyncio
import math
from typing import ClassVar, Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from vllm.entrypoints.generate.decisions.backends import (
    BackendResult, register_backend, restricted_softmax)
from vllm.entrypoints.generate.decisions.limits import (
    DecisionLimits, set_limits_for_tests)
from vllm.entrypoints.generate.decisions.protocol import DecisionsQuery
from vllm.entrypoints.generate.decisions.question_types import (
    QuestionModel, QuestionType, register_question_type)
from vllm.entrypoints.generate.decisions.serving import ServingDecisions
from vllm.entrypoints.generate.decisions.openai_protocol import (
    DecisionsRequest)
from vllm.entrypoints.generate.decisions.systemone_protocol import (
    SystemOneRequest)
from vllm.entrypoints.generate.decisions.systemone_serving import (
    ServingSystemOne)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse

from test_unify import _Engine, _Models, _Renderer, _fake_backend, _make


@pytest.fixture(autouse=True)
def default_limits():
    set_limits_for_tests(DecisionLimits())
    yield
    set_limits_for_tests(DecisionLimits())


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------
# a plugin question type
# ---------------------------------------------------------------------

class SentimentQuestion(QuestionModel):
    type: Literal["sentiment"]


class Sentiment(QuestionType):
    """Three fixed options; answers with the label and a polarity."""

    name = "sentiment"
    model = SentimentQuestion
    jev = False

    def options(self, question):
        from vllm.entrypoints.generate.decisions.protocol import (
            DecisionOption)
        return [DecisionOption(id=i, description=i)
                for i in ("positive", "neutral", "negative")]

    def answer(self, probabilities, options):
        return {"label": max(probabilities, key=probabilities.get),
                "polarity": probabilities["positive"]
                - probabilities["negative"]}


register_question_type(Sentiment())

SENTIMENT = {"state": "I love it", "questions": {
    "tone": {"type": "sentiment", "instructions": "How does it sound?"}}}


def _serving(backend: str):
    return ServingDecisions(_Engine(), _Models(), _Renderer(),
                            request_logger=None, default_backend=backend)


def test_plugin_question_type_is_answered():
    s = _serving(_fake_backend())
    r = _run(s.answer_query(DecisionsQuery(**SENTIMENT)))
    a = r["answers"]["tone"]
    assert a["type"] == "sentiment"
    assert a["label"] == "positive"
    assert a["polarity"] == pytest.approx(
        a["probabilities"]["positive"] - a["probabilities"]["negative"])
    assert set(a) == {"type", "probabilities", "label", "polarity",
                      "confidence", "extra"}


TONE = {"type": "sentiment", "instructions": "How does it sound?"}


def test_plugin_question_type_on_the_jev_wire():
    s = _make(ServingSystemOne, backend=_fake_backend())
    r = _run(s.create_systemone(SystemOneRequest(model="m", **SENTIMENT)))
    a = r.model_dump()["answers"]["tone"]
    # no Jev shape for a plugin type: the whole answer comes back
    assert set(a) == {"type", "probabilities", "label", "polarity",
                      "confidence", "extra"}
    assert a["type"] == "sentiment"


def test_plugin_question_type_on_the_openai_wire():
    req = DecisionsRequest(model="m", input="I love it", questions=[
        {**TONE, "name": "tone"},
        {"type": "predicate", "instructions": "Is it a review?"}])
    r = _run(_make(backend=_fake_backend()).create_decisions(req))
    a = r["answers"][0]
    assert a["name"] == "tone" and a["type"] == "sentiment"
    assert set(a) == {"name", "type", "probabilities", "label", "polarity",
                      "confidence", "extra"}
    assert r["answers"][1]["type"] == "predicate"


def test_plugin_question_same_answer_on_both_wires():
    name = _fake_backend()
    dec = _run(_make(backend=name).create_decisions(DecisionsRequest(
        model="m", input="I love it", questions=[TONE])))
    jev = _run(_make(ServingSystemOne, backend=name).create_systemone(
        SystemOneRequest(model="m", state="I love it",
                         questions={"0": TONE}))).model_dump()
    a, b = dec["answers"][0], jev["answers"]["0"]
    assert a["probabilities"] == b["probabilities"]
    assert a["label"] == b["label"]


def test_plugin_question_name_on_the_openai_wire():
    with pytest.raises(ValidationError, match="name must be a string"):
        DecisionsRequest(model="m", input="x",
                         questions=[{**TONE, "name": None}])
    req = DecisionsRequest(model="m", input="x", questions=[TONE])
    assert req.questions[0].name is None


def test_plugin_question_fields_checked_on_the_openai_wire():
    with pytest.raises(ValidationError, match="sentiment question"):
        DecisionsRequest(model="m", input="x",
                         questions=[{**TONE, "criteria": {}}])


def test_unknown_type_on_the_openai_wire():
    with pytest.raises(ValidationError,
                       match="unknown question type 'ranking'.*sentiment"):
        DecisionsRequest(model="m", input="x", questions=[
            {"type": "ranking", "instructions": "i"}])


def test_jev_types_stay_off_the_openai_wire():
    # predicate is OpenAI's yes/no; Jev's noul is not a plugin
    with pytest.raises(ValidationError, match="unknown question type "
                       "'noul'"):
        DecisionsRequest(model="m", input="x", questions=[
            {"type": "noul", "instructions": "i"}])


@pytest.mark.parametrize("name", ["predicate", "refusal"])
def test_openai_type_names_are_reserved(name):
    class Clash(Sentiment):
        pass
    Clash.name = name
    with pytest.raises(ValueError, match="reserved"):
        register_question_type(Clash())


def test_openai_sdk_reads_a_plugin_answer():
    # The SDK builds responses leniently: an unknown answer type keeps
    # its type and fields.
    pytest.importorskip("openai.types.decision")
    from openai._models import construct_type
    from openai.types.decision import Decision
    req = DecisionsRequest(model="m", input="I love it", questions=[TONE])
    r = _run(_make(backend=_fake_backend()).create_decisions(req))
    a = construct_type(type_=Decision, value=r).answers[0]
    assert a.type == "sentiment" and a.label == r["answers"][0]["label"]


def test_plugin_question_type_validates_its_own_fields():
    with pytest.raises(ValidationError, match="tone"):
        DecisionsQuery(state="s", questions={"tone": {
            "type": "sentiment", "instructions": "i", "criteria": {}}})


def test_unregistered_type_is_refused():
    with pytest.raises(ValidationError, match="unknown type 'ranking'"):
        DecisionsQuery(state="s", questions={"q": {
            "type": "ranking", "instructions": "i"}})


def test_duplicate_question_type_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        register_question_type(Sentiment())


# ---------------------------------------------------------------------
# a plugin backend: claims an architecture, takes options, reports facts
# ---------------------------------------------------------------------

class ToyOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    boost: float = 0.0


class ToyBackend:
    """Favours the first option by `boost`; reports its own facts."""

    name = "toy"
    architectures = ("ToyForDecision",)
    options_model = ToyOptions

    def __init__(self, host, **kwargs):
        self.host = host

    async def read(self, question, request_id):
        opts = question.backend_options or ToyOptions()
        ids = [o.id for o in question.options]
        logits = {oid: (opts.boost if i == 0 else 0.0)
                  for i, oid in enumerate(ids)}
        return BackendResult(logits, restricted_softmax(logits), 1,
                             meta={"input_tokens": 5,
                                   "cached_input_tokens": 2,
                                   "readout": "toy-read",
                                   "toy_fact": 42})


register_backend("toy", ToyBackend)

TWO = {"state": "s", "questions": {"q": {
    "type": "choice", "instructions": "i",
    "criteria": {"a": "A", "b": "B"}}}}


class _ToyEngine(_Engine):
    class model_config:  # noqa: N801 - stands in for ModelConfig
        max_model_len = 10**6
        max_logprobs = 600
        architectures = ["ToyForDecision"]


def _toy_serving():
    # no VLLM_TYPED_DECISIONS_BACKEND: the wiring passes default_backend=None
    return ServingDecisions(_ToyEngine(), _Models(), _Renderer(),
                            request_logger=None, default_backend=None)


def test_backend_claiming_the_architecture_is_selected():
    assert _toy_serving().decision_backend.name == "toy"


def test_explicit_backend_beats_the_claim():
    s = ServingDecisions(_ToyEngine(), _Models(), _Renderer(),
                         request_logger=None, default_backend="logit")
    assert s.decision_backend.name == "logit"


def test_duplicate_backend_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        register_backend("toy", ToyBackend)


def test_backend_options_reach_the_backend():
    s = _toy_serving()
    r = _run(s.answer_query(DecisionsQuery(
        **TWO, backend_options={"boost": math.log(3)})))
    assert r["answers"]["q"]["probabilities"]["a"] == pytest.approx(0.75)


def test_backend_options_are_validated_by_the_backend():
    s = _toy_serving()
    r = _run(s.answer_query(DecisionsQuery(
        **TWO, backend_options={"bost": 1.0})))
    assert isinstance(r, ErrorResponse) and r.error.code == 422
    assert "toy" in r.error.message and "bost" in r.error.message


def test_backend_without_options_refuses_them():
    s = _serving(_fake_backend())
    r = _run(s.answer_query(DecisionsQuery(
        **TWO, backend_options={"readout": "direct"})))
    assert isinstance(r, ErrorResponse) and r.error.code == 422
    assert "takes no backend_options" in r.error.message


def test_backend_facts_land_in_extra_backend_and_audit():
    s = _toy_serving()
    a = _run(s.answer_query(DecisionsQuery(**TWO)))["answers"]["q"]
    backend, audit = a["extra"]["backend"], a["extra"]["audit"]
    assert backend["name"] == "toy"
    assert backend["toy_fact"] == 42
    assert set(backend["option_logits"]) == {"a", "b"}
    # the standard keys are lifted into audit, not duplicated
    assert audit["readout"] == "toy-read"
    assert audit["input_tokens"] == 5 and audit["cached_input_tokens"] == 2
    assert "readout" not in backend and "input_tokens" not in backend


def test_a_plugin_type_on_a_plugin_backend():
    s = _toy_serving()
    r = _run(s.answer_query(DecisionsQuery(
        **SENTIMENT, backend_options={"boost": 2.0})))
    assert r["answers"]["tone"]["label"] == "positive"


# ---------------------------------------------------------------------
# per-block extra
# ---------------------------------------------------------------------

def test_extra_per_block():
    s = _toy_serving()
    a = _run(s.answer_query(DecisionsQuery(
        **TWO, extra={"audit": "none", "backend": "basic"})))["answers"]["q"]
    assert set(a["extra"]) == {"backend"}
    assert "option_logits" not in a["extra"]["backend"]
    assert a["extra"]["backend"]["toy_fact"] == 42


def test_extra_block_left_out_of_the_map_is_full():
    s = _toy_serving()
    a = _run(s.answer_query(DecisionsQuery(
        **TWO, extra={"backend": "none"})))["answers"]["q"]
    assert set(a["extra"]) == {"audit"}


@pytest.mark.parametrize("bad", [{"trace": "full"}, {"audit": "some"},
                                 "summary", 3])
def test_extra_refuses_unknown_blocks_and_levels(bad):
    with pytest.raises(ValidationError):
        DecisionsQuery(**TWO, extra=bad)


# ---------------------------------------------------------------------
# canvas per-request options
# ---------------------------------------------------------------------

def test_canvas_samples_per_request():
    from test_decisions_offline import _canvas_request, _FakeEngine, \
        _FakeServing
    from vllm.entrypoints.generate.decisions.backends.canvas_backend import (
        CanvasBackend, CanvasOptions)
    from vllm.entrypoints.generate.decisions.backends.host import (
        BackendHost)
    backend = CanvasBackend(BackendHost(_FakeServing(_FakeEngine())),
                            samples=1)
    q = _canvas_request(seed=3,
                        backend_options=CanvasOptions(samples=4))
    result = _run(backend.read(q, "r"))
    assert result.forward_passes == 4
    assert result.meta["samples"]["n"] == 4
    with pytest.raises(ValidationError):
        CanvasOptions(samples=0)
