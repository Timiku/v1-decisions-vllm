"""The API is open for extension without core edits (offline).

Each test adds something the way a third-party package would, through
the public registries only, and checks the whole request path picks it
up:

- a new question type (answered on /v1/decisions, refused on the Jev wire);
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
from vllm.entrypoints.generate.decisions.protocol import DecisionsRequest
from vllm.entrypoints.generate.decisions.question_types import (
    QuestionModel, QuestionType, register_question_type)
from vllm.entrypoints.generate.decisions.serving import ServingDecisions
from vllm.entrypoints.generate.decisions.systemone_protocol import (
    SystemOneRequest)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse

from test_unify import _Engine, _Models, _Renderer, _fake_backend


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
    r = _run(s.create_decisions(DecisionsRequest(**SENTIMENT)))
    a = r["answers"]["tone"]
    assert a["type"] == "sentiment"
    assert a["label"] == "positive"
    assert a["polarity"] == pytest.approx(
        a["probabilities"]["positive"] - a["probabilities"]["negative"])
    assert set(a) == {"type", "probabilities", "label", "polarity",
                      "confidence", "extra"}


def test_plugin_question_type_is_not_on_the_jev_wire():
    with pytest.raises(ValidationError, match="unknown type 'sentiment' "
                       "on the Jev wire"):
        SystemOneRequest(model="m", **SENTIMENT)


def test_plugin_question_type_validates_its_own_fields():
    with pytest.raises(ValidationError, match="tone"):
        DecisionsRequest(state="s", questions={"tone": {
            "type": "sentiment", "instructions": "i", "criteria": {}}})


def test_unregistered_type_is_refused():
    with pytest.raises(ValidationError, match="unknown type 'ranking'"):
        DecisionsRequest(state="s", questions={"q": {
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
    r = _run(s.create_decisions(DecisionsRequest(
        **TWO, backend_options={"boost": math.log(3)})))
    assert r["answers"]["q"]["probabilities"]["a"] == pytest.approx(0.75)


def test_backend_options_are_validated_by_the_backend():
    s = _toy_serving()
    r = _run(s.create_decisions(DecisionsRequest(
        **TWO, backend_options={"bost": 1.0})))
    assert isinstance(r, ErrorResponse) and r.error.code == 422
    assert "toy" in r.error.message and "bost" in r.error.message


def test_backend_without_options_refuses_them():
    s = _serving(_fake_backend())
    r = _run(s.create_decisions(DecisionsRequest(
        **TWO, backend_options={"readout": "direct"})))
    assert isinstance(r, ErrorResponse) and r.error.code == 422
    assert "takes no backend_options" in r.error.message


def test_backend_facts_land_in_extra_backend_and_audit():
    s = _toy_serving()
    a = _run(s.create_decisions(DecisionsRequest(**TWO)))["answers"]["q"]
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
    r = _run(s.create_decisions(DecisionsRequest(
        **SENTIMENT, backend_options={"boost": 2.0})))
    assert r["answers"]["tone"]["label"] == "positive"


# ---------------------------------------------------------------------
# per-block extra
# ---------------------------------------------------------------------

def test_extra_per_block():
    s = _toy_serving()
    a = _run(s.create_decisions(DecisionsRequest(
        **TWO, extra={"audit": "none", "backend": "basic"})))["answers"]["q"]
    assert set(a["extra"]) == {"backend"}
    assert "option_logits" not in a["extra"]["backend"]
    assert a["extra"]["backend"]["toy_fact"] == 42


def test_extra_block_left_out_of_the_map_is_full():
    s = _toy_serving()
    a = _run(s.create_decisions(DecisionsRequest(
        **TWO, extra={"backend": "none"})))["answers"]["q"]
    assert set(a["extra"]) == {"audit"}


@pytest.mark.parametrize("bad", [{"trace": "full"}, {"audit": "some"},
                                 "summary", 3])
def test_extra_refuses_unknown_blocks_and_levels(bad):
    with pytest.raises(ValidationError):
        DecisionsRequest(**TWO, extra=bad)


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
