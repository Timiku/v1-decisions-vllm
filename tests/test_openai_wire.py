"""/v1/decisions speaks OpenAI's Decisions format (PLAN-5), offline.

- the request: OpenAI's fields, validated the way vLLM's draft (#60465)
  validates them, plus our `extra` block;
- the mapping onto Jev's question types renders exactly as the same Jev
  question does (the saved calibration stays valid);
- the response: OpenAI's answer list, refusals, usage, `extra`;
- /v1/systemone gives the same answers for the same questions, and takes
  the same `extra` block.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
from pydantic import ValidationError

from vllm.entrypoints.generate.decisions.backends import restricted_softmax
from vllm.entrypoints.generate.decisions.compile import compile_question
from vllm.entrypoints.generate.decisions.limits import (
    DecisionLimits, set_limits_for_tests)
from vllm.entrypoints.generate.decisions.openai_protocol import (
    DecisionsRequest)
from vllm.entrypoints.generate.decisions.protocol import DecisionsQuery
from vllm.entrypoints.generate.decisions.systemone_protocol import (
    SystemOneRequest)
from vllm.entrypoints.generate.decisions.systemone_serving import (
    ServingSystemOne)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse

sys.path.insert(0, os.path.dirname(__file__))
from test_unify import _fake_backend, _make, _renders, _run  # noqa: E402


@pytest.fixture(autouse=True)
def default_limits():
    set_limits_for_tests(DecisionLimits())
    yield
    set_limits_for_tests(DecisionLimits())


BODY = {
    "model": "m",
    "input": "Ticket: payouts failing for 3 days.",
    "questions": [
        {"type": "predicate", "name": "urgent",
         "instructions": "Is it urgent?"},
        {"type": "choice", "name": "team", "instructions": "Which team?",
         "choices": [{"value": "billing", "description": "Payments"},
                     {"value": "technical"},
                     {"value": True, "description": "Either"}]},
        {"type": "score", "name": "mood", "instructions": "How frustrated?",
         "levels": [{"label": "Calm"},
                    {"label": "Angry", "description": "shouting"}]},
    ],
}


def _req(**kw):
    return DecisionsRequest(**{**BODY, **kw})


def _q(*questions):
    return {**BODY, "questions": list(questions)}


PRED = {"type": "predicate", "instructions": "i"}


# ---------------------------------------------------------------------
# request: same as #60465
# ---------------------------------------------------------------------

def test_minimal_request():
    r = DecisionsRequest(model="m", input="x", questions=[PRED])
    assert r.questions[0].name is None


@pytest.mark.parametrize("body", [
    {**BODY, "state": "x"},                              # unknown field
    {**BODY, "calibration_temperature": 0.5},            # belongs in extra
    {k: v for k, v in BODY.items() if k != "model"},     # model required
    {**BODY, "questions": []},                           # no questions
    _q({**PRED, "name": None}),                          # explicit null name
    _q({**PRED, "criteria": "x"}),                       # unknown question key
    _q({"type": "noul", "instructions": "i"}),           # Jev type
    _q({"type": "choice", "instructions": "i",
        "choices": [{"value": "a"}]}),                   # one choice
    _q({"type": "choice", "instructions": "i",
        "choices": [{"value": "a"}, {"value": "a"}]}),   # duplicate
    _q({"type": "choice", "instructions": "i",
        "choices": [{"value": "a"}, {"value": 1}]}),     # not str/bool
    _q({"type": "score", "instructions": "i",
        "levels": [{"label": f"l{i}"} for i in range(11)]}),  # > 10
    {**BODY, "safety_identifier": "x" * 129},
    {**BODY, "input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,AA"}]}]},
    {**BODY, "input": [{"role": "assistant", "content": "x"}]},
    {**BODY, "extra": {"temperature": 1.0}},             # unknown extra key
    {**BODY, "extra": {"detail": "summary"}},
])
def test_invalid_requests_rejected(body):
    with pytest.raises(ValidationError):
        DecisionsRequest(**body)


def test_bool_and_string_values_are_distinct():
    r = DecisionsRequest(**_q({"type": "choice", "instructions": "i",
                               "choices": [{"value": "true"},
                                           {"value": True}]}))
    assert [c.value for c in r.questions[0].choices] == ["true", True]


def test_images_named_in_the_error():
    with pytest.raises(ValidationError, match="images"):
        DecisionsRequest(**{**BODY, "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:,"}]}]})


def test_input_join_matches_upstream():
    r = _req(input=[
        {"role": "user", "content": "first"},
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "a"},
            {"type": "input_text", "text": "b"}]}])
    assert r.to_query().state == "first\n\na\nb"


def test_safety_identifier_accepted():
    assert _req(safety_identifier="u-1").safety_identifier == "u-1"


def test_duplicate_names_allowed():
    # like #60465: answers come back in order, so names need not be unique
    DecisionsRequest(**_q({**PRED, "name": "x"}, {**PRED, "name": "x"}))


# ---------------------------------------------------------------------
# request: our additions
# ---------------------------------------------------------------------

@pytest.mark.parametrize("k", [27, 100, 255])
def test_choice_past_26(k):
    DecisionsRequest(**_q({"type": "choice", "instructions": "i",
                           "choices": [{"value": f"o{i}"}
                                       for i in range(k)]}))


def test_choice_over_255_rejected():
    with pytest.raises(ValidationError):
        DecisionsRequest(**_q({"type": "choice", "instructions": "i",
                               "choices": [{"value": f"o{i}"}
                                           for i in range(256)]}))


def test_server_limits_apply():
    set_limits_for_tests(DecisionLimits(max_options=5, max_questions=2))
    with pytest.raises(ValidationError,
                       match="VLLM_TYPED_DECISIONS_MAX_OPTIONS"):
        DecisionsRequest(**_q({"type": "choice", "instructions": "i",
                               "choices": [{"value": f"o{i}"}
                                           for i in range(6)]}))
    with pytest.raises(ValidationError,
                       match="VLLM_TYPED_DECISIONS_MAX_QUESTIONS"):
        _req()


def test_extra_fields_reach_the_query():
    q = _req(extra={"calibration_temperature": 2.0, "backend": "logit",
                    "backend_options": {"readout": "auto"}, "seed": 3,
                    "detail": "basic"}).to_query()
    assert (q.calibration_temperature, q.backend, q.backend_options,
            q.seed) == (2.0, "logit", {"readout": "auto"}, 3)
    assert q.extra == {"audit": "basic", "backend": "basic"}


def test_detail_defaults_to_full():
    assert _req().to_query().extra == {"audit": "full", "backend": "full"}


# ---------------------------------------------------------------------
# mapping onto Jev's types: identical renders
# ---------------------------------------------------------------------

JEV = {"state": "Ticket: payouts failing for 3 days.", "questions": {
    "0": {"type": "noul", "instructions": "Is it urgent?"},
    "1": {"type": "choice", "instructions": "Which team?",
          "criteria": {'"billing"': "Payments",
                       '"technical"': "technical",
                       "true": "Either"}},
    "2": {"type": "score", "instructions": "How frustrated?",
          "criteria": ["Calm", "Angry: shouting"]}}}


def test_maps_to_jev_questions():
    q = _req().to_query()
    assert q.questions == DecisionsQuery(**JEV).questions


def test_renders_like_the_jev_question():
    s = _make()
    q = _req().to_query()
    assert _renders(s, {"state": q.state, "questions": q.questions}) \
        == _renders(s, JEV)


def test_predicate_renders_true_first():
    q = compile_question(_req().to_query(), "0")
    assert [(o.id, o.description) for o in q.options] == [
        ("true", "Yes"), ("false", "No")]


# ---------------------------------------------------------------------
# response
# ---------------------------------------------------------------------

def _logits(ids):
    return {oid: -0.7 * i for i, oid in enumerate(ids)}


def _respond(body=None, fail=(), **kw):
    s = _make(backend=_fake_backend(fail=fail))
    return _run(s.create_decisions(DecisionsRequest(**{**(body or BODY),
                                                       **kw})))


def test_answers_in_order_with_names():
    r = _respond()
    assert [(a["type"], a["name"]) for a in r["answers"]] == [
        ("predicate", "urgent"), ("choice", "team"), ("score", "mood")]
    assert r["model"] == "m"
    assert set(r) == {"model", "answers", "usage", "extra"}


def test_model_aliases_resolve():
    assert _respond(model="jev-latest")["model"] == "jev-1.13.0"


def test_name_is_null_when_not_given():
    r = _respond(_q(PRED))
    assert r["answers"][0]["name"] is None


def test_predicate_answer():
    a = _respond()["answers"][0]
    want = restricted_softmax({"true": 0.0, "false": -0.7})["true"]
    assert a["probability"] == pytest.approx(want)
    assert set(a) == {"type", "name", "probability", "extra"}


def test_choice_answer_keeps_value_types():
    a = _respond()["answers"][1]
    p = restricted_softmax(_logits(["a", "b", "c"]))
    assert a["choice"] == "billing"
    assert a["probabilities"] == [
        {"value": "billing", "probability": pytest.approx(p["a"])},
        {"value": "technical", "probability": pytest.approx(p["b"])},
        {"value": True, "probability": pytest.approx(p["c"])}]
    assert 0 < a["confidence"] < 1
    assert set(a) == {"type", "name", "choice", "probabilities",
                      "confidence", "extra"}


def test_choice_bool_wins():
    body = _q({"type": "choice", "instructions": "i",
               "choices": [{"value": False}, {"value": "x"}]})
    a = _respond(body)["answers"][0]
    assert a["choice"] is False


def test_score_answer():
    a = _respond()["answers"][2]
    p = restricted_softmax(_logits(["0", "1"]))
    assert a["score"] == pytest.approx(p["1"])
    assert a["probabilities"] == [
        {"value": 0, "label": "Calm", "probability": pytest.approx(p["0"])},
        {"value": 1, "label": "Angry",
         "probability": pytest.approx(p["1"])}]
    assert "legend" not in a


def test_extra_carries_audit_and_backend():
    a = _respond()["answers"][1]
    assert set(a["extra"]) == {"audit", "backend"}
    assert a["extra"]["audit"]["confidence_formula"] == "normalized-peak-v1"


def test_detail_none_gives_plain_answers():
    r = _respond(extra={"detail": "none"})
    assert all("extra" not in a for a in r["answers"])


def test_response_extra_has_id_and_created():
    x = _respond()["extra"]
    assert x["id"].startswith("decisions-") and isinstance(x["created"], int)


def test_failed_question_is_a_refusal():
    r = _respond(fail={"1"})
    a = r["answers"][1]
    assert set(a) == {"type", "name", "extra"}
    assert (a["type"], a["name"]) == ("refusal", "team")
    assert "boom in 1" in a["extra"]["error"]
    assert r["answers"][0]["type"] == "predicate"


def test_refusal_keeps_error_at_detail_none():
    r = _respond(fail={"1"}, extra={"detail": "none"})
    assert "boom" in r["answers"][1]["extra"]["error"]


def test_all_failed_is_an_error():
    r = _respond(fail={"0", "1", "2"})
    assert isinstance(r, ErrorResponse) and r.error.code == 400
    assert "'urgent'" in r.error.message


def test_all_failed_unnamed_uses_position():
    r = _respond(_q(PRED), fail={"0"})
    assert isinstance(r, ErrorResponse)
    assert "question 0 " in r.error.message


def test_usage_shape():
    u = _respond()["usage"]
    assert u == {"input_tokens": 30,
                 "input_tokens_details": {"cached_tokens": 0,
                                          "cache_write_tokens": 0},
                 "output_tokens": 0,
                 "output_tokens_details": {"reasoning_tokens": 0},
                 "total_tokens": 30}


def test_response_is_json():
    json.dumps(_respond())


# ---------------------------------------------------------------------
# /v1/systemone: same answers, same extra block
# ---------------------------------------------------------------------

def test_systemone_gives_the_same_answers():
    name = _fake_backend()
    dec = _run(_make(backend=name).create_decisions(_req()))
    sys1 = _run(_make(ServingSystemOne, backend=name).create_systemone(
        SystemOneRequest(model="m", **JEV))).model_dump()
    assert dec["answers"][0]["probability"] == sys1["answers"]["0"]["noul"]
    assert dec["answers"][1]["probabilities"][0]["probability"] == \
        sys1["answers"]["1"]["probabilities"]['"billing"']
    assert dec["answers"][2]["score"] == sys1["answers"]["2"]["score"]
    for i in range(3):
        assert dec["answers"][i]["confidence" if i else "probability"] \
            is not None
        assert dec["answers"][i]["extra"]["audit"] == \
            sys1["answers"][str(i)]["extra"]["audit"]


def test_systemone_takes_extra():
    name = _fake_backend()
    s = _make(ServingSystemOne, backend=name)
    r = _run(s.create_systemone(SystemOneRequest(
        model="m", **JEV, extra={"calibration_temperature": 2.0,
                                 "detail": {"backend": "none"}})))
    a = r.model_dump()["answers"]["1"]
    assert a["extra"]["audit"]["calibration_temperature"] == 2.0
    assert "backend" not in a["extra"]
    plain = _run(s.create_systemone(SystemOneRequest(
        model="m", **JEV, extra={"detail": "none"}))).model_dump()
    assert all("extra" not in a for a in plain["answers"].values())


def test_systemone_default_detail_is_full():
    r = _run(_make(ServingSystemOne, backend=_fake_backend())
             .create_systemone(SystemOneRequest(model="m", **JEV)))
    assert set(r.model_dump()["answers"]["0"]["extra"]) == {"audit",
                                                             "backend"}


def test_systemone_refuses_top_level_settings():
    for field, value in (("calibration_temperature", 0.5), ("seed", 1),
                         ("backend_options", {})):
        with pytest.raises(ValidationError, match="extra"):
            SystemOneRequest(model="m", **JEV, **{field: value})


def test_systemone_backend_in_one_place():
    with pytest.raises(ValidationError, match="backend"):
        SystemOneRequest(model="m", **JEV, backend="logit",
                         extra={"backend": "logit"})


# ---------------------------------------------------------------------
# the OpenAI SDK parses the response (skipped when the SDK is missing)
# ---------------------------------------------------------------------

def test_openai_sdk_parses_the_response():
    decision = pytest.importorskip("openai.types.decision")
    r = _respond(fail={"1"})
    parsed = decision.Decision.model_validate(r)
    assert [a.type for a in parsed.answers] == [
        "predicate", "refusal", "score"]
