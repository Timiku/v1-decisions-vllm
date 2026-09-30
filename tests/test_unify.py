"""The unification contract (PLAN-4 section 2), offline through the stubs.

- golden: every pinned wire body renders byte-identically to the
  pre-unify code (tests/golden/renders-8bd7268.json);
- one form: the one-question shorthand and a one-entry `questions` map
  compile, render and answer identically;
- one set of limits for every form and both endpoints;
- temperature: server default, request override, applied once;
- partial failures, 422s, model echo, removed fields;
- /v1/systemone == project(/v1/decisions).
"""
from __future__ import annotations

import asyncio
import json
import math
import hashlib
import os

import pytest
from pydantic import ValidationError

from vllm.entrypoints.generate.decisions.backends import (
    BackendError, BackendResult, register_backend, restricted_softmax)
from vllm.entrypoints.generate.decisions.compile import compile_question
from vllm.entrypoints.generate.decisions.limits import (
    DecisionLimits, set_limits_for_tests)
from vllm.entrypoints.generate.decisions.protocol import DecisionsRequest
from vllm.entrypoints.generate.decisions.serving import ServingDecisions
from vllm.entrypoints.generate.decisions.systemone_protocol import (
    SystemOneRequest)
from vllm.entrypoints.generate.decisions.systemone_serving import (
    ServingSystemOne, project_response)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse

from golden.cases import CASES
from vllm.entrypoints.generate.decisions.compile import RENDER_VERSION

GOLDEN = os.path.join(os.path.dirname(__file__), "golden",
                      "renders-8bd7268.json")


@pytest.fixture(autouse=True)
def default_limits():
    set_limits_for_tests(DecisionLimits())
    yield
    set_limits_for_tests(DecisionLimits())


# ---------------------------------------------------------------------
# fake engine + renderer: records the messages handed to the renderer
# ---------------------------------------------------------------------

class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text] or [0]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _BaseRenderer:
    def __init__(self):
        self.seen = []

    def get_tokenizer(self):
        return _Tok()

    async def render_chat_async(self, conversations, chat_params,
                                tok_params):
        self.seen.append(conversations[0])
        text = ("\n".join(m["content"] for m in conversations[0])
                + "\nASSISTANT:")
        return None, [{"prompt_token_ids":
                       self.get_tokenizer().encode(text)}]


class _Renderer:
    chat_template = None
    chat_template_content_format = "string"

    def __init__(self):
        self.renderer = _BaseRenderer()


class _ModelConfig:
    max_model_len = 10**6
    max_logprobs = 600
    architectures = []


class _Engine:
    model_config = _ModelConfig()
    errored = False
    dead_error = RuntimeError("engine dead")


class _Models:
    def model_name(self, _):
        return "served-model"


def _make(cls=ServingDecisions, backend="logit"):
    return cls(_Engine(), _Models(), _Renderer(), request_logger=None,
               default_backend=backend)


_FAKE_COUNT = 0


def _fake_backend(fail=(), logits_for=None):
    """Register a backend that returns fixed logits per option position
    and fails the listed qids. Returns its name."""
    global _FAKE_COUNT
    _FAKE_COUNT += 1
    name = f"unify-fake-{_FAKE_COUNT}"
    fail = set(fail)

    class _Fake:
        def __init__(self, serving, **kw):
            pass

        async def read(self, question, request_id):
            qid = request_id.rsplit("-", 1)[-1]
            if qid in fail:
                raise BackendError(f"boom in {qid}")
            ids = [o.id for o in question.options]
            logits = (logits_for(ids) if logits_for else
                      {oid: -0.7 * i for i, oid in enumerate(ids)})
            return BackendResult(logits, restricted_softmax(logits), 1,
                                 meta={"input_tokens": 10})

    _Fake.name = name
    register_backend(name, _Fake)
    return name


def _run(coro):
    return asyncio.run(coro)


def _renders(serving, body):
    req = DecisionsRequest(**body)
    out = []
    for qid in req.questions:
        q = compile_question(req, qid)
        serving.renderer.renderer.seen.clear()
        build = _run(serving._build_prompt(q))
        assert isinstance(build, tuple), build
        out.append({"messages": serving.renderer.renderer.seen[0],
                    "slot_ids": list(build[2]),
                    "option_ids": [o.id for o in q.options]})
    return out


# ---------------------------------------------------------------------
# golden renders
# ---------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(CASES))
def test_render_matches_pre_unify_golden(case):
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)["cases"][case]
    assert _renders(_make(), CASES[case]) == golden


# ---------------------------------------------------------------------
# one form
# ---------------------------------------------------------------------

SHORT = {"state": "s", "question": "Which?",
         "options": [{"id": "a", "description": "Alpha"},
                     {"id": "b", "description": "Beta"},
                     {"id": "c", "description": "Gamma"}]}
TYPED = {"state": "s", "questions": {"decision": {
    "type": "choice", "instructions": "Which?",
    "criteria": {"a": "Alpha", "b": "Beta", "c": "Gamma"}}}}


def test_shorthand_expands_to_one_entry_questions():
    assert (DecisionsRequest(**SHORT).questions
            == DecisionsRequest(**TYPED).questions)


def test_shorthand_and_typed_render_identically():
    s = _make()
    assert _renders(s, SHORT) == _renders(s, TYPED)


def test_shorthand_and_typed_answer_identically():
    name = _fake_backend()
    s = _make(backend=name)
    a = _run(s.create_decisions(DecisionsRequest(**SHORT)))
    b = _run(s.create_decisions(DecisionsRequest(**TYPED)))
    for r in (a, b):
        r.pop("id"), r.pop("created")
    assert a == b


def test_shorthand_noul_score_inference():
    noul = DecisionsRequest(state="s", question="q", options=[
        {"id": "false", "description": "no"},
        {"id": "true", "description": "yes"}])
    q = noul.questions["decision"]
    assert q.type == "noul"
    # noul always renders true first, whatever order the options came in
    assert [o.id for o in compile_question(noul, "decision").options] \
        == ["true", "false"]
    score = DecisionsRequest(state="s", question="q", qtype="score",
                             options=[{"id": "0", "description": "lo"},
                                      {"id": "1", "description": "hi"}])
    assert score.questions["decision"].criteria == ["lo", "hi"]


@pytest.mark.parametrize("body", [
    {**SHORT, "questions": TYPED["questions"]},          # both forms
    {"state": "s", "question": "q"},                     # no options
    {**SHORT, "options": [{"id": "a", "description": "A"},
                          {"id": "a", "description": "B"}]},  # dup ids
    {**SHORT, "qtype": "noul"},                          # bad noul ids
    {**SHORT, "qtype": "score"},                         # bad score ids
    {**SHORT, "qtype": "ranking"},                       # unknown qtype
    {"state": "s", "questions": {}},                     # empty
    {"state": "s", "questions": {"q": {"type": "choice",
        "instructions": "i", "criteria": {"a": "A", "b": "B"},
        "critera": "typo"}}},                            # typo in question
])
def test_invalid_bodies_rejected(body):
    with pytest.raises(ValidationError):
        DecisionsRequest(**body)


def test_unknown_top_level_field_ignored():
    req = DecisionsRequest(**{**TYPED, "some_future_field": 1})
    assert list(req.questions) == ["decision"]


# ---------------------------------------------------------------------
# one set of limits
# ---------------------------------------------------------------------

def _choice(k):
    return {"state": "s", "questions": {"q": {
        "type": "choice", "instructions": "i",
        "criteria": {f"o{i}": f"d{i}" for i in range(k)}}}}


@pytest.mark.parametrize("k", [17, 26, 100, 255])
def test_typed_choice_accepts_up_to_max_options(k):
    DecisionsRequest(**_choice(k))
    SystemOneRequest(model="m", **_choice(k))


def test_over_max_options_names_the_setting():
    for model, extra in ((DecisionsRequest, {}),
                         (SystemOneRequest, {"model": "m"})):
        with pytest.raises(ValidationError,
                           match="VLLM_TYPED_DECISIONS_MAX_OPTIONS"):
            model(**_choice(256), **extra)


def test_limits_follow_settings():
    set_limits_for_tests(DecisionLimits(max_options=5, max_questions=2))
    with pytest.raises(ValidationError):
        DecisionsRequest(**_choice(6))
    with pytest.raises(ValidationError,
                       match="VLLM_TYPED_DECISIONS_MAX_QUESTIONS"):
        DecisionsRequest(state="s", questions={
            f"q{i}": {"type": "noul", "instructions": "i"}
            for i in range(3)})


def test_render_refuses_more_options_than_markers():
    s = _make()
    q = compile_question(DecisionsRequest(**_choice(27)), "q")
    result = _run(s._build_prompt(q))
    assert isinstance(result, ErrorResponse)
    assert "27 options exceed the 26" in result.error.message


# ---------------------------------------------------------------------
# temperature
# ---------------------------------------------------------------------

def _answer(s, body, qid="decision"):
    r = _run(s.create_decisions(DecisionsRequest(**body)))
    return r["answers"][qid]


def test_default_temperature_is_raw():
    # with no request value, no operator temperature, and
    # no calibration run, T=1.0 with source "default" ("never measured").
    s = _make(backend=_fake_backend())
    a = _answer(s, TYPED)
    prov = a["extra"]["audit"]
    assert prov["calibration_temperature"] == 1.0
    assert prov["temperature_source"] == "default"
    # raw distribution = softmax(option_logits); at T=1 it is the answer
    raw = restricted_softmax(a["extra"]["backend"]["option_logits"])
    assert a["probabilities"] == pytest.approx(raw)
    assert "raw_probabilities" not in prov and "dominance" not in prov


def test_server_default_applies_to_every_form_and_endpoint():
    set_limits_for_tests(DecisionLimits(temperature=0.5))
    name = _fake_backend()
    s = _make(backend=name)
    logits = {"a": 0.0, "b": -0.7, "c": -1.4}
    want = restricted_softmax(logits, temperature=0.5)
    for body in (SHORT, TYPED):
        a = _answer(s, body)
        assert a["extra"]["audit"]["temperature_source"] == "server"
        assert a["probabilities"] == pytest.approx(want)
    sys1 = _run(_make(ServingSystemOne, backend=name).create_systemone(
        SystemOneRequest(model="m", **TYPED)))
    assert sys1.answers["decision"]["probabilities"] == pytest.approx(want)


def test_request_temperature_beats_server_default():
    set_limits_for_tests(DecisionLimits(temperature=0.5))
    s = _make(backend=_fake_backend())
    a = _answer(s, {**TYPED, "calibration_temperature": 2.0})
    prov = a["extra"]["audit"]
    assert (prov["calibration_temperature"], prov["temperature_source"]) \
        == (2.0, "request")
    want = restricted_softmax({"a": 0.0, "b": -0.7, "c": -1.4},
                              temperature=2.0)
    assert a["probabilities"] == pytest.approx(want)


@pytest.mark.parametrize("t", [0.25, 1.0, 4.0])
def test_temperature_never_changes_the_choice(t):
    s = _make(backend=_fake_backend())
    a = _answer(s, {**TYPED, "calibration_temperature": t})
    assert a["choice"] == "a"


def test_bad_server_temperature_refused():
    with pytest.raises(ValueError):
        DecisionLimits(temperature=0.0)


# ---------------------------------------------------------------------
# response envelope, partial failures
# ---------------------------------------------------------------------

THREE = {"state": "s", "questions": {
    "urgent": {"type": "noul", "instructions": "Urgent?"},
    "dept": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "Payments", "technical": "Bugs"}},
    "mood": {"type": "score", "instructions": "Mood?",
             "criteria": ["Calm", "Annoyed", "Angry"]}}}


def test_envelope_fields():
    s = _make(backend=_fake_backend())
    r = _run(s.create_decisions(DecisionsRequest(**THREE)))
    assert r["id"].startswith("decisions-")
    assert r["object"] == "decisions"
    assert r["model"] == "served-model"
    assert "partial_failures" not in r
    # the fake backend reports no cache figure -> None, not a made-up 0
    assert r["usage"] == {"input_tokens": 30, "cached_input_tokens": None,
                          "output_tokens": 0}
    assert r["answers"]["mood"]["legend"] == {
        "0": "Calm", "1": "Annoyed", "2": "Angry"}
    # type-specific fields appear only on their type
    assert "legend" not in r["answers"]["dept"]
    assert "noul" not in r["answers"]["dept"]
    assert set(r["answers"]["urgent"]) == {
        "type", "probabilities", "noul", "confidence", "extra"}
    assert r["answers"]["urgent"]["extra"]["audit"][
        "served_model"] == "served-model"


def test_model_echo_resolves_aliases():
    s = _make(backend=_fake_backend())
    for sent, echoed in (("jev-latest", "jev-1.13.0"),
                         ("my-label", "my-label")):
        r = _run(s.create_decisions(DecisionsRequest(model=sent, **TYPED)))
        assert r["model"] == echoed


def test_partial_failure_keeps_other_answers():
    s = _make(backend=_fake_backend(fail={"dept"}))
    r = _run(s.create_decisions(DecisionsRequest(**THREE)))
    assert set(r["answers"]) == {"urgent", "mood"}
    assert "boom in dept" in r["partial_failures"]["dept"]
    assert r["usage"]["input_tokens"] == 20


def test_all_failed_names_first_in_request_order():
    s = _make(backend=_fake_backend(fail={"urgent", "dept", "mood"}))
    r = _run(s.create_decisions(DecisionsRequest(**THREE)))
    assert isinstance(r, ErrorResponse)
    assert "'urgent'" in r.error.message
    assert r.error.code == 400


def test_removed_fields_are_ignored_or_refused():
    # response_schema / label_prompt / prefixed_layout were removed: as
    # unknown top-level fields they are ignored on /v1/decisions ...
    req = DecisionsRequest(**TYPED, response_schema={"type": "object"},
                           label_prompt="letter",
                           prefixed_layout="letter-first")
    assert list(req.questions) == ["decision"]
    for name in ("response_schema", "label_prompt", "prefixed_layout"):
        assert name not in DecisionsRequest.model_fields
    # ... the prefixed readout is refused by the logit backend's options
    s = _make()
    r = _run(s.create_decisions(DecisionsRequest(**{
        **_choice(30), "backend_options": {"readout": "prefixed"}})))
    assert isinstance(r, ErrorResponse) and r.error.code == 422
    assert "readout" in r.error.message



# ---------------------------------------------------------------------
# systemone = project(decisions)
# ---------------------------------------------------------------------

def test_systemone_rejects_decisions_only_fields():
    for field, value in (("calibration_temperature", 0.5), ("seed", 1),
                         ("backend_options", {}), ("extra", "none"),
                         ("question", "q")):
        with pytest.raises(ValidationError, match="not part of the Jev"):
            SystemOneRequest(model="m", **THREE, **{field: value})


def test_systemone_is_a_projection_of_decisions():
    name = _fake_backend(fail={"dept"})
    dec = _run(_make(backend=name).create_decisions(
        DecisionsRequest(model="jev-latest", **THREE)))
    sys1 = _run(_make(ServingSystemOne, backend=name).create_systemone(
        SystemOneRequest(model="jev-latest", **THREE)))
    projected = project_response(dec).model_dump()
    got = sys1.model_dump()
    for r in (projected, got):
        r.pop("id"), r.pop("created")
    assert got == projected
    # every Jev answer field exists with the same value on decisions
    for qid, ans in sys1.model_dump()["answers"].items():
        for key, value in ans.items():
            assert dec["answers"][qid][key] == value, (qid, key)
    assert got["model"] == "jev-1.13.0"
    assert set(got["partial_failures"]) == {"dept"}


def test_systemone_all_failed_is_an_error():
    name = _fake_backend(fail={"urgent", "dept", "mood"})
    r = _run(_make(ServingSystemOne, backend=name).create_systemone(
        SystemOneRequest(model="m", **THREE)))
    assert isinstance(r, ErrorResponse)
    assert "'urgent'" in r.error.message


# ---------------------------------------------------------------------
# option_mass and cached-token reporting (real logit backend, fake engine)
# ---------------------------------------------------------------------

class _LP:
    def __init__(self, logprob):
        self.logprob = logprob


class _Out:
    def __init__(self, logprobs):
        self.logprobs = [logprobs]
        self.finish_reason = "length"


class _Result:
    def __init__(self, logprobs, cached):
        self.outputs = [_Out(logprobs)]
        self.num_cached_tokens = cached


class _GatherEngine(_Engine):
    """Answers a restricted gather: option token i gets logprob
    log(p_i) from `probs` (in marker order); every request reports
    `cached` prompt tokens served from the prefix cache."""

    def __init__(self, probs, cached):
        self.probs = probs
        self.cached = cached

    def generate(self, engine_input, params, request_id):
        async def gen():
            want = list(params.logprob_token_ids)
            yield _Result({t: _LP(math.log(self.probs[i]))
                           for i, t in enumerate(want)}, self.cached)
        return gen()


def _gather_serving(probs, cached, cls=ServingDecisions, tok=None):
    renderer = _Renderer()
    if tok is not None:
        renderer.renderer.get_tokenizer = lambda: tok
    s = cls(_GatherEngine(probs, cached), _Models(), renderer,
            request_logger=None, default_backend="logit")
    s.startup.slot_check = "ok"  # the self-check: test_answer_slot.py
    return s


def test_option_mass_is_the_share_on_the_options():
    # the model puts 0.5 + 0.2 + 0.1 = 0.8 of its next-token probability
    # on the three option letters, 0.2 elsewhere
    s = _gather_serving([0.5, 0.2, 0.1], cached=0)
    a = _answer(s, TYPED)
    prov = a["extra"]["audit"]
    assert prov["option_mass"] == pytest.approx(0.8)
    # the probabilities are still renormalized over the options
    assert a["probabilities"] == pytest.approx(
        {"a": 0.625, "b": 0.25, "c": 0.125})


def test_option_mass_needs_full_vocab_logprobs():
    s = _gather_serving([0.5, 0.2, 0.1], cached=0)
    s.model_config.logprobs_mode = "raw_logits"
    try:
        a = _answer(s, TYPED)
        assert a["extra"]["audit"]["option_mass"] is None
    finally:
        del s.model_config.logprobs_mode


def test_cached_tokens_per_answer_and_in_usage():
    s = _gather_serving([0.5, 0.2, 0.1], cached=40)
    r = _run(s.create_decisions(DecisionsRequest(**THREE)))
    for a in r["answers"].values():
        assert a["extra"]["audit"]["cached_input_tokens"] == 40
    assert r["usage"]["cached_input_tokens"] == 120
    assert r["usage"]["input_tokens"] > 120


def test_two_stage_counts_every_read():
    s = _gather_serving([0.3] * 26, cached=7)
    body = _choice(30)
    r = _run(s.create_decisions(DecisionsRequest(**body)))
    a = r["answers"]["q"]
    block = a["extra"]["backend"]
    assert block["name"] == "logit"
    assert a["extra"]["audit"]["readout"] == "two-stage"
    # 30 stage-1 reads + 1 stage-2 read, each reporting 7 cached tokens
    assert a["extra"]["audit"]["cached_input_tokens"] == 31 * 7
    assert a["extra"]["audit"]["option_mass"] is None
    assert a["extra"]["audit"]["forward_passes"] == 31


def test_systemone_usage_stays_jev_shaped():
    s = _gather_serving([0.5, 0.2, 0.1], cached=40, cls=ServingSystemOne)
    r = _run(s.create_systemone(SystemOneRequest(model="m", **TYPED)))
    assert set(r.usage.model_dump()) == {"input_tokens", "output_tokens"}


# ---------------------------------------------------------------------
# the `extra` level: full | basic | none
# ---------------------------------------------------------------------

def test_extra_full_is_the_default():
    s = _gather_serving([0.5, 0.2, 0.1], cached=4)
    a = _answer(s, TYPED)
    assert set(a["extra"]) == {"backend", "audit"}
    assert a["extra"]["backend"]["name"] == "logit"
    assert set(a["extra"]["backend"]["option_logits"]) == {"a", "b", "c"}


def test_extra_basic_drops_only_per_option_lists():
    s = _gather_serving([0.5, 0.2, 0.1], cached=4)
    full = _answer(s, TYPED)
    basic = _answer(s, {**TYPED, "extra": "basic"})
    assert "option_logits" not in basic["extra"]["backend"]
    assert basic["extra"]["audit"] == full["extra"]["audit"]
    assert basic["extra"]["backend"]["name"] == "logit"
    # the answer itself is untouched
    for key in ("probabilities", "choice", "confidence"):
        assert basic[key] == full[key]


def test_extra_basic_trims_two_stage_scores():
    s = _gather_serving([0.3] * 26, cached=7)
    a = _answer(s, {**_choice(30), "extra": "basic"}, qid="q")
    block = a["extra"]["backend"]
    assert "stage1_scores" not in block and "option_logits" not in block
    assert len(block["shortlist"]) == 16      # a short list, kept
    assert a["extra"]["audit"]["forward_passes"] == 31


def test_extra_none_has_no_block():
    s = _gather_serving([0.5, 0.2, 0.1], cached=4)
    r = _run(s.create_decisions(DecisionsRequest(**{**THREE,
                                                    "extra": "none"})))
    for a in r["answers"].values():
        assert "extra" not in a
    # usage still carries the request totals
    assert r["usage"]["cached_input_tokens"] == 12


def test_extra_level_validated():
    with pytest.raises(ValidationError):
        DecisionsRequest(**{**TYPED, "extra": "summary"})


def test_bad_readout_is_a_422_not_a_crash():
    # the refusal must carry its real status. The strict stub
    # fails on a plain int, so this test catches the int-status bug.
    s = _make()
    r = _run(s.create_decisions(DecisionsRequest(
        **{**_choice(30), "backend_options": {"readout": "bogus"}})))
    assert isinstance(r, ErrorResponse)
    assert r.error.code == 422
    assert "readout" in r.error.message


def test_all_questions_failed_returns_failure_code_not_500():
    # every question failing returns the first failure's own
    # code (400 for a backend error), with the question named in the
    # message - not a 500 from a broken error path.
    s = _make(backend=_fake_backend(fail={"q"}))
    r = _run(s.create_decisions(DecisionsRequest(**_choice(30))))
    assert isinstance(r, ErrorResponse)
    assert r.error.code == 400
    assert r.error.message.startswith("question 'q' failed:")


# ---------------------------------------------------------------------
# the auto readout rule (direct -> wide-direct -> two-stage)
# ---------------------------------------------------------------------

class _MergingTok(_Tok):
    """Like _Tok, but every pair starting with "A" (AA, AB, ...) encodes
    to ONE token that decodes back to the pair, so wide-direct has
    capacity above the 26 plain markers."""

    def encode(self, text, add_special_tokens=False):
        out = []
        i = 0
        while i < len(text):
            if text.startswith("AA", i):
                out.append(1000)
                i += 2
            else:
                out.append(ord(text[i]))
                i += 1
        return out or [0]

    def decode(self, ids):
        if list(ids) == [1000]:
            return "AA"
        return "".join(chr(i) for i in ids)


def test_auto_picks_direct_within_markers():
    s = _gather_serving([0.5] * 26, cached=0)
    r = _run(s.create_decisions(DecisionsRequest(**_choice(26))))
    assert not isinstance(r, ErrorResponse), getattr(r, "error", None)
    a = r["answers"]["q"]
    assert a["extra"]["audit"]["readout"] == "direct"


def test_auto_picks_wide_direct_up_to_its_capacity():
    # "AA" merges everywhere it appears, so wide-direct holds 27 markers;
    # 27 options must land on wide-direct, not two-stage.
    s = _gather_serving([0.5] * 27, cached=0, tok=_MergingTok())
    r = _run(s.create_decisions(DecisionsRequest(**_choice(27))))
    a = r["answers"]["q"]
    assert a["extra"]["audit"]["readout"] == "wide-direct"


def test_auto_falls_back_to_two_stage_beyond_capacity():
    s = _gather_serving([0.3] * 26, cached=7)  # _Tok: no pair merges, cap 26
    r = _run(s.create_decisions(DecisionsRequest(**_choice(30))))
    a = r["answers"]["q"]
    assert a["extra"]["audit"]["readout"] == "two-stage"


# ---------------------------------------------------------------------
# RENDER_VERSION
# ---------------------------------------------------------------------

def test_render_version_lands_in_every_audit():
    name = _fake_backend()
    s = _make(backend=name)
    r = _run(s.create_decisions(DecisionsRequest(**TYPED)))
    for a in r["answers"].values():
        assert a["extra"]["audit"]["render_version"] == RENDER_VERSION


GOLDEN_RENDER_HASH = ("bfe4dbf3925667e56a553ae0eb0b2e36bfb457ab44d169a229b7d6042c5dac9e",
                      "2026-09-29.1")


def test_golden_render_hash_matches_render_version():
    """One hash over all golden renders, computed as the current code
    produces them. If this fails after a render change, bump
    RENDER_VERSION in compile.py and update the hash here. A saved
    startup calibration is only valid for one render version."""
    h = hashlib.sha256()
    for case in sorted(CASES):
        h.update(repr(_renders(_make(), CASES[case])).encode())
    assert (h.hexdigest(), RENDER_VERSION) == GOLDEN_RENDER_HASH, (
        "golden renders changed without a RENDER_VERSION bump: bump "
        "RENDER_VERSION in compile.py and update GOLDEN_RENDER_HASH")


# ---------------------------------------------------------------------
# the once-per-server startup, shared by both endpoints
# ---------------------------------------------------------------------

@pytest.fixture()
def fresh_startup():
    """A fresh once-per-server startup object for this test."""
    import vllm.entrypoints.generate.decisions.startup as _su
    _su._STARTUP = None
    yield
    _su._STARTUP = None


def test_self_check_runs_once_and_is_shared_by_both_endpoints(fresh_startup):
    calls = {"n": 0}

    class CountingEngine(_GatherEngine):
        def __init__(self, probs, cached):
            super().__init__(probs, cached)

        def generate(self, engine_input, params, request_id):
            async def gen():
                if "selfcheck" in request_id:
                    calls["n"] += 1
                async for item in _GatherEngine.generate(
                        self, engine_input, params, request_id):
                    yield item
            return gen()

    s1 = ServingDecisions(CountingEngine([0.5] * 4, cached=0),
                          _Models(), _Renderer(),
                          request_logger=None, default_backend="logit")
    r1 = _run(s1.create_decisions(DecisionsRequest(**TYPED)))
    assert not isinstance(r1, ErrorResponse), getattr(r1, "error", None)
    n_after_decisions = calls["n"]
    s2 = ServingSystemOne(CountingEngine([0.5] * 4, cached=0),
                          _Models(), _Renderer(),
                          request_logger=None, default_backend="logit")
    # the second instance shares the first one's startup object
    assert s2.startup is s1.startup
    r2 = _run(s2.create_systemone(
        SystemOneRequest(model="m", **TYPED)))
    assert not isinstance(r2, ErrorResponse), getattr(r2, "error", None)
    # the shared self-check ran once: no extra probe after the first request
    assert calls["n"] == n_after_decisions


def test_request_during_startup_gets_503(fresh_startup):
    import vllm.entrypoints.generate.decisions.startup as _su
    su = _su.get_startup()

    async def never():
        await asyncio.Event().wait()  # hangs until cancelled

    # simulate the scheduled background work still running: build the
    # serving INSIDE a running loop, the way the wire patch does, and
    # schedule through the real entry point.
    async def main():
        su.schedule(never)
        s = ServingDecisions(_Engine(), _Models(), _Renderer(),
                             request_logger=None, default_backend="logit")
        assert s.startup.starting is True
        r = await s.create_decisions(DecisionsRequest(**TYPED))
        return r

    r = asyncio.run(main())
    assert isinstance(r, ErrorResponse)
    assert r.error.code == 503
    assert "decision server is starting" in r.error.message
    # the task is still pending; cancel it so the test leaves nothing behind
    if su._task is not None and not su._task.done():
        su._task.cancel()
        try:
            asyncio.run(su._task)
        except (asyncio.CancelledError, RuntimeError):
            pass


def test_startup_crash_gives_refusal_not_hang(fresh_startup):
    import vllm.entrypoints.generate.decisions.startup as _su
    su = _su.get_startup()

    async def boom():
        raise RuntimeError("probe exploded")

    loop = asyncio.new_event_loop()
    try:
        su._task = loop.run_until_complete(
            loop.create_task(su._guard(boom())))
        # let the guard record the failure
        s = _make()
        r = _run(s.create_decisions(DecisionsRequest(**TYPED)))
        assert isinstance(r, ErrorResponse)
        assert r.error.code == 503
        assert "self-check failed" in r.error.message
        assert "probe exploded" in r.error.message
    finally:
        loop.close()
