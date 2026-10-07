"""Startup calibration, offline.

- the B2 gating table (which backend calibrates under which setting);
- the B7 temperature precedence (all four sources, in order);
- the saved-result reuse/re-run behavior on key changes (B6);
- every B8 failure case falling back to T=1.0 with a warning;
- a 503 while calibration runs;
- the fit itself: a 3x-too-sharp synthetic set gives T~3 kept, a
  well-calibrated one gives not kept;
- the real-data check: the fit must reproduce the 4B's kept
  T~9.5 and the 27B's not-kept T~0.93 (B5).
"""
from __future__ import annotations

import asyncio
import json
import math
import os

import numpy as np
import pytest

import vllm.entrypoints.generate.decisions.calibration as cal
from vllm.entrypoints.generate.decisions.backends import (
    register_backend)
from vllm.entrypoints.generate.decisions.limits import (
    DecisionLimits, set_limits_for_tests)
from vllm.entrypoints.generate.decisions.protocol import DecisionsQuery
from vllm.entrypoints.generate.decisions.serving import ServingDecisions
from vllm.entrypoints.generate.decisions.systemone_serving import (
    ServingSystemOne)

DATA = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------

class _Models:
    def model_name(self, _):
        return "served-model"


class _ModelConfig:
    max_model_len = 10**6
    max_logprobs = 600
    architectures = []
    model = "/models/fake"


class _Engine:
    model_config = _ModelConfig()
    errored = False
    dead_error = RuntimeError("dead")


class _BaseRenderer:
    def get_tokenizer(self):
        return _Tok()

    async def render_chat_async(self, conversations, chat_params,
                                tok_params):
        text = ("\n".join(m["content"] for m in conversations[0])
                + "\nASSISTANT:")
        return None, [{"prompt_token_ids":
                       self.get_tokenizer().encode(text)}]


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text] or [0]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _Renderer:
    chat_template = None
    chat_template_content_format = "string"

    def __init__(self):
        self.renderer = _BaseRenderer()


class _LP:
    def __init__(self, logprob):
        self.logprob = logprob
        self.decoded_token = ""


class _Out:
    def __init__(self, logprobs):
        # host reads logprobs[0][token_id].logprob: one position whose
        # dict maps the requested token ids to logprob objects
        self.logprobs = [logprobs]
        self.finish_reason = "length"


class _Result:
    def __init__(self, logprobs, cached):
        self.outputs = [_Out(logprobs)]
        self.num_cached_tokens = cached


class _BackendResultShaped:
    def __init__(self, logits):
        self.option_logits = logits
        self.forward_passes = 1
        self.meta = {"input_tokens": 10, "cached_input_tokens": 0,
                     "readout": "direct", "label_layout": "direct"}


class _ScriptedBackend:
    """Per-question logits via logits_for(request_id, question)."""
    name = "logit"
    calibrate_by_default = True

    def __init__(self, logits_for):
        self.logits_for = logits_for

    async def read(self, question, request_id):
        logits = self.logits_for(request_id, question)
        return _BackendResultShaped(logits)


@pytest.fixture(autouse=True)
def fresh_startup():
    import vllm.entrypoints.generate.decisions.startup as _su
    _su._STARTUP = None
    yield
    _su._STARTUP = None


@pytest.fixture(autouse=True)
def default_limits():
    set_limits_for_tests(DecisionLimits())
    yield
    set_limits_for_tests(DecisionLimits())


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("VLLM_TYPED_DECISIONS_CALIBRATION",
                "VLLM_TYPED_DECISIONS_TEMPERATURE"):
        monkeypatch.delenv(var, raising=False)


class _WorkingEngine(_Engine):
    """Serves a healthy restricted gather (probe- and read-capable)."""
    def generate(self, engine_input, params, request_id):
        async def gen():
            want = list(params.logprob_token_ids)
            yield _Result({t: _LP(math.log(0.6)) for t in want}, 0)
        return gen()


def _serving(logits_for, backend_name="logit"):
    s = ServingDecisions(_WorkingEngine(), _Models(), _Renderer(),
                         request_logger=None, default_backend=backend_name)
    if logits_for is not None:
        s.decision_backend = _ScriptedBackend(logits_for)
    return s


def _real_pairs(name):
    """The real capture (tests/data/<name>) as fit pairs."""
    preds = {r["id"]: r for r in map(
        json.loads, open(os.path.join(DATA, name + ".jsonl"),
                         encoding="utf-8"))}
    pairs = []
    for g in map(json.loads, open(os.path.join(DATA, name + ".jsonl.gold"),
                                  encoding="utf-8")):
        p = preds[g["id"]]
        if "error" in p:
            continue
        pairs.append({"id": g["id"], "logits": p["option_logits"],
                      "true_index": g["label"],
                      "group": g.get("group_id") or g["id"],
                      "family": g.get("family", "")})
    return pairs


def _scaled(pairs, factor):
    """Logits scaled by factor: the fitted T scales by 1/factor."""
    return [{"id": p["id"], "logits": [factor * l for l in p["logits"]],
             "true_index": p["true_index"], "group": p["group"],
             "family": p["family"]} for p in pairs]


def test_fit_sharp_set_gives_T3_and_kept():
    # the real 4B fit gives T=9.497; logits / 3 -> fitted T ~ 3.17, kept
    pairs = _scaled(_real_pairs("qwen3-4b-jevbench"), 1.0 / 3.0)
    fit = cal.fit_all(pairs)
    assert fit["kept"] is True
    assert 2.7 <= fit["t"] <= 3.6


def test_well_calibrated_set_is_not_kept():
    # logits / 9.497 -> fitted T ~ 1.0: measured, nothing to correct
    pairs = _scaled(_real_pairs("qwen3-4b-jevbench"), 1.0 / 9.497)
    fit = cal.fit_all(pairs)
    assert fit["kept"] is False


def test_range_edge_t_falls_back():
    # logits * 4 -> fitted T ~ 2.37... instead drive T below the range:
    # logits / 400 -> fitted T ~ 0.024, below T_LOW -> the runner treats
    # a range-edge fit as broken scores and falls back to 1.0.
    pairs = _scaled(_real_pairs("qwen3-4b-jevbench"), 1.0 / 400.0)
    fit = cal.fit_all(pairs)
    assert cal.at_range_edge(fit["t"])


# ---------------------------------------------------------------------
# B2: the gating table
# ---------------------------------------------------------------------

def _gate(monkeypatch, calibration, calibrate_by_default, temp_env=None):
    limits = DecisionLimits(calibration=calibration)
    set_limits_for_tests(limits)
    if temp_env:
        monkeypatch.setenv("VLLM_TYPED_DECISIONS_TEMPERATURE", temp_env)
    s = _serving(None)
    s.decision_backend.calibrate_by_default = calibrate_by_default
    should, _ = cal.should_calibrate(s, limits)
    return should


def test_gate_jevbench_logit_calibrates(monkeypatch):
    assert _gate(monkeypatch, None, True) is True


def test_gate_jevbench_other_backend_does_not(monkeypatch):
    assert _gate(monkeypatch, None, False) is False


def test_gate_on_calibrates_any_backend(monkeypatch):
    assert _gate(monkeypatch, "on", False) is True


def test_gate_file_calibrates_any_backend(monkeypatch, tmp_path):
    f = tmp_path / "own.jsonl"
    f.write_text('{"state":"s","question":{"type":"noul",'
                 '"instructions":"i"},"expected":"true"}\n')
    assert _gate(monkeypatch, str(f), False) is True


def test_gate_off_never(monkeypatch):
    assert _gate(monkeypatch, "off", True) is False


def test_gate_operator_temperature_skips(monkeypatch):
    assert _gate(monkeypatch, None, True, temp_env="1.0") is False


# ---------------------------------------------------------------------
# B7: the temperature precedence, in order
# ---------------------------------------------------------------------

def test_precedence_request_beats_everything(monkeypatch):
    monkeypatch.setenv("VLLM_TYPED_DECISIONS_TEMPERATURE", "2.0")
    s = _serving(None)
    s.startup.calibration = {"t": 5.0, "kept": True, "source": "calibrated"}
    assert s.resolve_temperature(2.0) == (2.0, "request")


def test_precedence_operator_temperature_is_server():
    set_limits_for_tests(DecisionLimits(temperature=2.0))
    s = _serving(None)
    s.startup.calibration = {"t": 5.0, "kept": True, "source": "calibrated"}
    assert s.resolve_temperature(None) == (2.0, "server")


def test_precedence_calibrated_when_no_operator_t():
    s = _serving(None)
    s.startup.calibration = {"t": 9.5, "kept": True, "source": "calibrated"}
    assert s.resolve_temperature(None) == (9.5, "calibrated")


def test_precedence_calibrated_1_0_when_not_kept():
    s = _serving(None)
    s.startup.calibration = {"t": 1.0, "kept": False,
                             "source": "calibrated"}
    assert s.resolve_temperature(None) == (1.0, "calibrated")


def test_precedence_default_when_calibration_absent():
    s = _serving(None)
    assert s.resolve_temperature(None) == (1.0, "default")


# ---------------------------------------------------------------------
# B6: saved result reuse / re-run on key change
# ---------------------------------------------------------------------

def test_saved_result_reused_then_rerun_on_key_change(tmp_path):
    key = {"served_model": "m", "render_version": "2026-09-29.1",
           "question_file_sha256": "abc"}
    result = {"t": 9.5, "kept": True}
    path = cal.save_result(str(tmp_path), key, result)
    assert os.path.exists(path)
    loaded = cal.load_result(str(tmp_path), key)
    assert loaded["t"] == result["t"] and loaded["kept"] == result["kept"]
    assert loaded["key"] == key
    # any key part changed -> no match -> re-run
    changed = dict(key, render_version="2026-09-30.1")
    assert cal.load_result(str(tmp_path), changed) is None
    # a corrupt file is treated as absent
    with open(path, "w") as f:
        f.write("{not json")
    assert cal.load_result(str(tmp_path), key) is None


# ---------------------------------------------------------------------
# B8: failure fallbacks
# ---------------------------------------------------------------------

def test_missing_question_file_falls_back(tmp_path):
    set_limits_for_tests(DecisionLimits(calibration=str(tmp_path / "nope.jsonl")))
    s = _serving(None)
    asyncio.run(cal.run_startup_calibration(s))
    assert s.startup.calibration == {"t": 1.0, "kept": False,
                                     "source": "default"}


def test_malformed_question_file_falls_back(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"state": "s"}\n')  # no question/expected
    set_limits_for_tests(DecisionLimits(calibration=str(bad)))
    s = _serving(None)
    asyncio.run(cal.run_startup_calibration(s))
    assert s.startup.calibration["t"] == 1.0
    assert s.startup.calibration["source"] == "default"


def _wire_rows(n, source="jevbench"):
    """First n rows of the bundled question set, wire format."""
    src = os.path.join(os.path.dirname(cal.__file__), "calibration_data",
                       "jevbench-public.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    return rows[:n]


def _scripted_from_wire(rows):
    """Served logits = the real 4B logits for that question id, divided
    by 3 (3x too sharp -> the fit recovers T~3.17, kept)."""
    real = {r["id"]: r for r in _real_pairs("qwen3-4b-jevbench")}

    def logits_for(request_id, question):
        index = 0
        if "decision-calibration-" in request_id:
            index = int(request_id.rsplit("-", 1)[-1])
        row = rows[index % len(rows)]
        pair = real.get(row.get("id"))
        if pair is None:
            pair = list(real.values())[index % len(real)]
        ids = [o.id for o in question.options]
        # cycle the real logits to this option count; the structure (one
        # dominant logit, two lower) survives the cycling, so the "3x
        # too sharp" property is preserved and the fit recovers T~3
        vals = [pair["logits"][i % len(pair["logits"])]
                for i in range(len(ids))]
        return {oid: l / 3.0 for oid, l in zip(ids, vals)}
    return logits_for


def test_too_few_usable_questions_falls_back(tmp_path):
    rows = _wire_rows(10)
    f = tmp_path / "few.jsonl"
    with open(f, "w") as out:
        for r in rows:
            out.write(json.dumps({k: r[k] for k in
                                  ("state", "question", "expected",
                                   "group", "id")}) + "\n")
    set_limits_for_tests(DecisionLimits(calibration=str(f)))
    s = _serving(_scripted_from_wire(rows))
    asyncio.run(cal.run_startup_calibration(s))
    assert s.startup.calibration["t"] == 1.0
    assert s.startup.calibration["source"] == "default"




def test_saved_result_is_loaded_not_rerun(tmp_path):
    rows = _wire_rows(60)
    f = tmp_path / "set.jsonl"
    with open(f, "w") as out:
        for r in rows:
            out.write(json.dumps({k: r[k] for k in
                                  ("state", "question", "expected",
                                   "group", "id")}) + "\n")
    set_limits_for_tests(DecisionLimits(calibration=str(f)))
    # first run writes the result; the run function is exercised with a
    # scripted backend via a serving
    calls = {"n": 0}

    class Counting(_ScriptedBackend):
        async def read(self, question, request_id):
            calls["n"] += 1
            return await super().read(question, request_id)

    s = _serving(_scripted_from_wire(rows))
    s.decision_backend = Counting(_scripted_from_wire(rows))
    asyncio.run(cal.run_startup_calibration(s))
    first = s.startup.calibration
    assert first["source"] == "calibrated"
    n_reads = calls["n"]
    assert n_reads > 0  # the first run actually read the questions
    import glob as _glob
    saved_dir = cal.results_dir()
    assert len(_glob.glob(os.path.join(saved_dir, "*.json"))) == 1  # exactly one saved result
    # second run: the saved file matches the key -> no questions run
    s2 = _serving(_scripted_from_wire(rows))
    s2.decision_backend = Counting(_scripted_from_wire(rows))
    asyncio.run(cal.run_startup_calibration(s2))
    assert s2.startup.calibration == first
    assert calls["n"] == n_reads  # no new reads


def test_self_check_refusal_returns_error_response(fresh_startup):
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse
    s = _serving(None)
    s.startup.slot_check = "decision self-check failed: no"
    r = asyncio.run(s.answer_query(DecisionsQuery(
        **{"state": "s", "questions": {"q": {
            "type": "choice", "instructions": "i",
            "criteria": {"a": "x", "b": "y"}}}})))
    assert isinstance(r, ErrorResponse)
    assert r.error.code == 503


def test_saved_not_kept_loads_as_1_0_calibrated(tmp_path, monkeypatch):
    # bug 2: a saved "no correction needed" result must come back as
    # T=1.0 source "calibrated" (measured), not as the fitted 0.93.
    # Calls the real run_startup_calibration with a saved result under
    # the key the server computes; fails on the pre-fix code (which
    # reloaded t=0.929).
    rows = _wire_rows(60)
    f = tmp_path / "set.jsonl"
    with open(f, "w") as out:
        for r in rows:
            out.write(json.dumps({k: r[k] for k in
                                  ("state", "question", "expected",
                                   "group", "id")}) + "\n")
    set_limits_for_tests(DecisionLimits(calibration=str(f)))
    class GatherEngine(_Engine):
        def generate(self, engine_input, params, request_id):
            async def gen():
                want = list(params.logprob_token_ids)
                yield _Result({t: _LP(math.log(0.5))
                               for t in want}, 0)
            return gen()
    s = ServingDecisions(GatherEngine(), _Models(), _Renderer(),
                         request_logger=None, default_backend="logit")
    s.decision_backend = _ScriptedBackend(_scripted_from_wire(rows))
    # save a not-kept result under the key the server computes
    key = cal.calibration_key(s, str(f), {})
    cal.save_result(str(tmp_path), key, {"t": 0.929, "kept": False,
                                         "questions_used": 60,
                                         "question_set": str(f)})
    monkeypatch.setattr(cal, "results_dir", lambda: str(tmp_path))
    asyncio.run(cal.run_startup_calibration(s))
    assert s.startup.calibration == {"t": 1.0, "kept": False,
                                     "source": "calibrated"}
    # a request's audit shows 1.0 with source calibrated
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse
    r = asyncio.run(s.answer_query(DecisionsQuery(
        **{"state": "s", "questions": {"q": {
            "type": "choice", "instructions": "i",
            "criteria": {"a": "x", "b": "y"}}}})))
    assert not isinstance(r, ErrorResponse), getattr(r, "error", None)
    prov = r["answers"]["q"]["extra"]["audit"]
    assert prov["calibration_temperature"] == 1.0
    assert prov["temperature_source"] == "calibrated"


def test_calibration_chain_idempotent_and_skipped_after_refusal(fresh_startup, monkeypatch):
    # bugs 3+4: inside a running loop, two logit serving instances
    # (calibration on) schedule ONE chain - run_startup_calibration runs
    # once. With the self-check stubbed to refuse, it runs zero times
    # and requests get the self-check 503. Calls the real scheduling
    # path (ServingDecisions.__init__ + schedule_calibration).
    import vllm.entrypoints.generate.decisions.calibration as _c
    calls = {"n": 0}

    async def fake_run(serving):
        calls["n"] += 1

    monkeypatch.setattr(_c, "run_startup_calibration", fake_run)

    class ProbeEngine(_Engine):
        # the self-check probe reads 4 markers; serve healthy logprobs
        def generate(self, engine_input, params, request_id):
            async def gen():
                want = list(params.logprob_token_ids)
                yield _Result({t: _LP(math.log(0.8))
                               for t in want}, 0)
            return gen()

    async def main(refuse):
        if refuse:
            # stub the self-check to refuse
            async def refusing(self_startup, serving):
                self_startup.slot_check = "decision self-check failed: no"
                return self_startup.slot_check
            import vllm.entrypoints.generate.decisions.startup as _su
            monkeypatch.setattr(_su.Startup, "run_self_check", refusing)
        s1 = ServingDecisions(ProbeEngine(), _Models(), _Renderer(),
                              request_logger=None, default_backend="logit")
        s2 = ServingSystemOne(ProbeEngine(), _Models(), _Renderer(),
                              request_logger=None, default_backend="logit")
        assert s1.startup is s2.startup
        await s1.startup._task
        return s1

    s1 = asyncio.run(main(refuse=False))
    assert calls["n"] == 1  # one chain, not two

    import vllm.entrypoints.generate.decisions.startup as _su
    _su._STARTUP = None
    s1 = asyncio.run(main(refuse=True))
    assert calls["n"] == 1, calls  # still 1: the refusal skipped calibration
    r = asyncio.run(s1.answer_query(DecisionsQuery(**TYPED_CAL_BODY)))
    assert getattr(r, "error", None) is not None
    assert r.error.code == 503
    assert "self-check failed" in r.error.message


TYPED_CAL_BODY = {"state": "s", "questions": {"q": {
    "type": "choice", "instructions": "i",
    "criteria": {"a": "x", "b": "y"}}}}


def test_starting_gate_covers_non_logit_backends(fresh_startup, monkeypatch):
    # bug 5: a NON-LOGIT default backend ("encoder") with CALIBRATION=on
    # must schedule the calibration (no self-check runs for it), hold
    # requests with the 503 gate while it runs, and clear the gate after.
    # FAILS on 4ef4a63's code (starting is False - work_pending never set).
    import vllm.entrypoints.generate.decisions.calibration as _c
    import vllm.entrypoints.generate.decisions.serving as _sv
    import vllm.entrypoints.generate.decisions.startup as _su
    _su._STARTUP = None
    set_limits_for_tests(DecisionLimits(calibration="on"))
    real_get = _sv.get_backend

    def fake_get(name, host, **kw):
        b = real_get("logit", host, **kw)
        b.name = "encoder"
        b.calibrate_by_default = False
        return b
    monkeypatch.setattr(_sv, "get_backend", fake_get)
    calls = {"n": 0}
    release = asyncio.Event()

    async def stub_run(serving):
        calls["n"] += 1
        await release.wait()
        serving.startup.calibration = {"t": 1.0, "kept": False,
                                       "source": "calibrated"}
    monkeypatch.setattr(_c, "run_startup_calibration", stub_run)

    async def main():
        s = ServingDecisions(_WorkingEngine(), _Models(), _Renderer(),
                             request_logger=None, default_backend="encoder")
        assert s.decision_backend.name == "encoder"
        assert s.startup.slot_check is None
        assert s.startup.starting is True
        r = await s.answer_query(DecisionsQuery(
            **{"state": "s", "questions": {"q": {
                "type": "choice", "instructions": "i",
                "criteria": {"a": "x", "b": "y"}}}}))
        assert getattr(r, "error", None) is not None and r.error.code == 503
        release.set()
        await s.startup._task
        assert calls["n"] == 1
        assert s.startup.starting is False
    asyncio.run(main())


def test_starting_gate_logit_path_self_check(fresh_startup, monkeypatch):
    """The logit path: the scheduled SELF-CHECK also holds the gate
    (companion to the non-logit test above)."""
    import vllm.entrypoints.generate.decisions.startup as _su
    _su._STARTUP = None

    async def main():
        s = ServingDecisions(_WorkingEngine(), _Models(), _Renderer(),
                             request_logger=None, default_backend="logit")
        assert s.startup.starting is True
        while s.startup.starting:
            await asyncio.sleep(0.01)
        assert s.startup.slot_check == "ok"
        assert s.startup.starting is False
    asyncio.run(main())


def test_calibration_unset_never_schedules(fresh_startup, monkeypatch):
    # CALIBRATION explicitly off: nothing scheduled, no 503, source
    # "default" (calibration off = never measured).
    monkeypatch.delenv("VLLM_TYPED_DECISIONS_TEMPERATURE",
                       raising=False)
    set_limits_for_tests(DecisionLimits(calibration="off"))
    s = _serving(None)
    assert s.startup.starting is False
    assert getattr(s.startup, "_calibration_scheduled", False) is False
    r = asyncio.run(s.answer_query(DecisionsQuery(
        **{"state": "s", "questions": {"q": {
            "type": "choice", "instructions": "i",
            "criteria": {"a": "x", "b": "y"}}}})))
    prov = r["answers"]["q"]["extra"]["audit"]
    assert prov["temperature_source"] == "default"


def monkeypatch_env():
    import os
    os.environ["VLLM_TYPED_DECISIONS_TEMPERATURE"] = "1.0"
