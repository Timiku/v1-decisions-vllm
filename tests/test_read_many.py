"""read_many: the optional several-questions-at-once backend read, the
server's dispatch to it, and the canvas joint read. No GPU (stubs)."""
from __future__ import annotations

import asyncio
import json

import pytest

from vllm.entrypoints.generate.decisions.backends import (
    BackendError, BackendResult, register_backend)
from vllm.entrypoints.generate.decisions.backends.canvas_backend import (
    CanvasBackend)
from vllm.entrypoints.generate.decisions.backends.host import BackendHost
from vllm.entrypoints.generate.decisions.limits import (
    DecisionLimits, set_limits_for_tests)
from vllm.entrypoints.generate.decisions.protocol import (
    CompiledQuestion, DecisionOption, DecisionsQuery)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse

from test_decisions_offline import (
    _FakeBaseRenderer, _FakeLogProb, _FakeModelConfig, _FakeModels,
    _FakeRenderer, _FakeResult, _FakeServing, _FakeTokenizer)


@pytest.fixture(autouse=True)
def fresh_startup():
    import vllm.entrypoints.generate.decisions.startup as _su
    _su._STARTUP = None
    set_limits_for_tests(DecisionLimits())
    yield
    _su._STARTUP = None
    set_limits_for_tests(DecisionLimits())


# the fake tokenizer is per character: "1: A\n2: A\n3: A" puts the
# letters at canvas positions 3, 8 and 13
SLOT = {1: 3, 2: 8, 3: 13}


class _JointEngine:
    """Logprobs at every canvas position; at position p the label
    favour[p] scores highest. Records every request's params."""

    def __init__(self, favour=None, omit=()):
        self.favour = favour or {}
        self.omit = set(omit)
        self.params = []
        self.model_config = _FakeModelConfig()
        self.errored = False
        self.dead_error = RuntimeError("engine dead")

    def generate(self, engine_input, params, request_id):
        async def gen():
            self.params.append(params)
            want = list(params.logprob_token_ids)
            rows = []
            for p in range(params.extra_args["diffusion_canvas_length"]):
                fav = ord(self.favour.get(p, "A"))
                rows.append({t: _FakeLogProb(-0.1 if t == fav else -3.0)
                             for t in want if (p, t) not in self.omit})
            yield _FakeResult(rows)
        return gen()


class _JointServing(_FakeServing):
    """_FakeServing plus a joint render: 40 prompt tokens."""

    async def _build_joint_prompt(self, questions):
        self.joint_calls = getattr(self, "joint_calls", 0) + 1
        ids = list(range(40))
        return {"prompt_token_ids": ids}, len(ids), ids


def _q(n_options=2, state="s", seed=7, **kw):
    return CompiledQuestion(
        state=state, question="q", seed=seed,
        options=[DecisionOption(id=f"o{i}", description=f"d{i}")
                 for i in range(n_options)], **kw)


def _canvas(engine, **kw):
    return CanvasBackend(BackendHost(_JointServing(engine)), **kw)


class TestCanvasJointRead:
    def test_each_question_reads_its_own_slot(self):
        engine = _JointEngine(favour={SLOT[1]: "B", SLOT[2]: "A",
                                      SLOT[3]: "C"})
        backend = _canvas(engine)
        res = asyncio.run(backend.read_many(
            [_q(2), _q(3), _q(3)], "req"))
        picks = [max(r.probabilities, key=r.probabilities.get) for r in res]
        assert picks == ["o1", "o0", "o2"]
        # one engine request answered all three
        assert len(engine.params) == 1
        assert all(r.meta["readout"] == "joint" for r in res)
        assert res[1].meta["joint_read"]["canvas_slot"] == SLOT[2]

    def test_canvas_is_template_with_noise_at_slots(self):
        engine = _JointEngine()
        asyncio.run(_canvas(engine).read_many([_q(), _q()], "req"))
        canvas = engine.params[0].extra_args["diffusion_seed_canvas"]
        template = _FakeTokenizer().encode("1: A\n2: A")
        assert len(canvas) == 64
        for p, t in enumerate(template):
            if p not in (SLOT[1], SLOT[2]):
                assert canvas[p] == t
        assert canvas[len(template)] == 106          # turn close
        assert set(canvas[len(template) + 1:]) == {0}  # pad
        # logprobs == len(logprob_token_ids), the union of the letters
        params = engine.params[0]
        assert params.logprob_token_ids == [ord("A"), ord("B")]
        assert params.logprobs == 2

    def test_multi_step_declines(self):
        # the template can't be pinned on patches/00572
        engine = _JointEngine()
        res = asyncio.run(_canvas(engine, max_steps=4).read_many(
            [_q(), _q()], "req"))
        assert res == [None, None]
        assert engine.params == []

    def test_samples_reproducible_and_counted(self):
        e1, e2 = _JointEngine(), _JointEngine()
        r1 = asyncio.run(_canvas(e1, samples=3).read_many([_q(), _q()], "a"))
        r2 = asyncio.run(_canvas(e2, samples=3).read_many([_q(), _q()], "b"))
        assert [p.extra_args["diffusion_seed_canvas"] for p in e1.params] \
            == [p.extra_args["diffusion_seed_canvas"] for p in e2.params]
        assert len(e1.params) == 3
        assert r1[0].forward_passes == 3
        assert r1[0].meta["samples"]["n"] == 3
        assert r1[0].probabilities == r2[0].probabilities

    def test_input_tokens_split_adds_up(self):
        engine = _JointEngine()
        res = asyncio.run(_canvas(engine, samples=2).read_many(
            [_q(), _q(), _q()], "req"))
        # 40-token joint prompt, 2 samples, over 3 questions
        assert [r.meta["input_tokens"] for r in res] == [27, 27, 26]

    def test_questions_past_the_canvas_are_left_to_read(self):
        # width 12 holds "1: A\n2: A" (9 tokens) + turn close, not a third
        engine = _JointEngine()
        res = asyncio.run(_canvas(engine, canvas_width=12).read_many(
            [_q(), _q(), _q()], "req"))
        assert res[0] is not None and res[1] is not None
        assert res[2] is None

    def test_different_states_decline(self):
        engine = _JointEngine()
        res = asyncio.run(_canvas(engine).read_many(
            [_q(state="x"), _q(state="y")], "req"))
        assert res == [None, None]
        assert engine.params == []

    def test_single_question_declines(self):
        engine = _JointEngine()
        assert asyncio.run(_canvas(engine).read_many([_q()], "req")) \
            == [None]

    def test_missing_logprob_fails_the_chunk(self):
        engine = _JointEngine(omit={(SLOT[2], ord("B"))})
        res = asyncio.run(_canvas(engine).read_many([_q(), _q()], "req"))
        assert all(isinstance(r, BackendError) for r in res)
        assert "o1" in str(res[0])

    def test_letters_that_merge_decline(self):
        # markers whose tokens don't share one slot (a two-token label)
        set_limits_for_tests(DecisionLimits(
            markers=["A", "B!"] + list("CDEFGHIJKLMNOP")))
        engine = _JointEngine()
        res = asyncio.run(_canvas(engine).read_many([_q(), _q()], "req"))
        assert res == [None, None]


# ---------------------------------------------------------------------
# server dispatch
# ---------------------------------------------------------------------

def _serving(backend_cls=None, engine=None, default_backend=None):
    from vllm.entrypoints.generate.decisions.serving import ServingDecisions
    engine = engine or _JointEngine()
    name = default_backend
    if backend_cls is not None:
        name = f"rm-{id(backend_cls)}"
        backend_cls.name = name
        register_backend(name, backend_cls)
    renderer = _FakeRenderer()
    renderer.renderer = _FakeBaseRenderer()
    return ServingDecisions(engine, _FakeModels(), renderer,
                            request_logger=None, default_backend=name)


def _request(qids=("q1", "q2", "q3")):
    return DecisionsQuery(model="m", state="s", questions={
        q: {"type": "choice", "instructions": "decide",
            "criteria": {"A": "a", "B": "b"}} for q in qids})


def _result(tokens=3):
    return BackendResult({"A": -0.1, "B": -2.3}, {"A": 0.9, "B": 0.1},
                         forward_passes=1, meta={"input_tokens": tokens})


class TestDispatch:
    def test_joint_results_none_fallback_and_errors(self):
        calls = {"many": 0, "read": []}

        class B:
            def __init__(self, host, **kw):
                pass

            async def read_many(self, questions, request_id):
                calls["many"] += 1
                return [_result(10), None, BackendError("no room")]

            async def read(self, question, request_id):
                calls["read"].append(request_id.rsplit("-", 1)[-1])
                return _result(3)

        s = _serving(B)
        out = asyncio.run(s.answer_query(_request(), None))
        assert calls["many"] == 1
        assert calls["read"] == ["q2"]
        assert set(out["answers"]) == {"q1", "q2"}
        assert "no room" in out["partial_failures"]["q3"]
        assert out["usage"]["input_tokens"] == 13

    def test_read_many_raising_falls_back_to_read(self):
        reads = []

        class B:
            def __init__(self, host, **kw):
                pass

            async def read_many(self, questions, request_id):
                raise RuntimeError("joint read broke")

            async def read(self, question, request_id):
                reads.append(request_id)
                return _result()

        s = _serving(B)
        out = asyncio.run(s.answer_query(_request(), None))
        assert len(reads) == 3
        assert set(out["answers"]) == {"q1", "q2", "q3"}

    def test_single_question_skips_read_many(self):
        class B:
            def __init__(self, host, **kw):
                pass

            async def read_many(self, questions, request_id):
                raise AssertionError("read_many for one question")

            async def read(self, question, request_id):
                return _result()

        s = _serving(B)
        out = asyncio.run(s.answer_query(_request(("q1",)), None))
        assert set(out["answers"]) == {"q1"}

    def test_backend_without_read_many_is_unchanged(self):
        class B:
            def __init__(self, host, **kw):
                pass

            async def read(self, question, request_id):
                return _result()

        s = _serving(B)
        out = asyncio.run(s.answer_query(_request(), None))
        assert out["usage"]["input_tokens"] == 9

    def test_canvas_end_to_end_one_engine_request(self):
        engine = _JointEngine(favour={SLOT[1]: "B", SLOT[2]: "A",
                                      SLOT[3]: "B"})
        s = _serving(engine=engine, default_backend="canvas")
        out = asyncio.run(s.answer_query(_request(), None))
        assert len(engine.params) == 1
        picks = {q: max(a["probabilities"], key=a["probabilities"].get)
                 for q, a in out["answers"].items()}
        # choice criteria {"A": "a", "B": "b"}: option ids A, B
        assert picks == {"q1": "B", "q2": "A", "q3": "B"}
        audit = out["answers"]["q1"]["extra"]["audit"]
        assert audit["readout"] == "joint"
        # the joint prompt's tokens, counted once over the request
        joint = out["answers"]["q1"]["extra"]["backend"]["joint_read"]
        assert out["usage"]["input_tokens"] == joint["input_tokens"]


class TestJointPrompt:
    def test_payload_numbers_every_question(self):
        s = _serving(default_backend="canvas")
        qs = [_q(2), _q(3)]
        engine_input, n, ids = asyncio.run(s._build_joint_prompt(qs))
        # the fake renderer encodes the user message's JSON
        text = json.loads(_FakeTokenizer().decode(ids))
        payload = json.loads(text)
        assert payload["evidence"] == "s"
        assert [q["number"] for q in payload["questions"]] == [1, 2]
        assert [o["letter"] for o in payload["questions"][1]["options"]] \
            == ["A", "B", "C"]
        assert n == len(ids)
