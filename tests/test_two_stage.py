"""Regression tests for the two-stage read.

Pins three past bugs:
1. stage-1 options must carry yes/no semantics (descriptions Yes/No,
   not the bare marker letters "A"/"B");
2. every engine request gets a distinct request id;
3. shortlist ties are not broken by option position.
"""
import asyncio
import re

import pytest

from test_decisions_offline import _FakeLogProb, _FakeResult


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _Renderer:
    def get_tokenizer(self):
        return _Tok()


class _Engine:
    """Stage-1: yes-logprob depends on the option under evaluation via
    `yes_by_desc`; stage-2: favours `stage2_pick` among the finalists."""

    def __init__(self, yes_by_desc, stage2_pick=None, default_yes=-1.0):
        self.yes_by_desc = yes_by_desc
        self.stage2_pick = stage2_pick
        self.default_yes = default_yes
        self.request_ids = []

    def generate(self, engine_input, params, request_id):
        self.request_ids.append(request_id)
        req = engine_input["req"]

        async def gen():
            want = list(params.logprob_token_ids)
            lp = {}
            if engine_input["stage"] == 1:
                m = re.search(r"Option under evaluation: (\S+)", req.question)
                y = self.yes_by_desc.get(m.group(1), self.default_yes)
                lp[ord("A")] = _FakeLogProb(y)
                lp[ord("B")] = _FakeLogProb(-1.0)
            else:
                for j, tid in enumerate(want):
                    oid = req.options[j].id
                    lp[tid] = _FakeLogProb(
                        -0.1 if oid == self.stage2_pick else -3.0)
            yield _FakeResult([lp])
        return gen()


class _Serving:
    def __init__(self, engine):
        self.engine_client = engine
        self.base_renderer = _Renderer()
        self.stage1_requests = []

    async def _build_prompt(self, request, labels=None, prompt_family=None):
        markers = [chr(ord("A") + i) for i in range(len(request.options))]
        stage = 1 if [o.id for o in request.options] == ["true", "false"] \
            else 2
        if stage == 1:
            self.stage1_requests.append(request)
        return ({"prompt_token_ids": [1], "req": request, "stage": stage},
                1, [ord(m) for m in markers])


class _Backend:
    name = "logit"

    def __init__(self, serving):
        from vllm.entrypoints.generate.decisions.backends.host import (
            BackendHost)
        self.serving = serving
        self.host = BackendHost(serving)


def _run(k, yes_by_desc, stage2_pick=None, shortlist=4, options=None):
    from vllm.entrypoints.generate.decisions.backends.large_choice import (
        two_stage_read)
    from vllm.entrypoints.generate.decisions.limits import DecisionLimits
    from vllm.entrypoints.generate.decisions.protocol import (
        DecisionOption, CompiledQuestion)
    opts = options or [DecisionOption(id=f"o{i}", description=f"d{i}")
                       for i in range(k)]
    req = CompiledQuestion(state="s", question="q", options=opts)
    engine = _Engine(yes_by_desc, stage2_pick)
    serving = _Serving(engine)
    res = asyncio.run(two_stage_read(
        _Backend(serving), req, "rid",
        DecisionLimits(shortlist=shortlist)))
    return res, engine, serving


def test_stage1_options_are_yes_no_not_marker_letters():
    _, _, serving = _run(6, {})
    assert serving.stage1_requests
    for r in serving.stage1_requests:
        descs = [o.description for o in r.options]
        assert descs == ["Yes", "No"], descs
        assert descs != ["A", "B"]
        assert "Is this option the correct answer?" in r.question


def test_every_engine_request_has_a_distinct_id():
    _, engine, _ = _run(10, {})
    assert len(engine.request_ids) == 11
    assert len(set(engine.request_ids)) == 11
    assert "rid-s2" in engine.request_ids


def test_stage1_signal_reaches_shortlist_and_answer():
    # option d7 is the only one the stage-1 read says "yes" to
    res, _, _ = _run(20, {"d7": 3.0}, stage2_pick="o7", shortlist=4)
    assert "o7" in res.meta["shortlist"]
    assert max(res.probabilities, key=res.probabilities.get) == "o7"
    assert res.meta["stage1_prompt"] == "noul-yes-no-v1"


def test_ties_not_broken_by_position():
    # all stage-1 scores tie: the shortlist must not be the first N
    # options, and must be the same set of ids whatever the order
    from vllm.entrypoints.generate.decisions.protocol import DecisionOption
    opts = [DecisionOption(id=f"o{i}", description=f"d{i}")
            for i in range(40)]
    res_a, _, _ = _run(40, {}, shortlist=8, options=opts)
    res_b, _, _ = _run(40, {}, shortlist=8, options=list(reversed(opts)))
    sl_a, sl_b = set(res_a.meta["shortlist"]), set(res_b.meta["shortlist"])
    assert sl_a == sl_b
    assert sl_a != {f"o{i}" for i in range(8)}


def test_slot_mismatch_is_an_error():
    from vllm.entrypoints.generate.decisions.backends import BackendError

    class _BadServing(_Serving):
        async def _build_prompt(self, request, labels=None,
                                prompt_family=None):
            ei, n, slots = await super()._build_prompt(request)
            return ei, n, list(reversed(slots))

    from vllm.entrypoints.generate.decisions.backends.large_choice import (
        two_stage_read)
    from vllm.entrypoints.generate.decisions.limits import DecisionLimits
    from vllm.entrypoints.generate.decisions.protocol import (
        DecisionOption, CompiledQuestion)
    req = CompiledQuestion(state="s", question="q", options=[
        DecisionOption(id=f"o{i}", description=f"d{i}") for i in range(3)])
    serving = _BadServing(_Engine({}))
    with pytest.raises(BackendError):
        asyncio.run(two_stage_read(_Backend(serving), req, "rid",
                                   DecisionLimits(shortlist=2)))
