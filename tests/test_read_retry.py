"""retry-on-missing for restricted reads (handoff 15): the host retry
loop, the degraded generative fallback in the logit backend's direct
read, and their composition. No GPU (stubs)."""
from __future__ import annotations

import asyncio
import math

import pytest

from vllm.entrypoints.generate.decisions.backends import BackendError
from vllm.entrypoints.generate.decisions.backends import host as host_mod
from vllm.entrypoints.generate.decisions.backends.host import BackendHost
from vllm.entrypoints.generate.decisions.backends.logit_backend import (
    LogitBackend)
from vllm.entrypoints.generate.decisions.protocol import (
    DecisionOption, CompiledQuestion)

from test_decisions_offline import (
    _FakeLogProb, _FakeModelConfig, _FakeResult, _FakeTokenizer)


class _RetryEngine:
    """Per-call scripted outputs. Each generate consumes the next entry:
    a dict {token_id: _FakeLogProb} for position 0, or an Exception.
    Records (params, request_id)."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.params = []
        self.request_ids = []
        self.model_config = _FakeModelConfig()

    def generate(self, engine_input, params, request_id):
        async def gen():
            self.params.append(params)
            self.request_ids.append(request_id)
            out = self.outputs.pop(0) if self.outputs else {}
            if isinstance(out, Exception):
                raise out
            yield _FakeResult([dict(out)])
        return gen()


class _RecordingSleep:
    """Replaces asyncio.sleep inside the host module for one test."""

    def __init__(self):
        self.calls: list[float] = []

    async def __call__(self, seconds):
        self.calls.append(seconds)


@pytest.fixture
def no_sleep(monkeypatch):
    rec = _RecordingSleep()
    monkeypatch.setattr(host_mod.asyncio, "sleep", rec)
    return rec


class _Serving:
    def __init__(self, engine):
        self.engine_client = engine
        self.model_config = _FakeModelConfig()


def _lp(*token_ids, favour=None):
    """Position-0 logprob map: -0.1 for favour, -2.0 for the rest."""
    return {t: _FakeLogProb(-0.1 if t == favour else -2.0)
            for t in token_ids}


# ---------------------------------------------------------------------
# Change 1: the host retry loop
# ---------------------------------------------------------------------

class TestRestrictedReadRetries:
    def test_first_try_complete_no_retry(self, no_sleep):
        engine = _RetryEngine([_lp(32, 33)])
        host = BackendHost(_Serving(engine))
        host.read_retries = 2
        host.read_retry_backoff_s = 0.25
        found, result, attempts = asyncio.run(host.restricted_read(
            "in", [32, 33], "r"))
        assert found == {32: -2.0, 33: -2.0}
        assert attempts == 1
        assert no_sleep.calls == []
        assert engine.request_ids == ["r"]

    def test_missing_id_retried_and_merged(self, no_sleep):
        # first read lacks id 33, the retry reports it
        engine = _RetryEngine([_lp(32), _lp(32, 33)])
        host = BackendHost(_Serving(engine))
        host.read_retries = 2
        host.read_retry_backoff_s = 0.25
        found, result, attempts = asyncio.run(host.restricted_read(
            "in", [32, 33], "r"))
        assert found == {32: -2.0, 33: -2.0}
        assert attempts == 2
        assert no_sleep.calls == [0.25]  # backoff scaled by attempt
        # the retry carries a fresh request-id suffix
        assert engine.request_ids == ["r", "r-r1"]

    def test_retries_exhausted_returns_normally(self, no_sleep):
        # both attempts lack id 33: no raise, id simply absent
        engine = _RetryEngine([_lp(32), _lp(32)])
        host = BackendHost(_Serving(engine))
        host.read_retries = 2
        host.read_retry_backoff_s = 0.25
        found, result, attempts = asyncio.run(host.restricted_read(
            "in", [32, 33], "r"))
        assert 33 not in found and found[32] == -2.0
        assert attempts == 3  # 1 + 2 retries
        assert no_sleep.calls == [0.25, 0.5]
        assert engine.request_ids == ["r", "r-r1", "r-r2"]

    def test_zero_retries_single_pass(self, no_sleep):
        engine = _RetryEngine([_lp(32)])
        host = BackendHost(_Serving(engine))
        host.read_retries = 0
        found, result, attempts = asyncio.run(host.restricted_read(
            "in", [32, 33], "r"))
        assert attempts == 1 and no_sleep.calls == []

    def test_last_write_wins_per_token(self, no_sleep):
        # the retry re-reports 32 with a different value: it wins
        engine = _RetryEngine([_lp(32), {32: _FakeLogProb(-0.5),
                                         33: _FakeLogProb(-0.1)}])
        host = BackendHost(_Serving(engine))
        host.read_retries = 2
        found, result, attempts = asyncio.run(host.restricted_read(
            "in", [32, 33], "r"))
        assert found == {32: -0.5, 33: -0.1}


# ---------------------------------------------------------------------
# Change 2: the degraded fallback in _direct_read
# ---------------------------------------------------------------------

def _question(k=3):
    return CompiledQuestion(
        state="s", question="q",
        options=[DecisionOption(id=f"o{i}", description=f"d{i}")
                 for i in range(k)])


class _Renderer:
    def get_tokenizer(self):
        return _FakeTokenizer()


class _DirectServing:
    def __init__(self, engine):
        self.engine_client = engine
        self.model_config = _FakeModelConfig()
        self.base_renderer = _Renderer()

    async def _build_prompt(self, request, labels=None,
                            prompt_family=None):
        markers = [chr(ord("A") + i) for i in range(len(request.options))]
        return {"prompt_token_ids": [1]}, 1, [ord(m) for m in markers]

    def _log_inputs(self, *a, **k):
        pass


def _direct_backend(engine, retries=0, backoff=0.0):
    host = BackendHost(_DirectServing(engine))
    host.read_retries = retries
    host.read_retry_backoff_s = backoff
    return LogitBackend(host)


class TestDegradedFallback:
    def test_fallback_scores_from_window(self, no_sleep):
        # restricted read returns only marker A's logprob; the fallback
        # generative read's top-k window covers all three markers
        window = _lp(65, 66, 67, favour=66)
        engine = _RetryEngine([_lp(65), window])
        backend = _direct_backend(engine)
        res = asyncio.run(backend.read(_question(3), "rid"))
        assert res.meta["degraded"] is True
        assert res.meta["readout"] == "direct-degraded"
        # found from the window: B favoured; o0 carried over from the
        # restricted read's -2.0
        best = max(res.probabilities, key=res.probabilities.get)
        assert best == "o1"
        assert res.forward_passes == 2  # 1 restricted + 1 generative
        assert engine.request_ids[-1].endswith("-degraded")

    def test_marker_out_of_window_scores_neg_inf(self, no_sleep):
        # window holds markers A and B only: o2 (marker C) scores -inf
        # and loses the softmax, which stays well-formed
        window = _lp(65, 66, favour=65)
        engine = _RetryEngine([_lp(65), window])
        backend = _direct_backend(engine)
        res = asyncio.run(backend.read(_question(3), "rid"))
        assert res.option_logits["o2"] == -math.inf
        assert res.probabilities["o2"] == 0.0
        assert res.meta["degraded"] is True
        assert abs(sum(res.probabilities.values()) - 1.0) < 1e-9

    def test_no_fallback_when_restricted_succeeds(self, no_sleep):
        engine = _RetryEngine([_lp(65, 66, 67)])
        backend = _direct_backend(engine)
        res = asyncio.run(backend.read(_question(3), "rid"))
        assert res.meta["readout"] == "direct"
        assert "degraded" not in res.meta
        assert engine.params[-1].logprob_token_ids is not None

    def test_fallback_window_empty_raises(self, no_sleep):
        # every restricted attempt returns nothing and the generative
        # window holds no marker at all: honest failure
        engine = _RetryEngine([{}, {}])
        backend = _direct_backend(engine)
        with pytest.raises(BackendError) as e:
            asyncio.run(backend.read(_question(3), "rid"))
        assert "top-k window" in str(e.value)

    def test_fallback_no_logprobs_raises(self, no_sleep):
        # the generative read returns no logprobs at all
        class _NoLp:
            outputs = [type("_O", (), {"logprobs": None,
                                       "finish_reason": "stop"})()]

        engine = _RetryEngine([_lp(65), None])
        engine.generate_orig = engine.generate

        def gen_no_lp(engine_input, params, request_id):
            async def g():
                if request_id.endswith("-degraded"):
                    yield _FakeResult(None)
                else:
                    out = engine.outputs.pop(0)
                    yield _FakeResult([dict(out)])
            return g()

        engine.generate = gen_no_lp
        backend = _direct_backend(engine)
        with pytest.raises(BackendError) as e:
            asyncio.run(backend.read(_question(3), "rid"))
        assert "missing from the gathered logprobs" in str(e.value)


# ---------------------------------------------------------------------
# composition: retries then fallback, worst case bounded
# ---------------------------------------------------------------------

class TestComposition:
    def test_retries_then_fallback_counts_all_passes(self, no_sleep):
        # 1 restricted + 2 retries all lack markers B/C; the fallback
        # window covers everything: forward_passes = 4
        window = _lp(65, 66, 67, favour=67)
        engine = _RetryEngine([_lp(65), _lp(65), _lp(65), window])
        backend = _direct_backend(engine, retries=2, backoff=0.25)
        res = asyncio.run(backend.read(_question(3), "rid"))
        assert res.forward_passes == 4
        assert res.meta["degraded"] is True
        assert no_sleep.calls == [0.25, 0.5]
        assert max(res.probabilities, key=res.probabilities.get) == "o2"
