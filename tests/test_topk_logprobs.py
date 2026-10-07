"""logprobs `top-k`: the logit backend reads option markers from the
engine's plain top-k window (logprobs=k, no logprob_token_ids). No GPU
(stubs)."""
from __future__ import annotations

import asyncio
import json
import math

import pytest

from vllm.entrypoints.generate.decisions.backends import BackendError
from vllm.entrypoints.generate.decisions.backends.host import BackendHost
from vllm.entrypoints.generate.decisions.backends.logit_backend import (
    LogitBackend)
from vllm.entrypoints.generate.decisions.serving import _json_safe

from test_decisions_offline import _FakeLogProb
from test_read_retry import (  # noqa: F401  (no_sleep is a fixture)
    _DirectServing, _RetryEngine, _lp, _question, no_sleep)


def _backend(engine, logprobs="top-k", max_logprobs=20, read_limit=None):
    serving = _DirectServing(engine)
    serving.model_config.max_logprobs = max_logprobs  # instance attr only
    host = BackendHost(serving)
    host.read_retries = 0
    host.read_limit = read_limit
    return LogitBackend(host, logprobs=logprobs)


def _best(res):
    return max(res.probabilities, key=res.probabilities.get)


class TestConstructor:
    def test_default_is_exact(self):
        assert LogitBackend(object()).logprobs == "exact"

    def test_bad_logprobs_mode_refused(self):
        with pytest.raises(ValueError):
            LogitBackend(object(), logprobs="topk")


class TestTopkDirect:
    def test_one_pass_no_token_ids(self, no_sleep):
        engine = _RetryEngine([_lp(65, 66, 67, favour=66)])
        res = asyncio.run(_backend(engine).read(_question(3), "rid"))
        params = engine.params[-1]
        assert getattr(params, "logprob_token_ids", None) is None
        assert params.logprobs == 20          # the window = --max-logprobs
        assert res.forward_passes == 1
        assert res.meta["logprobs"] == "top-k"
        assert res.meta["topk_window"] == 20
        assert "floored" not in res.meta and "degraded" not in res.meta
        assert _best(res) == "o1"

    def test_out_of_window_marker_floored(self, no_sleep):
        window = _lp(65, 66, favour=65)
        window[300] = _FakeLogProb(-4.0)
        engine = _RetryEngine([window])
        res = asyncio.run(_backend(engine).read(_question(3), "rid"))
        assert res.option_logits["o2"] == -4.0
        assert res.meta["floored"] == ["o2"]
        # option_mass counts only the markers actually read: a lower bound
        expect = math.exp(-0.1) + math.exp(-2.0)
        assert res.meta["option_mass"] == pytest.approx(expect)

    def test_no_marker_in_window_raises(self, no_sleep):
        engine = _RetryEngine([{300: _FakeLogProb(-0.1)}])
        with pytest.raises(BackendError) as e:
            asyncio.run(_backend(engine).read(_question(3), "rid"))
        assert "top-k window" in str(e.value)

    def test_uncapped_engine_window_256(self, no_sleep):
        engine = _RetryEngine([_lp(65, 66, 67)])
        asyncio.run(_backend(engine, max_logprobs=-1).read(
            _question(3), "rid"))
        assert engine.params[-1].logprobs == 256

    def test_request_override_to_exact(self, no_sleep):
        engine = _RetryEngine([_lp(65, 66, 67)])
        q = _question(3)
        q.backend_options = {"logprobs": "exact"}
        res = asyncio.run(_backend(engine).read(q, "rid"))
        assert engine.params[-1].logprob_token_ids == [65, 66, 67]
        assert "logprobs" not in res.meta

    def test_request_override_to_topk(self, no_sleep):
        engine = _RetryEngine([_lp(65, 66, 67)])
        q = _question(3)
        q.backend_options = {"logprobs": "top-k"}
        res = asyncio.run(_backend(engine, logprobs="exact").read(q, "rid"))
        assert getattr(engine.params[-1], "logprob_token_ids", None) is None
        assert res.meta["logprobs"] == "top-k"


class TestTopkWide:
    def test_auto_goes_wide_direct_past_read_limit(self, no_sleep):
        # 30 options, engine read limit 20: exact would go two-stage;
        # top-k has no per-label limit, so auto stays one-pass
        # wide-direct. The 20-wide window holds the first 20 markers.
        k = 30
        ids = [65 + i for i in range(k)]
        engine = _RetryEngine([_lp(*ids[:20], favour=ids[3])])
        backend = _backend(engine, read_limit=20)
        res = asyncio.run(backend.read(_question(k), "rid"))
        assert res.meta["readout"] == "wide-direct"
        assert res.forward_passes == 1
        assert len(res.meta["floored"]) == 10
        assert _best(res) == "o3"

    def test_exact_auto_past_read_limit_reads_topk(self, no_sleep):
        # exact can't read 30 labels past a read limit of 20: auto keeps
        # the one-pass wide-direct read and takes it from the window
        k = 30
        ids = [65 + i for i in range(k)]
        engine = _RetryEngine([_lp(*ids[:20], favour=ids[3])])
        backend = _backend(engine, logprobs="exact", read_limit=20)
        res = asyncio.run(backend.read(_question(k), "rid"))
        assert getattr(engine.params[-1], "logprob_token_ids", None) is None
        assert res.meta["readout"] == "wide-direct"
        assert res.meta["logprobs"] == "top-k"
        assert res.forward_passes == 1
        assert _best(res) == "o3"

    def test_exact_auto_within_read_limit_stays_exact(self, no_sleep):
        k = 30
        ids = [65 + i for i in range(k)]
        engine = _RetryEngine([_lp(*ids, favour=ids[3])])
        backend = _backend(engine, logprobs="exact", read_limit=64)
        res = asyncio.run(backend.read(_question(k), "rid"))
        assert engine.params[-1].logprob_token_ids == ids
        assert res.meta["readout"] == "wide-direct"
        assert "logprobs" not in res.meta

    def test_auto_past_tokenizer_capacity_is_two_stage(self, no_sleep):
        # two-stage is the last resort: only past wide-direct's capacity
        k = 30
        engine = _RetryEngine([_lp(65, 66, favour=65)] * k
                              + [_lp(*[65 + i for i in range(16)])])
        backend = _backend(engine, logprobs="exact", read_limit=20)
        backend.host.wide_direct_capacity = lambda capped=True: 20
        res = asyncio.run(backend.read(_question(k), "rid"))
        assert res.meta["readout"] == "two-stage"

    def test_explicit_wide_direct_exact_over_limit_suggests_topk(
            self, no_sleep):
        backend = _backend(_RetryEngine([]), logprobs="exact", read_limit=20)
        q = _question(30)
        q.backend_options = {"readout": "wide-direct"}
        with pytest.raises(BackendError) as e:
            asyncio.run(backend.read(q, "rid"))
        assert "top-k" in str(e.value)


class TestTopkTwoStage:
    def test_two_stage_topk_reads(self, no_sleep):
        k = 4
        engine = _RetryEngine([_lp(65, 66, favour=65)] * k
                              + [_lp(65, 66, 67, 68, favour=67)])
        q = _question(k)
        q.backend_options = {"readout": "two-stage"}
        res = asyncio.run(_backend(engine).read(q, "rid"))
        assert len(engine.params) == k + 1
        assert all(getattr(p, "logprob_token_ids", None) is None for p in engine.params)
        assert _best(res) == "o2"


class TestJsonSafe:
    def test_non_finite_become_none(self):
        out = _json_safe({"a": -math.inf, "b": [1.0, math.nan],
                          "c": {"d": math.inf, "e": "x"}})
        assert out == {"a": None, "b": [1.0, None],
                       "c": {"d": None, "e": "x"}}
        json.dumps(out, allow_nan=False)
