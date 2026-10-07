"""Offline suite: request validation, backend spec, readout maths, the
canvas read with a fake engine, concurrency ordering, plugin isolation.
No GPU, no real vLLM install (stubs in vllm_stubs.py)."""
from __future__ import annotations

import asyncio
import json
import math

import pytest
from vllm.entrypoints.generate.decisions.limits import (
    DecisionLimits, set_limits_for_tests)

from vllm.entrypoints.generate.decisions.systemone_protocol import (
    SystemOneRequest,
)
from vllm.entrypoints.generate.decisions.compile import compile_question
from vllm.entrypoints.generate.decisions.protocol import (
    DecisionsQuery as UnifiedDecisionRequest,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.generate.decisions.backends import (
    BackendError,
    parse_backend_spec,
    register_backend,
    restricted_softmax,
    validation_error,
)


# ---------------------------------------------------------------------
# request validation (items 4, 5, 7, 8)
# ---------------------------------------------------------------------

def _base(**kw):
    body = {"state": "s", "questions": {"decision": {
        "type": "choice", "instructions": "q",
        "criteria": {"a": "A", "b": "B"}}}}
    body.update(kw)
    return body


class TestUnifiedValidation:
    def test_typed_options_accepted(self):
        req = UnifiedDecisionRequest(**_base())
        assert list(req.questions["decision"].criteria) == ["a", "b"]

    def test_shorthand_rejected(self):
        # the one-question shorthand was removed in 0.2.0
        with pytest.raises(ValueError):
            UnifiedDecisionRequest(state="s", question="q", options=[
                {"id": "a", "description": "A"},
                {"id": "b", "description": "B"}])

    def test_empty_questions_rejected(self):
        with pytest.raises(ValueError):
            UnifiedDecisionRequest(state="s", questions={})

    def test_seed_ge_zero(self):
        req = UnifiedDecisionRequest(**_base(seed=7))
        assert req.seed == 7
        with pytest.raises(ValueError):
            UnifiedDecisionRequest(**_base(seed=-1))

    def test_noul_requires_true_false(self):
        with pytest.raises(ValueError):
            UnifiedDecisionRequest(state="s", questions={
                "q": {"type": "noul", "instructions": "i",
                      "criteria": {"yes": "y", "no": "n"}}})

    def test_noul_accepts_true_false(self):
        req = UnifiedDecisionRequest(state="s", questions={
            "q": {"type": "noul", "instructions": "i",
                  "criteria": {"true": "t", "false": "f"}}})
        assert req.questions["q"].type == "noul"

    def test_multi_score_any_levels(self):
        # multi-form score criteria is an ordered list; ids '0'..'k-1'
        # are derived at compile time, so any level count is valid.
        req = UnifiedDecisionRequest(state="s", questions={
            "q": {"type": "score", "instructions": "i",
                  "criteria": ["Calm", "Angry"]}})
        assert req.questions["q"].type == "score"


class TestSystemOneOverrides:
    def test_calibration_temperature_rejected(self):
        with pytest.raises(Exception):
            SystemOneRequest(model="m", state="s", questions={
                "q": {"type": "noul", "instructions": "i",
                      "criteria": {"true": "t", "false": "f"}}},
                calibration_temperature=0.5)

    def test_seed_rejected(self):
        with pytest.raises(Exception):
            SystemOneRequest(model="m", state="s", questions={
                "q": {"type": "noul", "instructions": "i",
                      "criteria": {"true": "t", "false": "f"}}},
                seed=1)

    def test_unknown_extra_allowed(self):
        req = SystemOneRequest(model="m", state="s", questions={
            "q": {"type": "noul", "instructions": "i",
                  "criteria": {"true": "t", "false": "f"}}},
            some_future_field=1)
        assert req.model == "m"


# ---------------------------------------------------------------------
# backend spec + softmax (item 15 core coverage)
# ---------------------------------------------------------------------

class TestParseBackendSpec:
    def test_bare_name(self):
        assert parse_backend_spec("logit") == ("logit", {})

    def test_kwargs(self):
        name, kw = parse_backend_spec("canvas:samples=5,max_steps=2")
        assert name == "canvas" and kw == {"samples": 5, "max_steps": 2}

    def test_bool_and_float(self):
        _, kw = parse_backend_spec("x:verbose=true,t=0.5")
        assert kw == {"verbose": True, "t": 0.5}


class TestRestrictedSoftmax:
    def test_sums_to_one(self):
        p = restricted_softmax({"a": -0.1, "b": -2.3})
        assert abs(sum(p.values()) - 1.0) < 1e-9

    def test_argmax_invariant_under_temperature(self):
        d = {"a": -0.1, "b": -2.3}
        assert max(restricted_softmax(d, temperature=0.5),
                   key=lambda k: restricted_softmax(d, temperature=0.5)[k]) \
            == max(d, key=d.get)

    def test_validation_error_shape(self):
        err = validation_error("boom")
        assert err.error.type == "invalid_request_error"
        assert err.error.code == 422
        assert err.error.message == "boom"


# ---------------------------------------------------------------------
# compile_question (all three types)
# ---------------------------------------------------------------------

def _sysone(questions, **kw):
    return UnifiedDecisionRequest(state="s", questions=questions, **kw)


class TestCompileQuestion:
    def test_choice(self):
        r = _sysone({"q": {"type": "choice", "instructions": "i",
                           "criteria": {"a": "x", "b": "y"}}})
        dr = compile_question(r, "q")
        assert [o.id for o in dr.options] == ["a", "b"]
        assert dr.qtype == "choice"

    def test_noul(self):
        r = _sysone({"q": {"type": "noul", "instructions": "i",
                           "criteria": {"true": "t", "false": "f"}}})
        dr = compile_question(r, "q")
        assert [o.id for o in dr.options] == ["true", "false"]
        assert dr.qtype == "noul"

    def test_score_ids_are_indices(self):
        r = _sysone({"q": {"type": "score", "instructions": "i",
                           "criteria": ["Calm", "Frustrated", "Angry"]}})
        dr = compile_question(r, "q")
        assert [o.id for o in dr.options] == ["0", "1", "2"]
        assert dr.qtype == "score"

    def test_seed_passthrough(self):
        r = _sysone({"q": {"type": "choice", "instructions": "i",
                           "criteria": {"a": "x", "b": "y"}}}, seed=42)
        assert compile_question(r, "q").seed == 42


# ---------------------------------------------------------------------
# canvas readout with a fake engine (item 1 acceptance)
# ---------------------------------------------------------------------

class _FakeLogProb:
    def __init__(self, logprob):
        self.logprob = logprob


class _FakeOutput:
    def __init__(self, logprobs):
        self.logprobs = logprobs
        self.finish_reason = "stop"


class _FakeResult:
    def __init__(self, logprobs):
        self.outputs = [_FakeOutput(logprobs)]


class _FakeTokenizer:
    """Char-level codec with Qwen-like pair merging: uppercase letter
    pairs encode to ONE merged token (id = 1000 + 26*ord(x)+ord(y)),
    everything else is per-character (id = ord(char)). Mirrors the
    property the wide-direct read relies on: some two-character labels
    are single tokens that decode to the full label.
    """

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            if (i + 1 < len(text) and "A" <= text[i] <= "Z"
                    and "A" <= text[i + 1] <= "Z"):
                out.append(1000 + 26 * (ord(text[i]) - 65)
                           + (ord(text[i + 1]) - 65))
                i += 2
            else:
                out.append(ord(text[i]))
                i += 1
        return out or [0]

    def decode(self, ids):
        out = []
        for i in ids:
            if i >= 1000:
                i -= 1000
                out.append(chr(65 + i // 26) + chr(65 + i % 26))
            else:
                out.append(chr(i))
        return "".join(out)


class _FakeModelConfig:
    max_model_len = 200_000
    max_logprobs = 600

    def get_vocab_size(self):
        return 262144


class _FakeRenderer:
    def get_tokenizer(self):
        return _FakeTokenizer()

    chat_template = None
    chat_template_content_format = "string"

    def build_chat_params(self, *a, **k):
        return _ChatParamsShim()

    def build_tok_params(self, *a, **k):
        return {}


class _ChatParamsShim:
    def with_defaults(self, kw):
        return self


class _FakeBaseRenderer:
    def get_tokenizer(self):
        return _FakeTokenizer()

    async def render_chat_async(self, conversations, chat_params,
                                tok_params):
        # token-level fake: the "prompt ids" encode the user content so
        # downstream append-stability checks run against a real string
        text = json.dumps(conversations[0][-1]["content"])
        return None, [{"prompt_token_ids":
                       self.get_tokenizer().encode(text,
                                                   add_special_tokens=False)}]


class _FakeEngine:
    """Serves one_read-shaped results; position-0 logprobs favour label
    B unless the noise draw index says otherwise."""

    def __init__(self, favour="B", omit_ids=()):
        self.favour = favour
        self.omit_ids = set(omit_ids)
        self.calls = 0

    def generate(self, engine_input, params, request_id):
        async def gen():
            self.calls += 1
            want = list(params.logprob_token_ids)
            favour = ord(self.favour)
            lp = {}
            for tid in want:
                if tid in self.omit_ids:
                    continue
                lp[tid] = _FakeLogProb(
                    -0.1 if tid == favour else -2.0 - 0.01 * want.index(tid))
            yield _FakeResult([lp] + [None] * 63)
        return gen()


class _FakeServing:
    def __init__(self, engine):
        self.engine_client = engine
        self.model_config = _FakeModelConfig()
        self.base_renderer = _FakeBaseRenderer()

    async def _build_prompt(self, request):
        return {"prompt_token_ids": [1, 2, 3]}, 3, \
            [ord(c) for c in [o.id for o in request.options]]

    def _log_inputs(self, *a, **k):
        pass


def _canvas_request(**kw):
    from vllm.entrypoints.generate.decisions.protocol import (
        DecisionOption, CompiledQuestion)
    body = {"state": "s", "question": "q",
            "options": [DecisionOption(id="A", description="a"),
                        DecisionOption(id="B", description="b")]}
    body.update(kw)
    return CompiledQuestion(**body)


class TestCanvasRead:
    def _backend(self, favour="B", **kw):
        from vllm.entrypoints.generate.decisions.backends.canvas_backend \
            import CanvasBackend
        from vllm.entrypoints.generate.decisions.backends.host import (
            BackendHost)
        engine = _FakeEngine(favour=favour)
        serving = _FakeServing(engine)
        return CanvasBackend(BackendHost(serving), **kw), engine

    def test_argmax_is_favoured_label(self):
        backend, _ = self._backend(favour="B")
        result = asyncio.run(backend.read(_canvas_request(), "req"))
        assert max(result.probabilities, key=result.probabilities.get) == "B"

    def test_samples_reproducible_from_seed(self):
        backend, _ = self._backend(samples=3)
        r1 = asyncio.run(backend.read(_canvas_request(seed=7), "req1"))
        r2 = asyncio.run(backend.read(_canvas_request(seed=7), "req2"))
        assert r1.probabilities == r2.probabilities
        assert r1.forward_passes == 3
        assert r1.meta["samples"]["n"] == 3

    def test_missing_label_raises(self):
        backend, engine = self._backend()

        # remove option C's label id from the fake output: add an option
        # whose label the engine never reports
        from vllm.entrypoints.generate.decisions.protocol import (
            DecisionOption, CompiledQuestion)
        req = CompiledQuestion(
            state="s", question="q",
            options=[DecisionOption(id="A", description="a"),
                     DecisionOption(id="C", description="c")])
        engine.omit_ids = {ord("C")}
        with pytest.raises(BackendError):
            asyncio.run(backend.read(req, "req"))

    def test_option_logits_resoftmax_matches(self):
        backend, _ = self._backend()
        result = asyncio.run(backend.read(_canvas_request(), "req"))
        p = restricted_softmax(result.option_logits)
        for k, v in p.items():
            assert abs(v - result.probabilities[k]) < 1e-9

    def test_width_below_two_refused(self):
        with pytest.raises(ValueError):
            self._backend(canvas_width=1)


# ---------------------------------------------------------------------
# truncate_state isolation
# ---------------------------------------------------------------------

def _laya_host(tmp_path, laya_config="hf"):
    """A BackendHost stand-in for a LayaForDecision checkpoint.
    laya_config: "hf" (on hf_config), "file" (laya_config.json next to
    the weights) or None (nowhere)."""
    import json
    import types
    hf = types.SimpleNamespace()
    if laya_config == "hf":
        hf.laya_config = {"max_len": 2048, "head_max_len": 512}
    if laya_config == "file":
        (tmp_path / "laya_config.json").write_text(
            json.dumps({"max_len": 1024, "head_max_len": 256}))
    mc = types.SimpleNamespace(architectures=["LayaForDecision"],
                               model=str(tmp_path), hf_config=hf)
    return types.SimpleNamespace(model_config=mc,
                                 architectures=["LayaForDecision"],
                                 tokenizer=_FakeTokenizer())


class TestTruncateStateIsolation:
    def test_override_does_not_leak_startup_flag(self, tmp_path):
        from vllm.entrypoints.generate.decisions.backends import get_backend
        host = _laya_host(tmp_path)
        # the startup instance, configured with truncate_state=true
        startup_enc = get_backend("encoder", host, truncate_state=True)
        # a per-request override is a second instance with defaults
        override = get_backend("encoder", host)
        assert override is not startup_enc
        assert startup_enc.truncate_state is True
        assert override.truncate_state is False
        assert not hasattr(host, "encoder_truncate_state")


class TestLayaConfigFallback:
    def test_fallback_reads_laya_config_json(self, tmp_path):
        from vllm.entrypoints.generate.decisions.backends.encoder_backend \
            import laya_prompt_builder
        assert laya_prompt_builder(_laya_host(tmp_path, "file")) is not None

    def test_no_config_anywhere_raises(self, tmp_path):
        from vllm.entrypoints.generate.decisions.backends.encoder_backend \
            import laya_prompt_builder
        with pytest.raises(ValueError, match="max_len and head_max_len"):
            laya_prompt_builder(_laya_host(tmp_path, None))

    def test_laya_checkpoint_selects_encoder(self, tmp_path):
        from vllm.entrypoints.generate.decisions.backends import (
            select_backend_name)
        assert select_backend_name(["LayaForDecision"]) == "encoder"
        assert select_backend_name(
            ["DiffusionGemmaForBlockDiffusion"]) == "canvas"
        assert select_backend_name(["Qwen3ForCausalLM"]) == "logit"


# ---------------------------------------------------------------------
# answer_query end-to-end through the stubs
# ---------------------------------------------------------------------

class _UnifiedServing:
    """Minimal ServingDecisions host: real answer_query, fake engine
    and renderer, a registered backend the tests control."""

    def __init__(self, fail_qids=(), delays=None):
        from vllm.entrypoints.generate.decisions.serving import (
            ServingDecisions)
        from vllm.entrypoints.generate.decisions.backends import (
            register_backend, BackendResult)

        engine = _FakeEngine()
        self.engine = engine
        self.fail_qids = set(fail_qids)
        self.delays = delays or {}

        outer = self
        backend_name = f"fake-{id(self)}"

        class _FakeBackend:
            name = backend_name

            def __init__(self, serving, **kwargs):
                pass

            async def read(self, request, request_id):
                qid = request_id.rsplit("-", 1)[-1]
                await asyncio.sleep(outer.delays.get(qid, 0.0))
                if qid in outer.fail_qids:
                    raise BackendError(f"boom in {qid}")
                return BackendResult(
                    option_logits={"A": -0.1, "B": -2.3},
                    probabilities={"A": 0.9, "B": 0.1},
                    forward_passes=1, meta={"input_tokens": 3})

        register_backend(backend_name, _FakeBackend)

        self._backend_class = _FakeBackend
        engine.model_config = _FakeModelConfig()
        engine.errored = False
        engine.dead_error = RuntimeError("engine dead")
        renderer = _FakeRenderer()
        renderer.renderer = _FakeBaseRenderer()
        self.serving = ServingDecisions(
            engine, _FakeModels(), renderer,
            request_logger=None, default_backend=backend_name)


class _FakeModels:
    def model_name(self, _):
        return "test-model"


class TestCreateDecisions:
    """Concurrency ordering through the real answer_query (the full
    partial-failure contract is in test_unify.py)."""

    def _request(self, questions):
        return UnifiedDecisionRequest(model="m", state="s",
                                      questions=questions)

    def _three(self):
        return self._request({
            q: {"type": "choice", "instructions": "decide",
                "criteria": {"A": "a", "B": "b"}}
            for q in ("q1", "q2", "q3")})

    def test_backend_refusal_is_a_partial_failure_naming_question(self):
        s = _UnifiedServing(fail_qids={"q2"}).serving
        result = asyncio.run(s.answer_query(self._three(), None))
        assert set(result["answers"]) == {"q1", "q3"}
        assert "q2" in result["partial_failures"]["q2"]

    def test_all_failed_names_first_in_request_order(self):
        # q3 fails fast, q1 fails slowly: the error must name q1.
        s = _UnifiedServing(
            fail_qids={"q1", "q2", "q3"},
            delays={"q1": 0.05, "q3": 0.0}).serving
        result = asyncio.run(s.answer_query(self._three(), None))
        assert isinstance(result, ErrorResponse)
        assert "q1" in result.error.message
        assert "q3" not in result.error.message

    def test_all_succeed_answers_and_usage_sum(self):
        s = _UnifiedServing().serving
        req = self._request({
            q: {"type": "choice", "instructions": "decide",
                "criteria": {"A": "a", "B": "b"}} for q in ("q1", "q2")})
        result = asyncio.run(s.answer_query(req, None))
        assert set(result["answers"]) == {"q1", "q2"}
        assert result["usage"]["input_tokens"] == 6
        assert result["usage"]["output_tokens"] == 0


# ---------------------------------------------------------------------
# plugin isolation (item 11e)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# limits, readout routing, wide-direct
# ---------------------------------------------------------------------

class TestLimits:
    def test_expand_char_spec_ranges_and_errors(self):
        from vllm.entrypoints.generate.decisions.limits import (
            expand_char_spec)
        assert expand_char_spec("A-Z") == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert expand_char_spec("A-Z,a-z") == list(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        assert expand_char_spec("1-3,X") == ["1", "2", "3", "X"]
        with pytest.raises(ValueError):
            expand_char_spec("Z-A")
        with pytest.raises(ValueError):
            expand_char_spec("AB")
        with pytest.raises(ValueError):
            expand_char_spec("A-Z,A-B")

    def test_relationship_validation(self):
        from vllm.entrypoints.generate.decisions.limits import (
            DecisionLimits)
        with pytest.raises(ValueError):
            DecisionLimits(markers=tuple("AB"), shortlist=3)
        with pytest.raises(ValueError):
            DecisionLimits(markers=tuple("AB"), max_score_levels=3)

    def test_set_for_tests(self):
        from vllm.entrypoints.generate.decisions.limits import (
            DecisionLimits, get_limits, set_limits_for_tests)
        set_limits_for_tests(DecisionLimits(markers=tuple("ABC"),
                                            shortlist=2,
                                            max_score_levels=2))
        try:
            assert get_limits().markers == ("A", "B", "C")
        finally:
            set_limits_for_tests(DecisionLimits())


class TestReadoutRouting:
    def _serving_with_limits(self, markers="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        from vllm.entrypoints.generate.decisions.serving import (
            ServingDecisions)
        from vllm.entrypoints.generate.decisions.limits import (
            DecisionLimits, set_limits_for_tests)
        set_limits_for_tests(DecisionLimits(
            markers=tuple(markers)))
        engine = _FakeEngine()
        engine.model_config = _FakeModelConfig()
        engine.errored = False
        engine.dead_error = RuntimeError("dead")
        renderer = _FakeRenderer()
        renderer.renderer = _FakeBaseRenderer()
        return ServingDecisions(
            engine, _FakeModels(), renderer,
            request_logger=None, default_backend="logit")

    def test_auto_above_markers_is_wide_direct(self):
        # past the marker count, auto picks wide-direct
        # while k is within its capacity on this tokenizer. The fake
        # tokenizer merges every uppercase pair, so capacity far exceeds
        # 27; the beyond-capacity branch is covered in test_unify.py.
        import asyncio
        from vllm.entrypoints.generate.decisions.limits import (
            DecisionLimits, set_limits_for_tests)
        from vllm.entrypoints.generate.decisions.protocol import (
            DecisionOption, CompiledQuestion)
        s = self._serving_with_limits()
        k = 27  # past the default marker count
        req = CompiledQuestion(
            state="s", question="q",
            options=[DecisionOption(id=f"o{i}", description=f"d{i}")
                     for i in range(k)])
        backend = s.decision_backend
        try:
            result = asyncio.run(backend.read(req, "req"))
            assert result.meta.get("readout") == "wide-direct"
        finally:
            from vllm.entrypoints.generate.decisions.limits import (
                set_limits_for_tests as _slft)
            _slft(DecisionLimits())

    def test_prefixed_readout_removed(self):
        from vllm.entrypoints.generate.decisions.protocol import (
            DecisionOption, CompiledQuestion)
        s = self._serving_with_limits()
        req = CompiledQuestion(
            state="s", question="q", backend_options={"readout": "prefixed"},
            options=[DecisionOption(id=f"o{i}", description=f"d{i}")
                     for i in range(30)])
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="readout"):
            asyncio.run(s.decision_backend.read(req, "req"))

    def test_direct_refuses_past_markers(self):
        import asyncio
        from vllm.entrypoints.generate.decisions.limits import (
            DecisionLimits, set_limits_for_tests)
        s = self._serving_with_limits()
        from vllm.entrypoints.generate.decisions.protocol import (
            DecisionOption, CompiledQuestion)
        req = CompiledQuestion(
            state="s", question="q", backend_options={"readout": "direct"},
            options=[DecisionOption(id=f"o{i}", description=f"d{i}")
                     for i in range(30)])
        backend = s.decision_backend
        try:
            with pytest.raises(BackendError, match="marker count"):
                asyncio.run(backend.read(req, "req"))
        finally:
            set_limits_for_tests(DecisionLimits())


class TestEngineReadLimit:
    """The engine's restricted-read limit (max_logprobs x the vLLM
    token-id cap) bounds every one-pass read: auto routes around it,
    explicit readouts refuse with the raising setting named, and the
    two-stage shortlist never exceeds it."""

    def _serving_with(self, max_logprobs=600, token_id_cap=600,
                      markers="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        from vllm.entrypoints.generate.decisions.serving import (
            ServingDecisions)
        from vllm.entrypoints.generate.decisions.limits import (
            DecisionLimits, set_limits_for_tests)
        set_limits_for_tests(DecisionLimits(markers=tuple(markers)))
        engine = _FakeEngine()
        engine.model_config = _FakeModelConfig()
        engine.model_config.max_logprobs = max_logprobs
        engine.errored = False
        engine.dead_error = RuntimeError("dead")
        renderer = _FakeRenderer()
        renderer.renderer = _FakeBaseRenderer()
        s = ServingDecisions(
            engine, _FakeModels(), renderer,
            request_logger=None, default_backend="logit")
        # the token-id cap comes from the running vLLM; tests pin it
        if token_id_cap is None:
            s.host.token_id_cap = None
        else:
            s.host.token_id_cap = token_id_cap
        if max_logprobs is not None and max_logprobs >= 0:
            limit = max_logprobs
        else:
            limit = None
        if token_id_cap is not None:
            limit = token_id_cap if limit is None else min(limit, token_id_cap)
        s.host.read_limit = limit
        return s

    def _req(self, k, readout=None):
        from vllm.entrypoints.generate.decisions.protocol import (
            DecisionOption, CompiledQuestion)
        opts = {"readout": readout} if readout else None
        return CompiledQuestion(
            state="s", question="q", backend_options=opts,
            options=[DecisionOption(id=f"o{i}", description=f"d{i}")
                     for i in range(k)])

    def test_negative_max_logprobs_means_uncapped(self):
        # vLLM allows max_logprobs=-1: "no cap" on the logprob window.
        # The read limit then comes from the token-id cap alone, and the
        # marker cap must not slice markers[:1].
        from vllm.entrypoints.generate.decisions.limits import (
            engine_read_limit)
        cfg = _FakeModelConfig()
        cfg.max_logprobs = -1
        limit, cap = engine_read_limit(cfg)
        assert cap == 600 and limit == 600  # the token-id cap binds
        s = self._serving_with(max_logprobs=-1)
        try:
            assert len(s.host.limits.markers) == 26  # not markers[:-1]
        finally:
            set_limits_for_tests(DecisionLimits())

    def test_no_limits_at_all_returns_none(self):
        # no max_logprobs and no token-id cap: uncapped, not a crash
        from vllm.entrypoints.generate.decisions.limits import (
            engine_read_limit)
        cfg = _FakeModelConfig()
        cfg.max_logprobs = None
        import vllm.entrypoints.generate.decisions.limits as _lim
        real = _lim.engine_read_limit
        # simulate a vLLM without the constant
        class _NoCap:
            @staticmethod
            def __getattr__(name):
                raise AttributeError(name)
        import sys
        saved = sys.modules.get("vllm.sampling_params")
        try:
            import types
            fake = types.ModuleType("vllm.sampling_params")
            sys.modules["vllm.sampling_params"] = fake
            limit, cap = engine_read_limit(cfg)
            assert limit is None and cap is None
        finally:
            if saved is not None:
                sys.modules["vllm.sampling_params"] = saved
        # and an explicit wide-direct on an uncapped host is served, not
        # crashed on (k > None)
        s = self._serving_with(max_logprobs=None, token_id_cap=None)
        try:
            s.host.read_limit = None
            result = asyncio.run(
                s.decision_backend.read(self._req(40), "r"))
            assert result.meta.get("readout") == "wide-direct"
        finally:
            set_limits_for_tests(DecisionLimits())

    def test_both_limits_named_when_both_bind(self):
        # max_logprobs 20 AND token-id cap 30, k=40: the message names
        # both settings
        s = self._serving_with(max_logprobs=20, token_id_cap=30)
        try:
            with pytest.raises(BackendError) as ei:
                asyncio.run(s.decision_backend.read(
                    self._req(40, readout="wide-direct"), "r"))
            msg = str(ei.value)
            assert "--max-logprobs" in msg
            assert "MAX_LOGPROB_TOKEN_IDS" in msg
        finally:
            set_limits_for_tests(DecisionLimits())

    def test_helper_combines_caps(self):
        from vllm.entrypoints.generate.decisions.limits import (
            engine_read_limit)
        cfg = _FakeModelConfig()
        cfg.max_logprobs = 20
        limit, cap = engine_read_limit(cfg)
        assert cap == 600 and limit == 20  # max_logprobs binds
        cfg.max_logprobs = 1000
        limit, cap = engine_read_limit(cfg)
        assert limit == 600  # token-id cap binds

    def _auto_route(self, s, k):
        """(readout, logprobs) auto picks for k options, without reading."""
        seen = {}

        async def spy(question, request_id, labels=None,
                      readout_name="direct", layout_name="direct",
                      logprobs="exact"):
            seen.update(readout=readout_name, logprobs=logprobs)
            return None

        s.decision_backend._direct_read = spy
        asyncio.run(s.decision_backend.read(self._req(k), "r"))
        return seen["readout"], seen["logprobs"]

    def test_auto_topk_when_max_logprobs_small(self):
        # k=40, max_logprobs=20: an exact wide-direct read would need 40
        # one-pass logprobs the engine can't serve -> wide-direct, top-k
        s = self._serving_with(max_logprobs=20)
        try:
            assert self._auto_route(s, 40) == ("wide-direct", "top-k")
        finally:
            set_limits_for_tests(DecisionLimits())

    def test_auto_wide_direct_within_128(self):
        s = self._serving_with(max_logprobs=600, token_id_cap=128)
        try:
            result = asyncio.run(
                s.decision_backend.read(self._req(100), "r"))
            assert result.meta.get("readout") == "wide-direct"
        finally:
            set_limits_for_tests(DecisionLimits())

    def test_auto_topk_beyond_128(self):
        # the token-id cap, not max_logprobs, is the binding limit here
        s = self._serving_with(max_logprobs=600, token_id_cap=128)
        try:
            assert self._auto_route(s, 200) == ("wide-direct", "top-k")
        finally:
            set_limits_for_tests(DecisionLimits())

    def test_auto_wide_direct_at_255_when_uncapped(self):
        # regression: patched 600 + --max-logprobs 600 must serve the
        # full Jev choice size wide-direct
        s = self._serving_with(max_logprobs=600, token_id_cap=600)
        try:
            result = asyncio.run(
                s.decision_backend.read(self._req(255), "r"))
            assert result.meta.get("readout") == "wide-direct"
        finally:
            set_limits_for_tests(DecisionLimits())

    def test_explicit_wide_direct_over_limit_refused_with_message(self):
        # an explicit wide-direct past the limit is refused per question
        # with the k, the limit and which setting raises it
        s = self._serving_with(max_logprobs=20, token_id_cap=128)
        try:
            with pytest.raises(BackendError) as ei:
                asyncio.run(s.decision_backend.read(
                    self._req(100, readout="wide-direct"), "r"))
            msg = str(ei.value)
            assert "100 one-pass logprobs" in msg and "at most 20" in msg
            assert "--max-logprobs" in msg or "MAX_LOGPROB_TOKEN_IDS" in msg
        finally:
            set_limits_for_tests(DecisionLimits())

    def test_shortlist_capped_to_read_limit(self):
        # two-stage stage 2 is a direct marker read: with shortlist 16
        # but an engine read limit of 8, only 8 finalists go to stage 2
        import re
        from vllm.entrypoints.generate.decisions.backends.host import (
            BackendHost)
        from vllm.entrypoints.generate.decisions.backends.large_choice import (
            two_stage_read)
        from vllm.entrypoints.generate.decisions.backends.logit_backend import (
            LogitBackend)
        from vllm.entrypoints.generate.decisions.limits import DecisionLimits
        from vllm.entrypoints.generate.decisions.protocol import (
            DecisionOption, CompiledQuestion)

        class _Stage2Spy:
            """counts the stage-2 direct read size via the host render
            (two_stage_read renders the finalists itself)."""

            def __init__(self, host):
                self.host = host
                self.stage2_ks = []
                self._render = host.render

            async def render(self, question, labels=None):
                if len(question.options) > 2:  # stage 2, not a yes/no read
                    self.stage2_ks.append(len(question.options))
                return await self._render(question, labels=labels)

        class _Engine:
            def __init__(self):
                self.calls = 0

            def generate(self, engine_input, params, request_id):
                async def gen():
                    self.calls += 1
                    lp = {t: _FakeLogProb(-0.5)
                          for t in params.logprob_token_ids}
                    yield _FakeResult([lp])
                return gen()

        class _Renderer:
            def get_tokenizer(self):
                return _FakeTokenizer()

        class _Serving:
            def __init__(self, engine):
                self.engine_client = engine
                self.model_config = _FakeModelConfig()
                self.base_renderer = _Renderer()

            async def _build_prompt(self, request, labels=None,
                                    prompt_family=None):
                markers = [chr(ord("A") + i)
                           for i in range(len(request.options))]
                return ({"prompt_token_ids": [1]}, 1,
                        [ord(m) for m in markers])

        engine = _Engine()
        serving = _Serving(engine)
        host = BackendHost(serving)
        host.read_limit = 8
        spy = _Stage2Spy(host)
        host.render = spy.render
        req = CompiledQuestion(
            state="s", question="q",
            options=[DecisionOption(id=f"o{i}", description=f"d{i}")
                     for i in range(40)])
        res = asyncio.run(two_stage_read(
            LogitBackend(host), req, "rid",
            DecisionLimits(shortlist=16)))
        assert spy.stage2_ks == [8]  # 16 finalists asked, 8 read


class TestWideDirect:
    """The wide-direct read."""

    def _markers(self, tok, n_pairs=10):
        from vllm.entrypoints.generate.decisions.backends.large_choice \
            import wide_direct_markers
        return wide_direct_markers(tok, tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

    def test_marker_list_order_and_filtering(self):
        tok = _FakeTokenizer()
        markers = self._markers(tok)
        # plain markers first, in order
        assert markers[:26] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        # then single-token merged pairs, lexicographic
        pairs = markers[26:]
        assert pairs == sorted(pairs)
        assert all(len(m) == 2 for m in pairs)
        # every merged pair encodes to one token, distinct ids
        ids = [tok.encode(m)[0] for m in markers]
        assert len(ids) == len(set(ids))
        # the fake merges ALL 676 pairs; nothing non-merging to filter
        assert len(pairs) == 676

    def test_merged_pairs_filtered_when_not_single_token(self):
        from vllm.entrypoints.generate.decisions.backends.large_choice \
            import wide_direct_markers

        class _NoMerge(_FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                if len(text) == 2 and all("A" <= c <= "Z" for c in text):
                    # encodes to two tokens: NOT usable as one marker
                    return [ord(text[0]), ord(text[1])]
                return super().encode(text, add_special_tokens=False)

        markers = wide_direct_markers(_NoMerge(),
                                      tuple("ABC"))
        assert markers == ["A", "B", "C"]

    def _serving(self, k=40):
        from vllm.entrypoints.generate.decisions.serving import (
            ServingDecisions)
        from vllm.entrypoints.generate.decisions.limits import (
            DecisionLimits, set_limits_for_tests)
        set_limits_for_tests(DecisionLimits(max_options=1024))
        engine = _FakeEngine()
        engine.model_config = _FakeModelConfig()
        engine.errored = False
        engine.dead_error = RuntimeError("dead")
        renderer = _FakeRenderer()
        renderer.renderer = _FakeBaseRenderer()
        s = ServingDecisions(engine, _FakeModels(), renderer,
                             request_logger=None, default_backend="logit")
        s._log_inputs = lambda *a, **k: None
        req = self._req(k)
        return s, s.decision_backend, req

    def _req(self, k, **kw):
        from vllm.entrypoints.generate.decisions.protocol import (
            DecisionOption, CompiledQuestion)
        return CompiledQuestion(
            state="s", question="q",
            options=[DecisionOption(id=f"o{i}", description=f"d{i}")
                     for i in range(k)],
            backend_options={"readout": "wide-direct"}, **kw)

    def test_one_engine_call_per_question(self):
        import asyncio
        s, backend, req = self._serving(k=40)
        asyncio.run(backend.read(req, "req"))
        assert s.engine_client.calls == 1

    def test_over_capacity_is_error_never_fallback(self):
        import asyncio
        s, backend, req = self._serving(k=800)
        try:
            result = asyncio.run(backend.read(req, "req"))
            assert result.meta.get("readout") != "two-stage"
        except BackendError:
            pass

    def test_meta_fields(self):
        import asyncio
        s, backend, req = self._serving(k=40)
        result = asyncio.run(backend.read(req, "req"))
        assert result.meta.get("readout") == "wide-direct"
        assert result.meta.get("label_layout") == "merged-pairs"
        assert result.forward_passes == 1

    def test_direct_small_k_still_golden(self):
        # k <= 16 via plain direct: render byte-identical to golden
        import asyncio
        import json as _json
        s, backend, req = self._serving(k=4)
        req.backend_options = {"readout": "direct"}
        result = asyncio.run(backend.read(req, "req"))
        assert result.meta.get("readout") == "direct"
        assert result.meta.get("label_layout") == "direct"


class TestMaxLogprobsCap:
    """Default markers cap to max_logprobs with a
    warning; explicitly configured markers refuse to start."""

    def _serving_with(self, markers_env=None, max_logprobs=20):
        from vllm.entrypoints.generate.decisions.serving import (
            ServingDecisions)
        from vllm.entrypoints.generate.decisions.limits import (
            DecisionLimits, set_limits_for_tests)
        set_limits_for_tests(DecisionLimits())
        if markers_env is not None:
            import os
            os.environ["VLLM_TYPED_DECISIONS_MARKERS"] = markers_env
        try:
            engine = _FakeEngine()
            engine.model_config = _FakeModelConfig()
            engine.model_config.max_logprobs = max_logprobs
            engine.errored = False
            engine.dead_error = RuntimeError("dead")
            renderer = _FakeRenderer()
            renderer.renderer = _FakeBaseRenderer()
            return ServingDecisions(
                engine, _FakeModels(), renderer,
                request_logger=None, default_backend="logit")
        finally:
            if markers_env is not None:
                import os
                del os.environ["VLLM_TYPED_DECISIONS_MARKERS"]

    def test_default_markers_capped_with_warning(self, caplog):
        import logging
        from unittest.mock import patch
        from vllm.entrypoints.generate.decisions.limits import get_limits
        from vllm.entrypoints.generate.decisions import serving as srv
        with patch.object(srv.logger, "warning") as warn:
            self._serving_with(markers_env=None, max_logprobs=20)
        assert warn.called
        assert "max_logprobs" in warn.call_args[0][0]
        assert len(get_limits().markers) == 20

    def test_explicit_markers_refuse_to_start(self):
        import pytest
        with pytest.raises(ValueError, match="max_logprobs"):
            self._serving_with(markers_env="A-Z", max_logprobs=20)

    def test_within_cap_untouched(self):
        from vllm.entrypoints.generate.decisions.limits import get_limits
        self._serving_with(markers_env=None, max_logprobs=64)
        assert len(get_limits().markers) == 26


class TestRendersDiffer:
    """The direct render through the REAL _build_prompt path."""

    def _serving(self, markers="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        from vllm.entrypoints.generate.decisions.serving import (
            ServingDecisions)
        from vllm.entrypoints.generate.decisions.limits import (
            DecisionLimits, set_limits_for_tests)
        set_limits_for_tests(DecisionLimits(markers=tuple(markers)))
        engine = _FakeEngine()
        engine.model_config = _FakeModelConfig()
        engine.errored = False
        engine.dead_error = RuntimeError("dead")
        renderer = _FakeRenderer()
        renderer.renderer = _FakeBaseRenderer()
        s = ServingDecisions(engine, _FakeModels(), renderer,
                             request_logger=None, default_backend="logit")
        return s

    def _req(self, k=64, **kw):
        from vllm.entrypoints.generate.decisions.protocol import (
            DecisionOption, CompiledQuestion)
        return CompiledQuestion(
            state="s", question="q",
            options=[DecisionOption(id=f"o{i}", description=f"d{i}")
                     for i in range(k)], **kw)

    def test_direct_render_byte_identical_golden(self):
        import asyncio
        s = self._serving()
        # golden render captured from the same code path with no labels
        import json as _json
        from vllm.entrypoints.generate.decisions.serving import (
            _user_payload)
        opts = self._req(4).options
        golden = _json.dumps(
            {"evidence": "s", "criterion": "q",
             "options": [{"letter": m, "description": o.description}
                         for m, o in zip(["A", "B", "C", "D"], opts)]},
            ensure_ascii=False)
        assert _user_payload("s", "q", opts) == golden


class TestLabelPrompt:
    def test_wide_direct_render_uses_labels_prompt(self):
        # _build_prompt with multi-char labels switches system prompt
        # and payload field; the default render stays byte-identical.
        import json as _json
        from vllm.entrypoints.generate.decisions.serving import (
            DECISION_SYSTEM, DECISION_SYSTEM_LABELS, _user_payload)
        from vllm.entrypoints.generate.decisions.protocol import (
            DecisionOption)

        opts = [DecisionOption(id="a", description="da"),
                DecisionOption(id="b", description="db")]
        # default render: letter field, DECISION_SYSTEM text shape
        payload = _json.loads(_user_payload("s", "q", opts))
        assert set(payload["options"][0].keys()) == {"letter",
                                                     "description"}
        assert payload["options"][0]["letter"] == "A"
        # label render: label field
        payload2 = _json.loads(_user_payload("s", "q", opts,
                                             labels=["AA", "AB"],
                                             label_field="label"))
        assert payload2["options"][0]["label"] == "AA"
        assert "letter" not in payload2["options"][0]
        assert DECISION_SYSTEM_LABELS != DECISION_SYSTEM
        assert "its label" in DECISION_SYSTEM_LABELS
