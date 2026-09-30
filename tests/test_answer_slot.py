"""Where the answer goes (answer_slot.py) and the once-per-server
answer-slot self-check (ServingDecisions.check_answer_slot), offline.

- template endings that leave a reasoning block or output channel open
  are closed (GLM-5.x `<think>`, GPT-OSS `<|start|>assistant`);
- each label is read at its token in context after the prompt, so a
  tokenizer that spells a word-initial letter differently is read right,
  and one that doesn't parse control tokens out of text (Mistral) works;
- a model that isn't at its answer is found by the self-check, fixed with
  an empty reasoning block when that works, and refused otherwise.

tools/check_tokenizers.py runs the same checks on real tokenizers.
"""
from __future__ import annotations

import asyncio
import math

import pytest

from vllm.entrypoints.generate.decisions.answer_slot import (
    SlotError, answer_slots, closing_suffix, encode_suffix, knows_suffix)
from vllm.entrypoints.generate.decisions.limits import (
    DecisionLimits, set_limits_for_tests)
from vllm.entrypoints.generate.decisions.protocol import DecisionsRequest
from vllm.entrypoints.generate.decisions.serving import ServingDecisions
from vllm.entrypoints.serve.engine.protocol import ErrorResponse


@pytest.fixture(autouse=True)
def default_limits():
    set_limits_for_tests(DecisionLimits())
    yield
    set_limits_for_tests(DecisionLimits())


@pytest.fixture(autouse=True)
def fresh_startup():
    """Each test gets a fresh once-per-server startup object: the tests
    build many servings in one process, production has exactly one."""
    import vllm.entrypoints.generate.decisions.startup as _su
    _su._STARTUP = None
    yield
    _su._STARTUP = None


# ---------------------------------------------------------------------
# fake tokenizer: characters plus a few control tokens
# ---------------------------------------------------------------------

CTRL = 0x110000   # control token ids: past the last Unicode code point
WORD = 0x120000   # _SpaceTok's word-initial letter ids


class _Tok:
    """Characters are tokens (id = ord), control strings are single
    tokens (ids from CTRL, above every Unicode character). `parse_controls=False` mimics tokenizers that
    read control strings in plain text as characters (Mistral's)."""

    CONTROLS = ["<think>", "</think>", "<|turn|>"]

    def __init__(self, parse_controls=True, controls=None):
        self.parse_controls = parse_controls
        self.controls = list(controls if controls is not None
                             else self.CONTROLS)

    @property
    def all_special_ids(self):
        return [CTRL + i for i in range(len(self.controls))]

    def convert_tokens_to_ids(self, text):
        return (CTRL + self.controls.index(text)
                if text in self.controls else None)

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for j, c in enumerate(self.controls):
                if self.parse_controls and text.startswith(c, i):
                    out.append(CTRL + j)
                    i += len(c)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def decode(self, ids):
        return "".join(self.controls[i - CTRL] if i >= CTRL else chr(i)
                       for i in ids)


class _SpaceTok(_Tok):
    """SentencePiece-like: a letter alone encodes to its word-initial
    form (id WORD + letter), the same letter after other text to its
    plain form (ord)."""

    def encode(self, text, add_special_tokens=False):
        if len(text) == 1 and text.isupper():
            return [WORD + ord(text)]
        return super().encode(text)

    def decode(self, ids):
        return "".join(" " + chr(i - WORD) if i >= WORD else
                       super(_SpaceTok, self).decode([i]) for i in ids)


# ---------------------------------------------------------------------
# answer_slot.py
# ---------------------------------------------------------------------

def test_open_endings_are_closed():
    assert closing_suffix('"}]}<|assistant|><think>') == ("</think>",)
    assert closing_suffix("<|end|><|start|>assistant") == (
        "<|channel|>", "final", "<|message|>")


def test_endings_at_the_answer_are_left_alone():
    for tail in ("<|im_start|>assistant\n<think>\n\n</think>\n\n",
                 "<|start_header_id|>assistant<|end_header_id|>\n\n",
                 "<start_of_turn>model\n",
                 "<｜Assistant｜></think>",
                 '"}]}[/INST]'):
        assert closing_suffix(tail) is None


def test_suffix_control_tokens_are_looked_up_not_parsed():
    tok = _Tok(parse_controls=False)
    assert encode_suffix(tok, ("<think>", "\n\n", "</think>")) == [
        CTRL, 10, 10, CTRL + 1]
    assert knows_suffix(tok, ("<think>", "\n\n", "</think>"))
    assert not knows_suffix(tok, ("[THINK]", "[/THINK]"))


def test_labels_are_read_at_their_token_in_context():
    # alone, "A" would be the word-initial token WORD + 65; after the prompt
    # the model writes the plain "A" (65), which is what gets read
    tok = _SpaceTok()
    prompt = tok.encode("question\n")
    assert answer_slots(tok, prompt, ["A", "B"]) == [65, 66]


def test_labels_after_a_control_token_need_no_text_round_trip():
    # the prompt ends on a control token that this tokenizer doesn't
    # parse back out of text: the old round-trip check refused every
    # letter here (all four Mistral tokenizers)
    tok = _Tok(parse_controls=False)
    prompt = [ord(c) for c in "question"] + [1002]
    assert answer_slots(tok, prompt, ["A", "B", "C"]) == [65, 66, 67]


def test_a_label_that_is_not_one_token_is_refused():
    tok = _Tok()
    prompt = tok.encode("question<|turn|>")
    with pytest.raises(SlotError, match="not a single token"):
        answer_slots(tok, prompt, ["AB"])


# ---------------------------------------------------------------------
# the self-check, through the real serving code
# ---------------------------------------------------------------------

class _LP:
    def __init__(self, logprob):
        self.logprob = logprob


class _Out:
    def __init__(self, logprobs):
        self.logprobs = [logprobs]
        self.finish_reason = "length"


class _Result:
    def __init__(self, logprobs):
        self.outputs = [_Out(logprobs)]
        self.num_cached_tokens = 0


class _ModelConfig:
    max_model_len = 10**6
    max_logprobs = 600
    architectures = []


class _Engine:
    """A model that answers only when the prompt ends with
    `answers_after` (token ids); anywhere else it puts 1% on each
    option letter (it wants to write something else first)."""

    errored = False
    dead_error = RuntimeError("engine dead")

    def __init__(self, answers_after=()):
        self.model_config = _ModelConfig()
        self.answers_after = list(answers_after)
        self.prompts = []

    def generate(self, engine_input, params, request_id):
        ids = engine_input["prompt_token_ids"]
        self.prompts.append((request_id, ids))
        at_answer = ids[len(ids) - len(self.answers_after):] == \
            self.answers_after
        want = list(params.logprob_token_ids)

        async def gen():
            p = [0.9] + [0.02] * (len(want) - 1) if at_answer \
                else [0.01] * len(want)
            yield _Result({t: _LP(math.log(v)) for t, v in zip(want, p)})
        return gen()


class _BaseRenderer:
    def __init__(self, tok, ending, refuses_thinking_off=False):
        self.tok, self.ending = tok, ending
        self.refuses_thinking_off = refuses_thinking_off
        self.calls = []

    def get_tokenizer(self):
        return self.tok

    async def render_chat_async(self, conversations, chat_params,
                                tok_params):
        self.calls.append(chat_params.reasoning_effort)
        if self.refuses_thinking_off and \
                chat_params.reasoning_effort == "none":
            raise ValueError("reasoning_effort='none' is not supported "
                             "for this model")
        text = "\n".join(m["content"] for m in conversations[0])
        return None, [{"prompt_token_ids":
                       self.tok.encode(text + self.ending)}]


class _Renderer:
    chat_template = None
    chat_template_content_format = "string"

    def __init__(self, tok, ending, refuses_thinking_off=False):
        self.renderer = _BaseRenderer(tok, ending, refuses_thinking_off)


class _Models:
    def model_name(self, _):
        return "served-model"


def _serving(engine, tok=None, ending="<|turn|>",
             refuses_thinking_off=False):
    return ServingDecisions(engine, _Models(),
                            _Renderer(tok or _Tok(), ending,
                                      refuses_thinking_off),
                            request_logger=None, default_backend="logit")


BODY = {"state": "s", "questions": {"q": {
    "type": "choice", "instructions": "Which?",
    "criteria": {"a": "Alpha", "b": "Beta"}}}}


def _ask(s):
    return asyncio.run(s.create_decisions(DecisionsRequest(**BODY)))


def _probes(engine):
    return [ids for rid, ids in engine.prompts if "selfcheck" in rid]


def test_a_model_at_its_answer_passes_with_one_probe():
    engine = _Engine()
    s = _serving(engine)
    r = _ask(s)
    assert r["answers"]["q"]["choice"] == "a"
    assert len(_probes(engine)) == 1
    assert s.answer_suffix is None
    _ask(s)
    assert len(_probes(engine)) == 1  # once per server


def test_a_model_that_thinks_first_gets_an_empty_reasoning_block():
    # DeepSeek-R1 style: the template ends at the assistant turn, but the
    # model opens <think> by itself; it answers after an empty block
    tok = _Tok()
    block = tok.encode("<think>\n\n</think>\n\n")
    engine = _Engine(answers_after=block)
    s = _serving(engine, tok)
    r = _ask(s)
    assert r["answers"]["q"]["choice"] == "a"
    assert len(_probes(engine)) == 2
    assert s.answer_suffix == ("<think>", "\n\n", "</think>", "\n\n")
    question_prompt = engine.prompts[-1][1]
    assert question_prompt[-len(block):] == block


def test_a_template_left_open_is_closed_before_the_read():
    # GLM-5.x style: the template ends on <think> whatever the switch says
    tok = _Tok()
    engine = _Engine(answers_after=tok.encode("<think></think>"))
    s = _serving(engine, tok, ending="<|turn|><think>")
    r = _ask(s)
    assert r["answers"]["q"]["choice"] == "a"
    assert len(_probes(engine)) == 1
    assert s.answer_suffix is None


def test_thinking_is_switched_off_in_the_template():
    s = _serving(_Engine())
    _ask(s)
    assert set(s.renderer.renderer.calls) == {"none"}


def test_a_template_that_refuses_the_switch_renders_without_it():
    # Ministral-3 Reasoning: reasoning_effort='none' is an error; the
    # prompt renders without the switch, and the model, which then opens
    # [THINK] by itself, answers after an empty [THINK][/THINK] block
    tok = _Tok(controls=["[THINK]", "[/THINK]", "<|turn|>"])
    engine = _Engine(answers_after=tok.encode("[THINK][/THINK]"))
    s = _serving(engine, tok, refuses_thinking_off=True)
    r = _ask(s)
    assert r["answers"]["q"]["choice"] == "a"
    assert s.thinking_switch is False
    assert s.answer_suffix == ("[THINK]", "[/THINK]")
    calls = s.renderer.renderer.calls
    assert calls[:2] == ["none", None]      # refused once, then without
    assert calls[2:] == [None] * (len(calls) - 2)  # never retried


def test_a_model_not_at_its_answer_is_refused():
    engine = _Engine(answers_after=[-1])  # never at its answer
    s = _serving(engine)
    r = _ask(s)
    assert isinstance(r, ErrorResponse)
    assert r.error.code == 503
    msg = r.error.message
    assert "self-check failed" in msg
    assert "as rendered: 4.0%" in msg          # 4 probe letters x 1%
    assert "empty reasoning block" in msg      # the retry was tried too
    assert "VLLM_TYPED_DECISIONS_MIN_OPTION_MASS=0" in msg
    n = len(_probes(engine))
    assert isinstance(_ask(s), ErrorResponse)
    assert len(_probes(engine)) == n            # not probed again


def test_the_retry_needs_reasoning_tokens_in_the_tokenizer():
    engine = _Engine(answers_after=[-1])
    s = _serving(engine, _Tok(controls=["<|turn|>"]))
    assert isinstance(_ask(s), ErrorResponse)
    assert len(_probes(engine)) == 1


def test_threshold_zero_turns_the_check_off():
    set_limits_for_tests(DecisionLimits(min_option_mass=0))
    engine = _Engine(answers_after=[-1])
    s = _serving(engine)
    r = _ask(s)
    assert r["answers"]["q"]["choice"] == "a"
    assert _probes(engine) == []


def test_without_full_vocab_logprobs_the_check_is_skipped():
    engine = _Engine(answers_after=[-1])
    engine.model_config.logprobs_mode = "raw_logits"
    s = _serving(engine)
    assert not isinstance(_ask(s), ErrorResponse)
    assert len(_probes(engine)) == 1


def test_scheduled_startup_runs_once_in_a_running_loop():
    """Build the serving the way a real start does: inside a running
    loop. The self-check must run once in the background and `starting`
    must go from True to False."""
    import asyncio
    import vllm.entrypoints.generate.decisions.startup as _su
    _su._STARTUP = None

    engine = _Engine()
    seen = []

    async def main():
        s = ServingDecisions(engine, _Models(),
                             _Renderer(_Tok(), "<|turn|>"),
                             request_logger=None, default_backend="logit")
        assert s.startup.starting is True
        # first request: the background work is running, the request 503s
        r = await s.create_decisions(DecisionsRequest(**BODY))
        assert isinstance(r, ErrorResponse) and r.error.code == 503
        # wait for the background work to finish
        while s.startup.starting:
            await asyncio.sleep(0.01)
        assert s.startup.slot_check == "ok"
        assert s.startup.starting is False
        seen.append(s.startup)

    asyncio.run(main())

    # a second instance shares the startup: the check does not rerun
    async def second():
        s2 = ServingDecisions(engine, _Models(),
                              _Renderer(_Tok(), "<|turn|>"),
                              request_logger=None, default_backend="logit")
        assert s2.startup.starting is False
        assert s2.startup.slot_check == "ok"

    asyncio.run(second())
    n = len([ids for rid, ids in engine.prompts if "selfcheck" in rid])
    assert n == 1  # exactly one probe for the whole server
    assert seen and seen[0] is _su_startup()
    _su_reset()


def _su_startup():
    import vllm.entrypoints.generate.decisions.startup as _su
    return _su._STARTUP


def _su_reset():
    import vllm.entrypoints.generate.decisions.startup as _su
    _su._STARTUP = None
