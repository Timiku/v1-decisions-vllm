# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed-decision serving: one prompt, one forward pass, restricted softmax.

The mechanism is letter-logit readout (the "decision model" contract): the
server renders the choice prompt with the served chat template (thinking
disabled), runs ONE forward pass, gathers next-token logits for the option
marker tokens via the fused restricted-logprob kernel, and softmaxes over
exactly those tokens.

Prompt order follows the verified evidence-first arrangement: state (evidence) first,
then options as parenthesized letter markers with descriptions, then the
question, then the assistant generation prompt. The prompt must end where
the option letter is the natural next token: answer_slot.py closes template
endings that leave a reasoning block or output channel open, and a
once-per-server self-check (check_answer_slot) confirms it on the served
model. Each option letter is read at its token in context after the
prompt, checked single-token and tokenization-stable.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time

from fastapi import Request
from http import HTTPStatus

from vllm.entrypoints.generate.decisions.answer_slot import (
    SlotError,
    answer_slots,
    closing_suffix,
    encode_suffix,
)
from vllm.entrypoints.generate.decisions.compile import RENDER_VERSION
from vllm.entrypoints.generate.decisions import startup
from vllm.entrypoints.generate.decisions.backends import (
    AUDIT_META_KEYS,
    BackendError,
    add_known,
    get_backend,
    restricted_softmax,
    select_backend_name,
    validate_backend_options,
)
from vllm.entrypoints.generate.decisions.backends.answer_schema import (
    CONFIDENCE_FORMULA,
    build_answer,
)
from vllm.entrypoints.generate.decisions.backends.host import BackendHost
from vllm.entrypoints.generate.decisions.protocol import (
    MODEL_ALIASES,
    CompiledQuestion,
    DecisionsRequest,
)
from vllm.entrypoints.generate.decisions.question_types import (
    get_question_type,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
try:
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse
except ImportError:  # locked fork moved engine protocol to openai
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.logger import init_logger
from vllm.renderers.params import ChatParams, TokenizeParams
from vllm.utils import random_uuid

logger = init_logger(__name__)

DECISION_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly "
    "one listed option. Respond with only its uppercase letter, with no "
    "explanation or reasoning."
)

# Label prompt: used only for renders whose labels are not single
# letters (the wide-direct read's two-letter labels). Never for the
# direct read, whose render is pinned byte for byte by tests/golden/.
DECISION_SYSTEM_LABELS = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly "
    "one listed option. Respond with only its label, exactly as written, "
    "with no explanation or reasoning."
)

def _render_state(state: str | dict | list) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, allow_nan=False)

def _user_payload(state: str, question: str, options: list,
                  labels: list[str] | None = None,
                  label_field: str = "letter") -> str:
    if labels is None:
        # markers beyond the shipped calibrated A-P alphabet change the
        # render: those reads default to T=1.0 until a profile exists
        # (calibration guard, item 8).
        from .limits import get_limits
        labels = [get_limits().markers[i] for i in range(len(options))]
    return json.dumps(
        {
            "evidence": state,
            "criterion": question,
            # NOTE: no "id" field here — the direct-options-v1 payload
            # carries only label+description; adding id would shift every
            # prompt away from the format the calibration was fit on.
            # The field is "letter" for the direct read (calibrated
            # render) and "label" for multi-character label renders.
            "options": [
                {label_field: lab, "description": o.description}
                for lab, o in zip(labels, options)
            ],
        },
        ensure_ascii=False,
    )

def _drop_per_option(block: dict, ids: set) -> dict:
    """`block` without its per-option maps (dicts keyed by option ids, at
    any depth: logits, stage-1 scores, canvas stderr)."""
    def per_option(v) -> bool:
        return isinstance(v, dict) and bool(v) and set(v) <= ids
    return {k: (_drop_per_option(v, ids) if isinstance(v, dict) else v)
            for k, v in block.items() if not per_option(v)}


def shape_extra(answer: dict, levels: dict[str, str]) -> dict:
    """Apply the request's per-block `extra` levels to one answer.
    full: the block as is. basic: without per-option maps. none: the
    block left out. No `extra` key at all when every block is none."""
    extra = answer.get("extra") or {}
    ids = set(answer["probabilities"])
    shaped = {}
    for block, content in extra.items():
        level = levels.get(block, "full")
        if level == "full":
            shaped[block] = content
        elif level == "basic":
            shaped[block] = _drop_per_option(content, ids)
    if shaped:
        answer["extra"] = shaped
    else:
        answer.pop("extra", None)
    return answer


def _infer_qtype(option_ids: set[str]) -> str:
    """noul when the only options are true/false, choice otherwise."""
    return "noul" if option_ids <= {"true", "false"} else "choice"

class ServingDecisions(BaseServing):
    """Typed decisions over pluggable backends and question types."""

    def __init__(
        self,
        engine_client,
        models: OpenAIServingModels,
        online_renderer,
        *,
        request_logger: RequestLogger | None,
        default_chat_template_kwargs: dict | None = None,
        default_backend: str | None = None,
        default_backend_kwargs: dict | None = None,
    ):
        super().__init__(models, engine_client.model_config,
                         request_logger=request_logger)
        self.engine_client = engine_client
        self.renderer = online_renderer
        self.base_renderer = online_renderer.renderer
        self.default_chat_template_kwargs = dict(default_chat_template_kwargs
                                                 or {})
        # the only surface backends use (backends/host.py)
        self.host = BackendHost(self)
        # Answer-slot self-check: runs once per SERVER process at startup
        # (startup.py), shared by both serving instances. The first
        # instance schedules the background work; until it finishes every
        # decision request gets a 503 "starting".
        self.startup = startup.get_startup()

        # Default markers must fit the engine's max_logprobs (the
        # restricted gather asks for one logprob per marker). With the
        # DEFAULT marker set, cap it and warn; explicitly configured
        # markers refuse to start instead.
        from .limits import get_limits as _gl, set_limits_for_tests as _sl
        max_lp = getattr(engine_client.model_config, "max_logprobs", None)
        lim = _gl()
        # max_logprobs < 0 (vLLM's -1) means uncapped: no marker cap
        if max_lp is not None and max_lp >= 0 and len(lim.markers) > max_lp:
            if os.environ.get("VLLM_TYPED_DECISIONS_MARKERS"):
                raise ValueError(
                    f"VLLM_TYPED_DECISIONS_MARKERS={''.join(lim.markers)} "
                    f"has {len(lim.markers)} markers but the engine allows "
                    f"max_logprobs={max_lp}. Raise --max-logprobs or reduce "
                    "the marker set.")
            import dataclasses
            _sl(dataclasses.replace(lim, markers=lim.markers[:max_lp]))
            logger.warning(
                "Default marker set (%d markers) exceeds the engine's "
                "max_logprobs=%d; capping markers to the first %d. Raise "
                "--max-logprobs (e.g. --max-logprobs 32) to read more "
                "options in one pass.", len(lim.markers), max_lp, max_lp)

        # Startup backend: VLLM_TYPED_DECISIONS_BACKEND (the wiring passes
        # it as default_backend, None when unset), else the model_config's
        # decision_backend, else the backend claiming the served model's
        # architecture, else logit.
        chosen = (default_backend
                  or getattr(engine_client.model_config, "decision_backend",
                             None)
                  or select_backend_name(self.host.architectures))
        self.decision_backend = get_backend(
            chosen, self.host, **(default_backend_kwargs or {}))
        # kept for the calibration key: the backend's startup kwargs
        # shape the readouts the T was fitted on
        self.default_backend_kwargs = dict(default_backend_kwargs or {})
        # The engine's restricted-read limit, computed once: the marker
        # cap, the wide-direct gate and the two-stage shortlist all
        # derive from it (backends check host.read_limit).
        from .limits import engine_read_limit
        read_limit, token_id_cap = engine_read_limit(
            engine_client.model_config)
        self.host.read_limit = read_limit
        self.host.token_id_cap = token_id_cap
        wd_capacity = self.host.wide_direct_capacity()
        direct_capacity = len(self.host.limits.markers)
        logger.info(
            "decision readouts: direct up to %d, wide-direct up to %s, "
            "two-stage beyond (read limit %s: max_logprobs=%s, "
            "token-id cap=%s)", direct_capacity,
            wd_capacity if wd_capacity is not None else "uncapped",
            read_limit if read_limit is not None else "uncapped",
            getattr(engine_client.model_config, "max_logprobs", "none"),
            token_id_cap if token_id_cap is not None else "none")
        logger.info("decision backend: %s", chosen)
        # Startup work: the self-check belongs to the logit backend
        # only; calibration runs for any backend the CALIBRATION setting
        # turns on (on / file path, or jevbench for a logit default).
        # While any of it runs, decision requests on every backend get
        # the 503 starting gate.
        if chosen == "logit":
            self.startup.schedule(
                lambda: self.startup.run_self_check(self))
        from vllm.entrypoints.generate.decisions.calibration import (
            should_calibrate, schedule_calibration)
        should, reason = should_calibrate(self)
        if should:
            schedule_calibration(self)
        else:
            # nothing scheduled: the plan still wants the reason logged
            # once per process ("off", or the operator-temperature skip)
            if not getattr(self.startup, "_skip_logged", False):
                self.startup._skip_logged = True
                logger.info("decision calibration: %s",
                            "off" if reason == "off" else reason)

        # Speculative decoding + restricted-logprob gather: historically
        # unsafe (rejection-sampler shape mismatch, vllm#42592 family).
        # EXPERIMENTAL: warn once at startup and proceed, so parity can be
        # measured instead of assumed.
        spec_config = getattr(
            getattr(engine_client, "vllm_config", None),
            "speculative_config", None)
        if spec_config is not None:
            logger.warning(
                "decision readout under speculative decoding "
                "(SPEC_N=%s): EXPERIMENTAL - verify parity against a "
                "non-speculative tier before trusting readouts.",
                getattr(spec_config, "num_speculative_tokens", "?"))

    # ------------------------------------------------------------------
    # prompt construction
    # ------------------------------------------------------------------
    async def _build_prompt(
        self,
        request: CompiledQuestion,
        labels: list[str] | None = None,
    ) -> tuple[list, int, list[int]]:
        """Render the decision prompt through the served chat template.

        ``labels`` overrides the per-option label strings (wide-direct);
        a render with any multi-character label switches to
        DECISION_SYSTEM_LABELS and the "label" payload field. The default
        (no labels) render is the direct read, pinned by tests/golden/.
        Every label must be ONE token that decodes back to itself.

        Returns (engine_input, input_token_count, slot_label_ids).
        """
        from vllm.inputs import tokens_input
        from .limits import get_limits as _get_limits

        n_markers = len(_get_limits().markers)
        if labels is None and len(request.options) > n_markers:
            # only the logit backend's wide-direct / two-stage reads
            # handle more options than markers; every other read
            # refuses here instead of failing on a missing marker
            return self.create_error_response(
                f"{len(request.options)} options exceed the "
                f"{n_markers} single-token markers of this read; on the "
                "logit backend use readout 'auto', 'wide-direct' or "
                "'two-stage'")

        use_labels = labels is not None and any(
            len(lab) != 1 for lab in labels)
        messages = [
            {"role": "system",
             "content": DECISION_SYSTEM_LABELS if use_labels
             else DECISION_SYSTEM},
            {
                "role": "user",
                "content": _user_payload(
                    _render_state(request.state),
                    request.question,
                    request.options,
                    labels=labels,
                    label_field="label" if use_labels else "letter",
                ),
            },
        ]

        # Minimal request-shaped object for the render path. build_chat_params
        # / build_tok_params are the two entry points the chat path uses.
        # Thinking is switched off (reasoning_effort "none", which vLLM
        # also passes as enable_thinking=False): thinking before a
        # single-token readout is drift. A template that refuses the
        # switch (a reasoning-only model) renders without it, from then on;
        # the answer-slot self-check then decides how to reach the answer.
        renderer = self.renderer
        tok_params = TokenizeParams(
            max_total_tokens=self.model_config.max_model_len,
        )

        async def render(thinking_off: bool):
            req = ChatCompletionRequest(
                messages=messages,  # type: ignore[arg-type]
                model=request.model or "decisions",
                max_tokens=1,
            )
            req.add_generation_prompt = True
            if thinking_off:
                req.reasoning_effort = "none"
            chat_params = req.build_chat_params(
                renderer.chat_template,
                renderer.chat_template_content_format,
            ).with_defaults(self.default_chat_template_kwargs)
            return await self.base_renderer.render_chat_async(
                [messages], chat_params, tok_params)

        if self.startup.thinking_switch:
            try:
                _, engine_inputs = await render(thinking_off=True)
            except Exception as e:
                try:
                    _, engine_inputs = await render(thinking_off=False)
                except Exception:
                    raise e from None
                self.startup.thinking_switch = False
                logger.warning(
                    "the chat template refuses reasoning_effort='none' "
                    "(%s); rendering decision prompts without it", e)
        else:
            _, engine_inputs = await render(thinking_off=False)
        engine_input = engine_inputs[0]

        # EngineInput is a TypedDict variant; extract prompt token ids from
        # whichever shape arrived (TokensInput dict or object wrapper).
        prompt_ids = engine_input.get("prompt_token_ids") \
            if isinstance(engine_input, dict) else getattr(
                engine_input, "prompt_token_ids", None)
        if prompt_ids is None and isinstance(engine_input, dict):
            prompt_ids = (engine_input.get("prompt_token_ids")
                          or (engine_input.get("prompt") or {}).get(
                              "prompt_token_ids"))
        if prompt_ids is None:
            return self.create_error_response(
                "Render path returned an engine input without prompt_token_ids.")
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]

        tokenizer = self.base_renderer.get_tokenizer()
        letters = labels if labels is not None else [
            _get_limits().markers[i] for i in range(len(request.options))]

        # End the prompt at the answer: close a reasoning block or output
        # channel the template left open, then add the empty reasoning
        # block the self-check found this model needs (answer_slot.py).
        prompt_ids = list(prompt_ids)
        closing = closing_suffix(tokenizer.decode(prompt_ids[-16:]))
        if closing:
            prompt_ids += encode_suffix(tokenizer, closing)
        if self.startup.answer_suffix:
            prompt_ids += encode_suffix(tokenizer, self.startup.answer_suffix)

        # Each label's token as the model's next token after this prompt.
        try:
            slot_ids = answer_slots(tokenizer, prompt_ids, letters)
        except SlotError as e:
            return self.create_error_response(str(e))

        if len(prompt_ids) + 8 > self.model_config.max_model_len:
            return self.create_error_response(
                f"Decision prompt ({len(prompt_ids)} tokens) exceeds the "
                "model length budget. Refused, never truncated.")

        return tokens_input(prompt_ids), len(prompt_ids), slot_ids

    # ------------------------------------------------------------------
    # answer-slot self-check
    # ------------------------------------------------------------------
    async def _probe_mass(self) -> float | None:
        """Option mass of one easy probe question: the share of the
        model's next-token probability on the option letters. None when
        the engine doesn't report full-vocabulary log-probabilities."""
        from vllm.entrypoints.generate.decisions.backends import option_mass
        from vllm.entrypoints.generate.decisions.protocol import (
            DecisionOption)
        probe = CompiledQuestion(
            state="Paris is the capital of France.",
            question="Which city is the capital of France?",
            options=[DecisionOption(id=c, description=c)
                     for c in ("Berlin", "Paris", "Madrid", "Rome")])
        prompt = await self.host.render(probe)
        found, _ = await self.host.restricted_read(
            prompt.engine_input, prompt.slot_ids,
            f"decision-selfcheck-{random_uuid()}")
        return option_mass(self.host, found)

    async def check_answer_slot(self) -> str:
        """The shared startup self-check (startup.py): "ok" or the text
        every logit read is refused with. Present for compatibility with
        tests that call it directly; the request path reads
        `self.startup` instead."""
        return await self.startup.run_self_check(self)

    @property
    def answer_suffix(self):
        return self.startup.answer_suffix

    @answer_suffix.setter
    def answer_suffix(self, v):
        self.startup.answer_suffix = v

    @property
    def thinking_switch(self):
        return self.startup.thinking_switch

    @thinking_switch.setter
    def thinking_switch(self, v):
        self.startup.thinking_switch = v

    # ------------------------------------------------------------------
    # backends
    # ------------------------------------------------------------------
    def _resolve_backend(self, name: str | None):
        """The startup backend when name is None; otherwise the named
        backend, built once over the same host with default arguments and
        cached."""
        backend = self.decision_backend
        if name and name != backend.name:
            cache = self.__dict__.setdefault("_override_backends", {})
            if name not in cache:
                cache[name] = get_backend(name, self.host)
            backend = cache[name]
        return backend

    def resolve_temperature(self, requested: float | None
                            ) -> tuple[float, str]:
        """(T, source), highest precedence first: the request's
        calibration_temperature; the operator's
        VLLM_TYPED_DECISIONS_TEMPERATURE; the startup calibration's
        result (kept, or measured-1.0); 1.0."""
        if requested is not None:
            return requested, "request"
        from .limits import get_limits, operator_temperature_set
        limits = get_limits()
        if operator_temperature_set():
            # the operator's hand-set value: calibration never ran
            return limits.temperature, "server"
        cal = getattr(self.startup, "calibration", None)
        if cal is not None and cal.get("source") == "calibrated":
            return cal["t"], "calibrated"
        # calibration off, not run for this backend, or failed
        return 1.0, "default"

    # ------------------------------------------------------------------
    # reading: one CompiledQuestion -> one answer
    # ------------------------------------------------------------------
    async def _read_one(
        self,
        question: CompiledQuestion,
        request_id: str,
        backend,
        temperature: float,
        temperature_source: str,
    ) -> "tuple[dict, int, int | None] | ErrorResponse":
        """One compiled question through one backend read. The ONLY place
        a calibration temperature is applied. Returns (answer, prompt
        tokens, prompt tokens served from the prefix cache or None)."""
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            result = await backend.read(question, request_id)
        except BackendError as e:
            return self.create_error_response(str(e))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error during decision scoring")
            return self.create_error_response(e)

        probs = restricted_softmax(result.option_logits,
                                   temperature=temperature)
        meta = dict(result.meta or {})
        meta.pop("backend", None)
        lifted = {k: meta.pop(k, None) for k in AUDIT_META_KEYS}
        state_digest = hashlib.sha256(
            json.dumps({"state": question.state}, ensure_ascii=False,
                       sort_keys=True).encode()).hexdigest()
        extra = {
            "backend": {"name": backend.name, **meta,
                        "option_logits": result.option_logits},
            "audit": {
                "served_model": self.models.model_name(None),
                "state_sha256": state_digest,
                "confidence_formula": CONFIDENCE_FORMULA,
                "render_version": RENDER_VERSION,
                "calibration_temperature": temperature,
                "temperature_source": temperature_source,
                "forward_passes": result.forward_passes,
                "input_tokens": lifted["input_tokens"] or 0,
                "cached_input_tokens": lifted["cached_input_tokens"],
                "option_mass": lifted["option_mass"],
                "readout": lifted["readout"] or "direct",
                "label_layout": lifted["label_layout"] or "direct",
            },
        }
        qtype = question.qtype or _infer_qtype(
            {o.id for o in question.options})
        answer = build_answer(get_question_type(qtype), question.options,
                              probs, extra)
        return (answer, extra["audit"]["input_tokens"],
                extra["audit"]["cached_input_tokens"])

    # ------------------------------------------------------------------
    # POST /v1/decisions (and, projected, /v1/systemone)
    # ------------------------------------------------------------------
    def response_model_name(self, requested: str | None) -> str:
        """The request's model with aliases resolved; the served model
        name when the request named none."""
        if requested:
            return MODEL_ALIASES.get(requested, requested)
        return self.models.model_name(None)

    async def create_decisions(
        self,
        request: DecisionsRequest,
        raw_request: Request | None = None,
    ):
        """Answer every question concurrently over the shared state
        prefix. A failed question goes to partial_failures; the request
        fails only when every question failed (first failure in request
        order, with its status code)."""
        from .compile import compile_question

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            backend = self._resolve_backend(request.backend)
        except ValueError as e:
            return self.create_error_response(str(e))
        try:
            options = validate_backend_options(backend,
                                               request.backend_options)
        except ValueError as e:
            return self.create_error_response(
                str(e), status_code=HTTPStatus.UNPROCESSABLE_ENTITY)
        temperature, t_source = self.resolve_temperature(
            request.calibration_temperature)
        if self.startup.starting:
            # The startup work (self-check / calibration) is running in
            # the background: refuse until it finishes, on every backend.
            # The routers add Retry-After to 503 "starting" responses.
            return self.create_error_response(
                startup.STARTING_MESSAGE,
                status_code=HTTPStatus.SERVICE_UNAVAILABLE)
        if backend.name == "logit":
            check = await self.startup.run_self_check(self)
            if check != "ok":
                return self.create_error_response(
                    check, status_code=HTTPStatus.SERVICE_UNAVAILABLE)

        created = int(time.time())
        base_id = self._base_request_id(raw_request,
                                        request.model or "decision")
        qids = list(request.questions)
        results = await asyncio.gather(
            *(self._read_one(compile_question(request, qid, options),
                             f"decision-{base_id}-{qid}", backend,
                             temperature, t_source)
              for qid in qids),
            return_exceptions=True)

        from vllm.v1.engine.exceptions import EngineDeadError
        extra_levels = request.extra
        answers: dict[str, dict] = {}
        failures: dict[str, tuple[str, int]] = {}
        total_input = 0
        cached_parts: list[int | None] = []
        for qid, res in zip(qids, results):
            if isinstance(res, EngineDeadError) or (
                    isinstance(res, BaseException)
                    and not isinstance(res, Exception)):
                raise res
            if isinstance(res, Exception):
                failures[qid] = (f"{type(res).__name__}: {res}", 500)
                continue
            if isinstance(res, ErrorResponse):
                failures[qid] = (res.error.message, res.error.code)
                continue
            answer, n_tokens, n_cached = res
            answers[qid] = shape_extra(answer, extra_levels)
            total_input += n_tokens
            cached_parts.append(n_cached)

        if not answers:
            qid, (message, code) = next(iter(failures.items()))
            return self.create_error_response(
                f"question {qid!r} failed: {message}",
                status_code=HTTPStatus(code))

        result = {
            "id": f"decisions-{random_uuid()}",
            "object": "decisions",
            "created": created,
            "model": self.response_model_name(request.model),
            "answers": answers,
            "usage": {"input_tokens": total_input,
                      "cached_input_tokens": add_known(*cached_parts),
                      "output_tokens": 0},
        }
        if failures:
            result["partial_failures"] = {q: m for q, (m, _) in
                                          failures.items()}
        return result
