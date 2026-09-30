# SPDX-License-Identifier: Apache-2.0
"""CanvasBackend: read-only canvas denoise for diffusion-style checkpoints.

Reads the decision from one canvas slot: the canvas is seeded with a
noise token at the answer slot followed by the turn-close marker, one
read-only denoise runs, and the logprobs of each option letter at that
slot are the option scores. With `samples=n` the read repeats with n
noise draws from one seed and the distributions are averaged.

Several questions of one request are read together (`read_many`): one
joint prompt asks them all, the canvas holds the answer template
"1: A\\n2: A..." with noise at each question's letter slot, and one
read-only denoise scores every slot at once. Questions that don't fit
the canvas, or whose letters don't tokenize to one shared slot, fall
back to `read`, as does every question when max_steps > 1: past one
step the template must be held, and patches/00572 has no
`diffusion_pinned` (upstream vLLM does).

Ported from the merged precedent's (#57250) read loop and joint template
(`examples/.../structured_server.py`).

Claims DiffusionGemma checkpoints. Per-request options
(`backend_options`): `samples`, `max_steps`, overriding the startup
values for that request.
"""

from __future__ import annotations

import asyncio
import math
import random

from pydantic import BaseModel, ConfigDict, Field

from ..protocol import CompiledQuestion
from . import (BackendError, BackendResult, add_known, cached_tokens,
               restricted_softmax)


class CanvasOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: int | None = Field(default=None, ge=1, le=64,
                                description="Noise draws averaged.")
    max_steps: int | None = Field(default=None, ge=1, le=64,
                                  description="Denoise steps per read.")


class CanvasBackend:
    calibrate_by_default = False
    """Reads decisions via one read-only canvas denoise per sample."""

    name = "canvas"
    architectures = ("DiffusionGemmaForBlockDiffusion",)
    options_model = CanvasOptions

    # a joint read's canvas holds one answer line per question; fewer
    # than this many questions read one by one
    min_joint = 2

    def __init__(self, host, canvas_width: int = 64, samples: int = 1,
                 max_steps: int = 1, turn_close: int = 106, pad: int = 0):
        if canvas_width < 2:
            raise ValueError("canvas_width must be >= 2")
        self.host = host
        self.canvas_width = canvas_width
        self.samples = max(1, int(samples))
        self.max_steps = max(1, int(max_steps))
        self.turn_close = turn_close
        self.pad = pad

    def _settings(self, question: CompiledQuestion) -> tuple[int, int]:
        opts = question.backend_options
        if isinstance(opts, dict):
            opts = CanvasOptions.model_validate(opts)
        samples = (opts and opts.samples) or self.samples
        max_steps = (opts and opts.max_steps) or self.max_steps
        return samples, max_steps

    async def read(self, question: CompiledQuestion,
                   request_id: str) -> BackendResult:
        samples, max_steps = self._settings(question)

        prompt = await self.host.render(question)
        label_ids = prompt.slot_ids

        # One seed drives every sample's noise draw, so the whole read is
        # reproducible from `seed`; the seed used is always reported.
        seed = question.seed
        if seed is None:
            seed = random.SystemRandom().randrange(2**31)
        rng = random.Random(seed)
        vocab = self.host.model_config.get_vocab_size()
        option_ids = [o.id for o in question.options]
        cached: list[int | None] = []

        async def one_read(k: int) -> dict[str, float]:
            noise = rng.randrange(vocab)
            canvas = ([noise, self.turn_close]
                      + [self.pad] * (self.canvas_width - 2))
            result = await self.host.generate(
                prompt.engine_input,
                self._canvas_params(canvas, label_ids, max_steps),
                f"{request_id}-s{k}")
            cached.append(cached_tokens(result))
            pos0 = result.outputs[0].logprobs[0]
            per_read: dict[str, float] = {}
            for oid, lid in zip(option_ids, label_ids):
                lp = pos0.get(lid)
                if lp is None:
                    raise BackendError(
                        f"canvas read returned no logprob for option "
                        f"'{oid}' (label id {lid})", [oid])
                per_read[oid] = lp.logprob
            return restricted_softmax(per_read)

        dists = await asyncio.gather(*(one_read(k) for k in range(samples)))
        meta = {"input_tokens": prompt.input_tokens * samples,
                "cached_input_tokens": add_known(*cached),
                "canvas_width": self.canvas_width,
                "max_steps": max_steps,
                "seed": seed}
        return _averaged(dists, option_ids, samples, meta)

    # ------------------------------------------------------------------
    # joint read: several questions, one canvas
    # ------------------------------------------------------------------
    async def read_many(self, questions: list[CompiledQuestion],
                        request_id: str
                        ) -> list[BackendResult | BackendError | None]:
        """Read the questions in joint canvases, as many per canvas as
        the canvas holds. None for every question left to `read`."""
        out: list = [None] * len(questions)
        if len(questions) < self.min_joint:
            return out
        first = questions[0]
        if any(q.state != first.state or q.seed != first.seed
               or q.backend_options != first.backend_options
               for q in questions):
            return out  # not one request's questions: read one by one
        if self._settings(first)[1] > 1:
            # past one step accept/renoise rewrites an unpinned template
            return out
        markers = self.host.limits.markers
        idx = [i for i, q in enumerate(questions)
               if 2 <= len(q.options) <= len(markers)]
        for n, chunk in enumerate(self._chunks(idx, questions)):
            if len(chunk) < self.min_joint:
                continue
            qs = [questions[i] for i in chunk]
            try:
                template, slots = self._template(qs)
            except _NoJoint:
                continue
            try:
                results = await self._joint_read(
                    qs, template, slots, f"{request_id}-j{n}")
            except BackendError as e:
                results = [e] * len(qs)
            for i, r in zip(chunk, results):
                out[i] = r
        return out

    def _answer_ids(self, letters: list[int]) -> list[int]:
        """Token ids of the answer lines, question k (1-based) answering
        with marker letters[k-1]."""
        markers = self.host.limits.markers
        text = "\n".join(f"{n}: {markers[li]}"
                         for n, li in enumerate(letters, 1))
        return list(self.host.tokenizer.encode(text,
                                               add_special_tokens=False))

    def _chunks(self, idx: list[int], questions: list[CompiledQuestion]):
        """Consecutive runs of `idx` whose answer template plus the turn
        close fits the canvas."""
        chunk: list[int] = []
        for i in idx:
            trial = chunk + [i]
            size = len(self._answer_ids([0] * len(trial)))
            if size + 1 > self.canvas_width and chunk:
                yield chunk
                chunk = [i]
            else:
                chunk = trial
        if chunk:
            yield chunk

    def _template(self, qs: list[CompiledQuestion]
                  ) -> tuple[list[int], list[dict]]:
        """The answer template's token ids and each question's slot:
        {"pos", "label_ids"}. Every letter must change exactly one token,
        at the same position for all of a question's letters, to a
        distinct id; otherwise _NoJoint."""
        base_letters = [0] * len(qs)
        base = self._answer_ids(base_letters)
        if len(base) + 1 > self.canvas_width:
            raise _NoJoint
        slots = []
        for qi, q in enumerate(qs):
            pos = None
            ids = [0] * len(q.options)
            for li in range(1, len(q.options)):
                letters = list(base_letters)
                letters[qi] = li
                e = self._answer_ids(letters)
                if len(e) != len(base):
                    raise _NoJoint
                diffs = [k for k in range(len(e)) if e[k] != base[k]]
                if len(diffs) != 1 or (pos is not None and diffs[0] != pos):
                    raise _NoJoint
                pos = diffs[0]
                ids[li] = e[pos]
            ids[0] = base[pos]
            if len(set(ids)) != len(ids):
                raise _NoJoint
            slots.append({"pos": pos, "label_ids": ids})
        return base, slots

    async def _joint_read(self, qs: list[CompiledQuestion],
                          template: list[int], slots: list[dict],
                          request_id: str) -> list[BackendResult]:
        samples, max_steps = self._settings(qs[0])
        prompt = await self.host.render_joint(qs)

        seed = qs[0].seed
        if seed is None:
            seed = random.SystemRandom().randrange(2**31)
        rng = random.Random(seed)
        vocab = self.host.model_config.get_vocab_size()
        canvases = []
        for _ in range(samples):
            canvas = (list(template) + [self.turn_close]
                      + [self.pad] * (self.canvas_width - len(template) - 1))
            for s in slots:
                canvas[s["pos"]] = rng.randrange(vocab)
            canvases.append(canvas)
        # every slot's letters in one request; each slot reads its own
        label_ids = sorted({t for s in slots for t in s["label_ids"]})
        cap = self.host.token_id_cap
        if cap is not None and len(label_ids) > cap:
            raise BackendError(
                f"joint canvas read needs {len(label_ids)} label ids; the "
                f"engine allows {cap}")
        cached: list[int | None] = []

        async def one_read(k: int) -> list[dict[str, float]]:
            result = await self.host.generate(
                prompt.engine_input,
                self._canvas_params(canvases[k], label_ids, max_steps),
                f"{request_id}-s{k}")
            cached.append(cached_tokens(result))
            positions = result.outputs[0].logprobs or []
            per_q = []
            for q, s in zip(qs, slots):
                at = (positions[s["pos"]] if s["pos"] < len(positions)
                      else None)
                per_read: dict[str, float] = {}
                for o, lid in zip(q.options, s["label_ids"]):
                    lp = at.get(lid) if at else None
                    if lp is None:
                        raise BackendError(
                            f"joint canvas read returned no logprob for "
                            f"option '{o.id}' (label id {lid}) at canvas "
                            f"position {s['pos']}", [o.id])
                    per_read[o.id] = lp.logprob
                per_q.append(restricted_softmax(per_read))
            return per_q

        reads = await asyncio.gather(*(one_read(k) for k in range(samples)))

        # One engine request per sample serves every question: split its
        # prompt tokens (and cached tokens) across them so the request's
        # usage adds up to what the engine read.
        n = len(qs)
        total_in = prompt.input_tokens * samples
        total_cached = add_known(*cached)
        out = []
        for qi, (q, s) in enumerate(zip(qs, slots)):
            meta = {"input_tokens": _share(total_in, n, qi),
                    "cached_input_tokens": (
                        None if total_cached is None
                        else _share(total_cached, n, qi)),
                    "readout": "joint",
                    "canvas_width": self.canvas_width,
                    "max_steps": max_steps,
                    "seed": seed,
                    "joint_read": {"questions": n,
                                   "position": qi + 1,
                                   "input_tokens": prompt.input_tokens,
                                   "canvas_slot": s["pos"]}}
            out.append(_averaged([r[qi] for r in reads],
                                 [o.id for o in q.options], samples, meta))
        return out

    def _canvas_params(self, canvas: list[int], label_ids: list[int],
                       max_steps: int):
        """Sampling params for the structured-decode read-only run. Engine
        knobs go through extra_args (the diffusion engine rejects them as
        SamplingParams fields). The request seed is not sent: the engine
        stores diffusion_seed but consumes nothing from it today."""
        from vllm.sampling_params import SamplingParams
        return SamplingParams(
            max_tokens=self.canvas_width,
            n=1,
            # the engine requires logprobs == len(logprob_token_ids)
            logprobs=len(label_ids),
            logprob_token_ids=label_ids,
            extra_args={
                "diffusion_seed_canvas": canvas,
                "diffusion_canvas_length": self.canvas_width,
                "diffusion_max_steps": max_steps,
                "diffusion_read_only": True,
            },
        )


class _NoJoint(Exception):
    """These questions can't share one canvas; read them one by one."""


def _share(total: int, n: int, i: int) -> int:
    """Question i's share of `total` split over n questions; the shares
    add up to total exactly."""
    return total // n + (1 if i < total % n else 0)


def _averaged(dists: list[dict[str, float]], option_ids: list[str],
              samples: int, meta: dict) -> BackendResult:
    """Mean of per-sample distributions. option_logits is log(mean prob):
    re-softmaxing with T reproduces the calibrated mean."""
    probabilities = {
        oid: sum(d[oid] for d in dists) / len(dists) for oid in option_ids}
    logits_map = {oid: math.log(p) for oid, p in probabilities.items()}

    argmax = max(probabilities, key=probabilities.get)
    samples_meta: dict = {"n": len(dists)}
    if len(dists) > 1:
        samples_meta["stderr"] = {
            oid: math.sqrt(
                sum((d[oid] - probabilities[oid]) ** 2 for d in dists)
                / (len(dists) - 1))
            for oid in option_ids}
        samples_meta["agreement"] = (
            sum(1 for d in dists if max(d, key=d.get) == argmax)
            / len(dists))
    return BackendResult(logits_map, probabilities, forward_passes=samples,
                         meta={**meta, "samples": samples_meta})
