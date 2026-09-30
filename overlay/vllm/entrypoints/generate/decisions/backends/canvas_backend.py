# SPDX-License-Identifier: Apache-2.0
"""CanvasBackend: read-only canvas denoise for diffusion-style checkpoints.

Reads the decision from one canvas slot: the canvas is seeded with a
noise token at the answer slot followed by the turn-close marker, one
read-only denoise runs, and the logprobs of each option letter at that
slot are the option scores. With `samples=n` the read repeats with n
noise draws from one seed and the distributions are averaged.

Ported from the merged precedent's (#57250) read loop
(`examples/.../structured_server.py`), one slot per question.

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

    async def read(self, question: CompiledQuestion,
                   request_id: str) -> BackendResult:
        opts = question.backend_options
        if isinstance(opts, dict):
            opts = CanvasOptions.model_validate(opts)
        samples = (opts and opts.samples) or self.samples
        max_steps = (opts and opts.max_steps) or self.max_steps

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

        # Mean of per-read distributions. option_logits is log(mean prob):
        # re-softmaxing with T reproduces the calibrated mean.
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
        meta = {"input_tokens": prompt.input_tokens * samples,
                "cached_input_tokens": add_known(*cached),
                "canvas_width": self.canvas_width,
                "max_steps": max_steps,
                "seed": seed,
                "samples": samples_meta}
        return BackendResult(logits_map, probabilities,
                             forward_passes=samples, meta=meta)

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
