# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One startup object per server process, shared by the decisions and
system-one serving instances (the wire patch builds two).

The first `ServingDecisions.__init__` schedules the startup work (the
answer-slot self-check today; startup calibration in a later stage) as a
background task on the running event loop — the wire patch builds both
instances inside vLLM's async app setup, so a loop is running. Without a
running loop (offline tests) the work starts at the first request
instead.

While the work runs, every decision request gets a 503 "server is
starting"; if the work crashes, its result is a failed self-check and
every logit request gets the 503 refusal. Never hangs."""
from __future__ import annotations

import asyncio
import traceback

from vllm.logger import init_logger
from vllm.entrypoints.generate.decisions.answer_slot import (
    EMPTY_REASONING_BLOCKS,
    knows_suffix,
)
from vllm.entrypoints.generate.decisions.limits import MIN_OPTION_MASS_ENV

logger = init_logger(__name__)

STARTING_MESSAGE = ("decision server is starting (self-check / "
                    "calibration in progress); retry shortly")


class Startup:
    """Shared startup state: the self-check result and what it found
    (`answer_suffix`, `thinking_switch`), later the calibration result."""

    def __init__(self):
        self.answer_suffix: tuple[str, ...] | None = None
        self.thinking_switch = True
        # None = still running (or not started); "ok" or the refusal text
        self.slot_check: str | None = None
        self.failed: BaseException | None = None
        self._task: asyncio.Task | None = None
        self._lock: asyncio.Lock | None = None
        # True from the first schedule() until the whole chained work
        # (self-check + calibration) is done - the 503 gate reads this,
        # not slot_check, so requests wait through calibration too.
        self.work_pending = False

    # -- scheduling ------------------------------------------------------
    def schedule(self, run) -> None:
        """Start the startup work in the background. `run` is a no-arg
        callable returning a coroutine, bound to one serving instance
        (the probe runs through its backend). Without a running loop
        (offline tests) nothing happens here: the first request starts
        the work. Idempotent: the second serving instance is a no-op."""
        if self._task is not None:
            return
        if not callable(run):
            raise TypeError(
                f"schedule() takes a callable returning a coroutine, "
                f"got {run!r} (pass a function, not a coroutine object)")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self.work_pending = True
        self._task = loop.create_task(self._guard(run()))

    async def _guard(self, coro) -> None:
        try:
            await coro
        except BaseException as e:  # noqa: BLE001 - never hang a request
            logger.error("decision startup work crashed:\n%s",
                         traceback.format_exc())
            self.failed = e
            # Only the SELF-CHECK crash clobbers slot_check (a real
            # refusal). A later-stage crash (calibration) must not
            # overwrite a finished "ok": calibration falls back to 1.0
            # on its own.
            if self.slot_check is None:
                self.slot_check = (
                    "decision self-check failed: startup work crashed "
                    f"({e!r}). Readouts would be unreliable; set "
                    f"{MIN_OPTION_MASS_ENV}=0 to serve anyway.")

    @property
    def starting(self) -> bool:
        """True while the SCHEDULED startup work (self-check AND
        calibration) runs. Never scheduled (no loop at construction):
        the first request runs the work inline, so there is nothing to
        wait for."""
        if self._task is None:
            return False
        if self.failed is not None:
            return False
        if not self.work_pending:
            return False
        return not self._task.done()

    # -- the self-check itself --------------------------------------------
    async def run_self_check(self, serving) -> str:
        """The answer-slot self-check, unchanged from the per-instance
        version it replaces: same probe, same empty-reasoning-block
        retry, same messages. Returns "ok" or the refusal text."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self.slot_check is not None:
                return self.slot_check
            from .limits import MIN_OPTION_MASS_ENV, get_limits
            need = get_limits().min_option_mass
            if need <= 0:
                self.slot_check = "ok"
                return "ok"
            tokenizer = serving.base_renderer.get_tokenizer()
            tried: list[str] = []
            for suffix in (None, *(b for b in EMPTY_REASONING_BLOCKS
                                   if knows_suffix(tokenizer, b))):
                self.answer_suffix = suffix
                mass = await serving._probe_mass()
                if mass is None:
                    logger.warning(
                        "decision answer-slot self-check skipped: the "
                        "engine doesn't report full-vocabulary "
                        "log-probabilities (logprobs_mode)")
                    self.slot_check = "ok"
                    return "ok"
                label = "as rendered" if suffix is None else (
                    "with an empty reasoning block "
                    + repr("".join(suffix)))
                tried.append(f"{label}: {mass:.1%}")
                if mass >= need:
                    logger.info("decision answer-slot self-check passed "
                                "(%s)", tried[-1])
                    self.slot_check = "ok"
                    return "ok"
            self.answer_suffix = None
            self.slot_check = (
                "decision self-check failed: on an easy probe question the "
                "model put too little probability on the option letters "
                f"({'; '.join(tried)}; need {need:.0%}). The chat template "
                "probably leaves the model at the start of a reasoning "
                "block or a preamble instead of at its answer, so readouts "
                "would be meaningless. Set a chat template that ends at "
                "the answer, or set "
                f"{MIN_OPTION_MASS_ENV}=0 to serve anyway.")
            logger.error(self.slot_check)
            return self.slot_check


# One per server process: the wire patch builds two serving instances,
# both attach to this object.
_STARTUP: Startup | None = None


def get_startup() -> Startup:
    global _STARTUP
    if _STARTUP is None:
        _STARTUP = Startup()
    return _STARTUP
