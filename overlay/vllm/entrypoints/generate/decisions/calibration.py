# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup calibration: the server fits its own
confidence temperature at startup, from a bundled or operator-provided
question set, on the same internal path a request takes.

The fit mirrors semif/benchmarks/calibrate.py: minimize the mean negative
log-likelihood of softmax(scores / T) at the right option (golden-section
search over T in [0.05, 20]); group-disjoint 5-fold cross-validated
calibrated ECE against raw ECE; 95% intervals by group bootstrap. T is
kept only when the calibrated interval lies entirely below the raw one.

Every step that fails falls back to T = 1.0 (source "default") with one
WARNING naming the cause; the server starts anyway."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from datetime import date, datetime

import numpy as np

from vllm.logger import init_logger
from vllm.entrypoints.generate.decisions.limits import (
    CALIBRATION_DIR_ENV,
    calibration_mode,
    default_calibration_dir,
    get_limits,
    operator_temperature_set,
)
from vllm.entrypoints.generate.decisions.protocol import (
    CompiledQuestion, DecisionOption, DecisionsQuery)

logger = init_logger(__name__)
def vllm_version() -> str:
    import vllm
    return getattr(vllm, "__version__", "unknown")

# Fit bounds; landing at either end means broken scores, not a real T.
T_LOW, T_HIGH = 0.05, 20.0
MIN_QUESTIONS = 50


# ---------------------------------------------------------------------
# question file
# ---------------------------------------------------------------------

def question_set_path(limits=None) -> tuple[str, str]:
    """(mode, path) for the configured calibration question set."""
    mode = calibration_mode(limits)
    if mode == "file":
        return mode, (limits or get_limits()).calibration
    bundled = os.path.join(os.path.dirname(__file__),
                           "calibration_data", "jevbench-public.jsonl")
    return "jevbench", bundled


def load_questions(path: str) -> list[dict]:
    """Read the question file. Raises on unreadable/malformed input."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: malformed JSON: {e}")
            for field in ("state", "question", "expected"):
                if field not in r:
                    raise ValueError(
                        f"{path}:{lineno}: missing field {field!r}")
            if not isinstance(r["question"], dict) or \
                    "type" not in r["question"] or \
                    "instructions" not in r["question"]:
                raise ValueError(
                    f"{path}:{lineno}: 'question' must be an object with "
                    "'type' and 'instructions'")
            rows.append(r)
    if not rows:
        raise ValueError(f"{path}: no questions")
    return rows


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
# running the questions through the internal path
# ---------------------------------------------------------------------

def _wire_question(row: dict) -> dict:
    """The wire's single typed question, exactly what
    tools/capture_logits.py sends: the canonical record's question block
    verbatim (type, instructions, optional criteria)."""
    q = {"type": row["question"]["type"],
         "instructions": row["question"]["instructions"]}
    if row["question"].get("criteria") is not None:
        q["criteria"] = row["question"]["criteria"]
    return q


def _gold_index(row: dict, option_ids: list[str]) -> int | None:
    """Index of the gold answer among the option ids; None when absent.
    Score tasks store the expected level as an int; the server's option
    ids are strings. Noul yes/no -> true/false, like the capture tool."""
    exp = str(row.get("expected"))
    if row["question"]["type"] == "noul":
        exp = {"yes": "true", "no": "false"}.get(exp, exp)
    return option_ids.index(exp) if exp in option_ids else None


async def run_questions(serving, rows: list[dict], backend_name: str | None,
                        concurrency: int = 8) -> tuple[list[dict], int]:
    """Run each row through the internal path (compile, then the backend
    read - no HTTP) and collect {logits, true_index, group, id}. Returns
    (usable pairs, skipped count). Questions that fail or whose gold
    answer is not among the options are skipped and counted."""
    sem = asyncio.Semaphore(concurrency)
    pairs: list[dict] = []
    skipped = 0

    async def one(index: int, row: dict):
        nonlocal skipped
        async with sem:
            qid = f"calib-{row.get('id', index)}"
            body = {"state": row["state"], "model": "calibration",
                    "questions": {"decision": _wire_question(row)}}
            try:
                request = DecisionsQuery(**body)
                compiled = _compile_one(request)
                backend = serving.decision_backend
                result = await backend.read(
                    compiled, f"decision-calibration-{index}")
            except Exception as e:  # noqa: BLE001 - skip, count
                logger.debug("calibration: skipped %s: %s", qid, e)
                skipped += 1
                return
            ids = list(result.option_logits.keys())
            gold = _gold_index(row, ids)
            if gold is None:
                skipped += 1
                return
            pairs.append({
                "id": str(row.get("id", index)),
                "logits": [result.option_logits[i] for i in ids],
                "true_index": gold,
                "group": row.get("group") or row.get("group_id")
                or row.get("id", str(index)),
                "family": row.get("family", ""),
            })

    await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    return pairs, skipped


def _compile_one(request: DecisionsQuery) -> CompiledQuestion:
    from vllm.entrypoints.generate.decisions.compile import compile_question
    return compile_question(request, "decision")


# ---------------------------------------------------------------------
# the fit (numpy only; mirrors semif/benchmarks/calibrate.py)
# ---------------------------------------------------------------------

def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted = logits - logits.max()
    weights = np.exp(shifted / temperature)
    return weights / weights.sum()


def _mean_nll(pairs: list[dict], temperature: float) -> float:
    total = 0.0
    for pair in pairs:
        probs = _softmax(np.asarray(pair["logits"]), temperature)
        total += -math.log(max(probs[pair["true_index"]], 1e-12))
    return total / len(pairs)


def fit_temperature(pairs: list[dict], bounds=(T_LOW, T_HIGH),
                    iterations: int = 60) -> float:
    """Minimize mean NLL over T by golden-section search (NLL is convex
    in 1/T)."""
    ratio = (math.sqrt(5) - 1) / 2
    low, high = bounds
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    f_left, f_right = _mean_nll(pairs, left), _mean_nll(pairs, right)
    for _ in range(iterations):
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - ratio * (high - low)
            f_left = _mean_nll(pairs, left)
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (high - low)
            f_right = _mean_nll(pairs, right)
    return (low + high) / 2


def scored(pairs: list[dict], temperature: float) -> list[dict]:
    """Per-row top-label confidence and correctness at a temperature
    (correctness is T-invariant)."""
    out = []
    for pair in pairs:
        probs = _softmax(np.asarray(pair["logits"]), temperature)
        out.append({"id": pair["id"], "group": pair["group"],
                    "confidence": float(probs.max()),
                    "correct": int(probs.argmax() == pair["true_index"])})
    return out


def ece(rows: list[dict], bins: int = 10) -> float:
    """Expected calibration error: 10 equal-width bins on the top
    option's probability; weighted mean of |accuracy - confidence|."""
    total = 0.0
    for index in range(bins):
        part = [r for r in rows
                if min(bins - 1, int(r["confidence"] * bins)) == index]
        if part:
            conf = sum(r["confidence"] for r in part) / len(part)
            acc = sum(r["correct"] for r in part) / len(part)
            total += abs(acc - conf) * len(part) / len(rows)
    return total


def bootstrap_ece(rows: list[dict], samples: int = 1000, seed: int = 217,
                  bins: int = 10) -> list[float]:
    """95% interval by resampling source groups with replacement."""
    by_group: dict[str, list] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    groups = list(by_group.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        draw = []
        for _ in groups:
            draw.extend(groups[int(rng.integers(len(groups)))])
        values.append(ece(draw, bins))
    values.sort()
    return [values[int(0.025 * samples)],
            values[min(samples - 1, int(0.975 * samples))]]


def fold_map(pairs: list[dict], folds: int = 5, seed: int = 217) -> dict:
    """Deterministic {group: fold} assignment, hash of the question id
    (group-disjoint; variants share a group)."""
    groups = sorted({p["group"] for p in pairs})
    return {g: int(int(hashlib.sha256(g.encode()).hexdigest()[:8], 16)
                   % folds) for g in groups}


def grouped_cv(pairs: list[dict], folds: int = 5) -> list[dict]:
    """Out-of-fold calibrated rows: fit T on the other folds, calibrate
    this one, pool."""
    fold_of = fold_map(pairs, folds)
    held: list[dict] = []
    for fold in range(folds):
        train = [p for p in pairs if fold_of[p["group"]] != fold]
        test = [p for p in pairs if fold_of[p["group"]] == fold]
        if not train or not test:
            continue
        held.extend(scored(test, fit_temperature(train)))
    return held


def fit_all(pairs: list[dict]) -> dict:
    """The full fit: T on all rows, raw and out-of-fold calibrated ECE
    with 95% intervals, and whether the gain is real (kept)."""
    t = fit_temperature(pairs)
    raw = scored(pairs, 1.0)
    held = grouped_cv(pairs)
    raw_ci = bootstrap_ece(raw)
    cal_ci = bootstrap_ece(held)
    kept = cal_ci[1] < raw_ci[0]
    return {"t": t, "kept": kept,
            "ece_raw": ece(raw), "ece_raw_ci": raw_ci,
            "ece_cal": ece(held), "ece_cal_ci": cal_ci}


def at_range_edge(t: float) -> bool:
    """Within 1% of either end of the allowed range: broken scores."""
    span = T_HIGH - T_LOW
    return abs(t - T_LOW) <= 0.01 * span or abs(t - T_HIGH) <= 0.01 * span


# ---------------------------------------------------------------------
# saved result: key, file name, load/save
# ---------------------------------------------------------------------

def calibration_key(serving, question_file: str,
                    backend_kwargs: dict | None = None) -> dict:
    """Everything a saved T is valid for. Any change -> recalibrate."""
    mc = serving.engine_client.model_config
    backend = serving.decision_backend
    return {
        "served_model": serving.models.model_name(None),
        "model_path": getattr(mc, "model", None),
        "model_revision": getattr(mc, "revision", None),
        "quantization": getattr(mc, "quantization", None),
        "backend": backend.name,
        # the backend's startup kwargs (VLLM_TYPED_DECISIONS_BACKEND's
        # spec): they shape the readouts the T was fitted on
        "backend_kwargs": backend_kwargs or {},
        "render_version": _render_version(),
        "question_file_sha256": file_sha256(question_file),
    }


def _render_version() -> str:
    from vllm.entrypoints.generate.decisions.compile import RENDER_VERSION
    return RENDER_VERSION


def key_id(key: dict) -> str:
    blob = json.dumps(key, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def save_result(directory: str, key: dict, result: dict) -> str:
    """Write the result JSON; returns the path. Caller logs on failure."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, key_id(key) + ".json")
    payload = {**result, "key": key}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path


def load_result(directory: str, key: dict) -> dict | None:
    """A saved result with a matching key, or None."""
    path = os.path.join(directory, key_id(key) + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return saved if saved.get("key") == key else None


def results_dir() -> str:
    limits = get_limits()
    return (limits.calibration_dir
            or os.environ.get(CALIBRATION_DIR_ENV)
            or default_calibration_dir())


# ---------------------------------------------------------------------
# the startup entry point
# ---------------------------------------------------------------------

def should_calibrate(serving, limits=None) -> tuple[bool, str]:
    """(should, reason): whether this server calibrates at startup, per
    the B2 table. Never when the operator set a temperature."""
    if operator_temperature_set():
        return False, "skipped (VLLM_TYPED_DECISIONS_TEMPERATURE is set)"
    limits = limits or get_limits()
    mode = calibration_mode(limits)
    if mode == "off":
        return False, "off"
    backend = serving.decision_backend
    wants = getattr(backend, "calibrate_by_default", False)
    if mode == "jevbench":
        return (True, "jevbench") if wants else (False, "off")
    return True, ("jevbench" if mode == "on" else limits.calibration)


async def run_startup_calibration(serving) -> None:
    """The full startup calibration: skip / load / run / fit / save, with
    one log line at the end. Any failure -> T=1.0 source 'default' with
    a WARNING naming the cause. Stores the outcome on the startup
    object; never raises."""
    startup = serving.startup
    limits = get_limits()
    should, reason = should_calibrate(serving, limits)
    if not should:
        startup.calibration = {"t": 1.0, "kept": False, "source": "default"}
        logger.info("decision calibration: %s",
                    "off" if reason == "off" else reason)
        return

    started = time.monotonic()
    try:
        mode, qfile = question_set_path(limits)
        key = calibration_key(
            serving, qfile,
            getattr(serving, "default_backend_kwargs", None) or {})
        directory = results_dir()
        saved = load_result(directory, key)
        if saved is not None:
            kept = bool(saved["kept"])
            # a "no correction needed" result keeps T=1.0: measured, not
            # never-measured
            t = float(saved["t"]) if kept else 1.0
            startup.calibration = {"t": t, "kept": kept,
                                   "source": "calibrated"}
            logger.info("decision calibration: loaded T=%.2f %s from %s",
                        t, "kept" if kept else "not kept",
                        path_name(directory, key))
            return
        rows = load_questions(qfile)
        pairs, skipped = await run_questions(
            serving, rows, serving.decision_backend.name)
        if len(pairs) < MIN_QUESTIONS:
            startup.calibration = {"t": 1.0, "kept": False,
                                   "source": "default"}
            logger.warning(
                "decision calibration: FAILED (only %d of %d questions "
                "usable; need %d); using T=1.0", len(pairs), len(rows),
                MIN_QUESTIONS)
            return
        fit = fit_all(pairs)
        if at_range_edge(fit["t"]):
            startup.calibration = {"t": 1.0, "kept": False,
                                   "source": "default"}
            logger.warning(
                "decision calibration: FAILED (fitted T=%.3f is at the "
                "edge of the allowed range %.2f..%.2f - broken scores, "
                "not a real T); using T=1.0", fit["t"], T_LOW, T_HIGH)
            return
        kept = bool(fit["kept"])
        t = fit["t"] if kept else 1.0
        startup.calibration = {"t": t, "kept": kept,
                               "source": "calibrated"}
        result = {
            "t": fit["t"], "kept": kept,
            "ece_raw": {"value": fit["ece_raw"], "ci95": fit["ece_raw_ci"]},
            "ece_cal": {"value": fit["ece_cal"], "ci95": fit["ece_cal_ci"]},
            "questions_used": len(pairs), "questions_skipped": skipped,
            "question_set": reason if mode != "file" else qfile,
            "date": date.today().isoformat(),
            "vllm_version": vllm_version(),
        }
        try:
            path = save_result(directory, key, result)
            logger.info("decision calibration: T=%.2f %s (ECE %.3f -> "
                        "%.3f, %d/%d questions, %s, %d s) -> %s",
                        result["t"], "kept" if kept else "not kept",
                        fit["ece_raw"], fit["ece_cal"], len(pairs),
                        len(pairs) + skipped, result["question_set"],
                        int(time.monotonic() - started), path)
        except OSError as e:
            logger.warning(
                "decision calibration: could not save the result (%s); "
                "carrying it in memory", e)
            logger.info("decision calibration: T=%.2f %s (ECE %.3f -> "
                        "%.3f, %d/%d questions, %s, %d s)",
                        result["t"], "kept" if kept else "not kept",
                        fit["ece_raw"], fit["ece_cal"], len(pairs),
                        len(pairs) + skipped, result["question_set"],
                        int(time.monotonic() - started))
    except Exception as e:  # noqa: BLE001 - start anyway at 1.0
        startup.calibration = {"t": 1.0, "kept": False, "source": "default"}
        logger.warning(
            "decision calibration: FAILED (%s); using T=1.0", e)


def path_name(directory: str, key: dict) -> str:
    return os.path.join(directory, key_id(key) + ".json")


def schedule_calibration(serving) -> None:
    """Chain the calibration after the self-check in the background
    task. Idempotent: both serving instances call this; one chain runs
    per server. Sets work_pending itself, because a non-logit server
    never calls Startup.schedule() - the 503 gate must still hold while
    the calibration runs."""
    startup = serving.startup
    if getattr(startup, "_calibration_scheduled", False):
        return
    startup._calibration_scheduled = True
    prev = startup._task
    startup.work_pending = True

    async def chained():
        try:
            if prev is not None:
                await prev
            # skip only on a REAL refusal; slot_check None means this
            # backend has no self-check - calibrate anyway
            if startup.slot_check is not None and \
                    startup.slot_check != "ok":
                return
            await run_startup_calibration(serving)
        finally:
            # every path clears the gate
            startup.work_pending = False

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        startup.work_pending = False
        return
    startup._task = loop.create_task(startup._guard(chained()))
