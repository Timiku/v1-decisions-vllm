# SPDX-License-Identifier: Apache-2.0
"""One answer shape for every question type and every backend.

Core fields, the same meaning everywhere:
  - `type`: the question type;
  - `probabilities`: calibrated distribution over the option ids, sums to 1;
  - the type's own answer fields (noul: `noul`; choice: `choice`;
    score: `score` + `legend`; a plugin type: whatever it defines);
  - `confidence`: normalized peak (k*max-1)/(k-1) over `probabilities`;
  - `extra`: `backend` (what the backend did) and `audit` (the facts to
    check how the answer was produced), each trimmable per request.
"""
from __future__ import annotations

from typing import Any

CONFIDENCE_FORMULA = "normalized-peak-v1"


def normalized_peak_confidence(probabilities: dict[str, float]) -> float:
    """(k*max-1)/(k-1), clamped to [0, 1]: 1.0 when all mass is on one
    option, 0.0 when uniform."""
    k = len(probabilities)
    if k < 2:
        return 1.0
    return max(0.0, min(1.0, (k * max(probabilities.values()) - 1) / (k - 1)))


def build_answer(question_type, options: list,
                 probabilities: dict[str, float],
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The answer dict for one question: core fields, the type's own
    fields, then `extra`."""
    if abs(sum(probabilities.values()) - 1.0) > 1e-6:
        raise ValueError("probabilities must sum to 1")
    own = question_type.answer(probabilities, options)
    clash = set(own) & {"type", "probabilities", "confidence", "extra"}
    if clash:
        raise ValueError(f"question type {question_type.name!r} answer "
                         f"fields clash with core fields: {sorted(clash)}")
    answer = {"type": question_type.name, "probabilities": probabilities}
    answer.update(own)
    answer["confidence"] = normalized_peak_confidence(probabilities)
    if extra is not None:
        answer["extra"] = extra
    return answer
