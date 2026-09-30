# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile one typed question into a CompiledQuestion. Pure: no I/O.

The question's type (from `question_types.py`) supplies the option set;
this module adds what every type shares: the state, the flattened
instructions and the request-level settings. Every question in a request
keeps a byte-identical state render, so the engine's prefix cache
absorbs the shared state prefill.

Prompt fidelity (pinned by tests/golden/): the payload carries marker +
description only, never option ids; structured values are JSON-serialized
verbatim.

`RENDER_VERSION` identifies the prompt-building code: bump it whenever a
change here (or in serving's render path) alters any render. A saved
startup calibration is only valid for one render version.
"""
from __future__ import annotations

from typing import Any

from .protocol import CompiledQuestion, DecisionsRequest
from .question_types import flatten, get_question_type

RENDER_VERSION = "2026-09-29.1"


def compile_question(request: DecisionsRequest, qid: str,
                     backend_options: Any = None) -> CompiledQuestion:
    """`backend_options`: the request's options, already validated by the
    backend that will read the question (serving does that once per
    request)."""
    q = request.questions[qid]
    return CompiledQuestion(
        state=request.state,
        question=flatten(q.instructions),
        options=get_question_type(q.type).options(q),
        qtype=q.type,
        model=request.model,
        seed=request.seed,
        backend=request.backend,
        backend_options=backend_options,
        request_id=f"decision-{qid}",
    )
