# SPDX-License-Identifier: Apache-2.0
"""Reads for choice questions with more options than markers, on the
logit backend.

- **wide-direct**: one pass. The plain markers plus every two-letter pair
  the tokenizer keeps as ONE token, so the restricted gather reads them
  exactly like `A`-`Z`. Capacity depends on the tokenizer.
- **two-stage**: stage 1 scores each option independently with a
  yes/no read over the shared state prefix; the top SHORTLIST options go
  through a direct marker read; combine mass-proportional. k+1 passes.

The earlier letter-first / digit-first prefixed reads were removed:
wide-direct (single-token letter pairs) reads the same options in one
pass and measured as good or better.
"""
from __future__ import annotations

import asyncio
import hashlib
import math

from vllm.logger import init_logger

from ..protocol import CompiledQuestion
from . import BackendError, BackendResult, add_known, cached_tokens

logger = init_logger(__name__)


def wide_direct_markers(tok, markers: tuple[str, ...]) -> list[str]:
    """Wide-direct marker list: the plain markers followed by every
    two-letter pair that encodes to exactly ONE token (a tokenizer
    merge), in lexicographic order. Each returned marker is a single
    token that decodes back to itself, so the one-pass restricted
    gather reads it exactly like `A`-`Z`."""
    out = list(markers)
    seen_ids = {tok.encode(m, add_special_tokens=False)[0]
                for m in markers
                if len(tok.encode(m, add_special_tokens=False)) == 1}
    for x in markers:
        for y in markers:
            pair = x + y
            enc = tok.encode(pair, add_special_tokens=False)
            if len(enc) != 1:
                continue
            if enc[0] in seen_ids:
                continue  # token id collision with an existing marker
            probe = tok.decode(enc).strip()
            if probe != pair:
                continue  # does not round-trip to the full label
            seen_ids.add(enc[0])
            out.append(pair)
    return out


# Stage-1 option pair. MUST carry yes/no semantics: the rendered payload
# shows only label + description, so bare-letter descriptions gave the
# model no idea which letter meant "fits". Same wording as
# the noul compile (compile.NOUL_YES / NOUL_NO): yes is
# always slot 0 / marker A.
STAGE1_PROMPT_VERSION = "noul-yes-no-v1"
STAGE1_YES_DESCRIPTION = "Yes"
STAGE1_NO_DESCRIPTION = "No"


def _noul_question(request: CompiledQuestion, option) -> dict:
    """The stage-1 scoring question for one option: same state and
    criterion, asked as a yes/no fit."""
    return {
        "state": request.state,
        "question": (
            f"{request.question}\n\nOption under evaluation: "
            f"{option.description}\n\nIs this option the correct "
            f"answer?"),
    }


def _stage1_options(option_cls) -> list:
    """[yes, no] options for the stage-1 read (ids true/false)."""
    return [option_cls(id="true", description=STAGE1_YES_DESCRIPTION),
            option_cls(id="false", description=STAGE1_NO_DESCRIPTION)]


def _tiebreak_ranks(request: CompiledQuestion) -> list[int]:
    """A deterministic, position-independent tie-break rank per option.

    Stage-1 scores are logprob differences of bf16 logits, so they sit on
    a coarse lattice (0.25 steps observed) and ties at the shortlist
    boundary are common. Breaking ties by option index biases the
    shortlist toward early positions. Instead rank by a hash of the
    option id (and the request seed when given): reproducible for the
    same request, independent of where the option sits in the list.
    """
    salt = "" if request.seed is None else str(request.seed)
    def h(opt) -> int:
        d = hashlib.sha256(f"{salt}\x00{opt.id}".encode()).digest()
        return int.from_bytes(d[:8], "big")
    return [h(o) for o in request.options]


def _shortlist_order(scores: list[float], ranks: list[int]) -> list[int]:
    """Option indices sorted by stage-1 score (desc), ties by hash rank."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], ranks[i]))


async def two_stage_read(backend, request: CompiledQuestion,
                         request_id: str, limits,
                         gather: str = "exact") -> BackendResult:
    """Independent scores, shortlist, explicit choice.

    Stage 1: per option, a yes/no read ("Is this option the correct
    answer?", options Yes/No on markers A/B) over the shared state
    prefix. Score = logprob(yes) - logprob(no).
    Stage 2: the top `shortlist` options go through the backend's direct
    marker read (original relative order).
    Combine: finalists share `mass_final` via the stage-2 distribution;
    non-finalists keep their stage-1 softmax share.

    Every engine request gets its own id (`<id>-s1-<i>`, `<id>-s2`):
    stage-1 reads run concurrently and must never share a request id.

    `gather` (`exact` | `top-k`) is how every read gets its markers'
    logprobs, as in the backend's direct read.
    """
    from . import restricted_softmax
    host = backend.host
    markers = list(limits.markers[:2])  # [yes_marker, no_marker]
    YES, NO = markers[0], markers[1]
    tok = host.tokenizer
    yes_id = tok.encode(YES, add_special_tokens=False)[0]
    no_id = tok.encode(NO, add_special_tokens=False)[0]
    if not request.options:
        raise BackendError("two-stage read needs at least one option")
    option_cls = type(request.options[0])

    stage1_tokens: list[int] = []
    stage1_cached: list[int | None] = []

    async def score_one(i: int, option) -> float:
        noul_req = CompiledQuestion(
            model=request.model,
            state=request.state,
            question=_noul_question(request, option)["question"],
            options=_stage1_options(option_cls),
            qtype="noul",
            seed=request.seed,
        )
        prompt = await host.render(noul_req)
        engine_input, slot_ids = prompt.engine_input, prompt.slot_ids
        stage1_tokens.append(prompt.input_tokens)
        # slot order follows options: [yes marker, no marker]
        if list(slot_ids[:2]) != [yes_id, no_id]:
            raise BackendError(
                f"two-stage stage-1 slot ids {list(slot_ids[:2])} do not "
                f"match the yes/no marker ids {[yes_id, no_id]}")
        lp, cached = await _gather(host, engine_input, [yes_id, no_id],
                                   f"{request_id}-s1-{i}", gather)
        stage1_cached.append(cached)
        return lp[yes_id] - lp[no_id]

    scores = await asyncio.gather(
        *(score_one(i, o) for i, o in enumerate(request.options)))
    order = _shortlist_order(list(scores), _tiebreak_ranks(request))
    # stage 2 is a direct marker read, so with an exact gather the
    # finalists must fit the engine's restricted-read limit (never more
    # ids than it accepts); a top-k read has no per-label limit
    shortlist_cap = limits.shortlist
    limit = getattr(host, "read_limit", None)
    if gather == "exact" and limit is not None:
        shortlist_cap = min(shortlist_cap, limit)
    shortlist = order[: shortlist_cap]

    # stage 2: direct read over the finalists, original relative order
    sub_opts = [request.options[i] for i in sorted(shortlist)]
    sub_req = CompiledQuestion(
        model=request.model,
        state=request.state,
        question=request.question,
        options=sub_opts,
        qtype=request.qtype,
        seed=request.seed,
    )
    prompt = await host.render(sub_req)
    engine_input, input_tokens = prompt.engine_input, prompt.input_tokens
    slot_ids = prompt.slot_ids
    lp, cached2 = await _gather(host, engine_input, slot_ids,
                                f"{request_id}-s2", gather)
    # slot_ids follow sub_opts order: map token id -> option id
    p2 = restricted_softmax(lp, temperature=1.0)
    p2_by_id = {opt.id: p2[slot_ids[j]]
                for j, opt in enumerate(sub_opts)}

    s1 = restricted_softmax({i: scores[i] for i in range(len(scores))},
                            temperature=1.0)
    mass_final = sum(s1[i] for i in shortlist)
    probs: dict[str, float] = {}
    for j, opt in enumerate(sub_opts):
        probs[opt.id] = mass_final * p2_by_id[opt.id]
    for i, opt in enumerate(request.options):
        if i not in shortlist:
            probs[opt.id] = s1[i]
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    # a probability that underflowed to 0 has no log; -inf is its limit
    logits = {k: (math.log(v) if v > 0 else -math.inf)
              for k, v in probs.items()}
    return BackendResult(
        logits, probs,
        forward_passes=len(request.options) + 1,
        meta={
            # every read's prompt counts: k stage-1 reads + 1 stage-2 read
            "input_tokens": sum(stage1_tokens) + input_tokens,
            "cached_input_tokens": add_known(*stage1_cached, cached2),
            # a sum over k+1 differently-shaped reads is not comparable to
            # a direct read's mass, so it is not reported
            "option_mass": None,
            "readout": "two-stage",
            "label_layout": "direct",
            "stage1_prompt": STAGE1_PROMPT_VERSION,
            "stage1_scores": {request.options[i].id: scores[i]
                              for i in range(len(scores))},
            "shortlist": [request.options[i].id for i in shortlist],
        })


async def _gather(host, engine_input, token_ids: list[int],
                  request_id: str, gather: str = "exact"
                  ) -> tuple[dict[int, float], int | None]:
    """Logprobs for exactly `token_ids`. `exact`: restricted gather,
    every id must come back. `top-k`: the top-k window, an id outside it
    scores the window's lowest logprob (at least one id must be inside).
    Returns (logprobs by token id, prompt tokens served from the
    cache)."""
    if gather == "top-k":
        found, floor, result = await host.topk_read(
            engine_input, token_ids, request_id)
        if not found:
            raise BackendError(
                f"none of the token ids {list(token_ids)} is in the "
                f"top-k window")
        return ({t: found.get(t, floor) for t in token_ids},
                cached_tokens(result))
    found, result, _attempts = await host.restricted_read(
        engine_input, token_ids, request_id)
    missing = [t for t in token_ids if t not in found]
    if missing:
        raise BackendError(
            f"restricted gather missing token ids: {missing}")
    return found, cached_tokens(result)
