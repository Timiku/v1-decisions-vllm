# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the answer goes: the end of the decision prompt and the token
each option label is read at.

The logit read needs the rendered prompt to stop exactly where the model's
next token is its answer. Chat templates differ in how they end:

- most (Qwen3, Llama 3/4, Gemma 3/4, DeepSeek V3.2/V4, Mistral) honour the
  thinking-off switch and end at the answer;
- some leave a reasoning block open whatever the switch says (GLM-5.x ends
  on `<think>`), or stop before the model picks its output channel
  (GPT-OSS ends on `<|start|>assistant`). `closing_suffix` returns the
  pieces that close such an ending with an empty block;
- some models open a reasoning block by themselves (DeepSeek-R1). No
  template sign shows it, so the serving self-check finds it from the
  model's own probabilities and then adds one of
  `EMPTY_REASONING_BLOCKS`.

`answer_slots` finds the token id of each label as it would follow the
prompt in context, so tokenizers that spell a word-initial letter
differently from a mid-text one are read at the right token.
"""
from __future__ import annotations

# A suffix is a sequence of pieces: control tokens (looked up as single
# tokens) and plain text (encoded). Some tokenizers don't parse control
# tokens out of plain text, so pieces are never joined before encoding.
Suffix = tuple[str, ...]

# (how a template-rendered prompt can end, the pieces that close it).
# Matched on the decoded prompt tail, ignoring trailing whitespace.
OPEN_ENDINGS: tuple[tuple[str, Suffix], ...] = (
    ("<think>", ("</think>",)),                                  # GLM-5.x
    ("<|start|>assistant",
     ("<|channel|>", "final", "<|message|>")),                   # GPT-OSS
)

# Empty reasoning blocks, for models that open one by themselves. The
# self-check tries each one whose control tokens the tokenizer has.
EMPTY_REASONING_BLOCKS: tuple[Suffix, ...] = (
    ("<think>", "\n\n", "</think>", "\n\n"),    # DeepSeek-R1 style
    ("[THINK]", "[/THINK]"),                    # Mistral reasoning
)


class SlotError(ValueError):
    """The prompt's end or a label can't be read reliably."""


def closing_suffix(tail_text: str) -> Suffix | None:
    """Pieces that close an open reasoning or channel ending, or None
    when the prompt already ends at the answer."""
    end = tail_text.rstrip()
    for opening, closing in OPEN_ENDINGS:
        if end.endswith(opening):
            return closing
    return None


def _control_id(tok, piece: str) -> int | None:
    """The id of `piece` when the tokenizer has it as one token."""
    convert = getattr(tok, "convert_tokens_to_ids", None)
    if convert is not None:
        try:
            tid = convert(piece)
        except Exception:
            tid = None
        unk = getattr(tok, "unk_token_id", None)
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            return tid
    enc = tok.encode(piece, add_special_tokens=False)
    return enc[0] if len(enc) == 1 else None


def _is_control(piece: str) -> bool:
    """`<…>` or `[…]` with no whitespace: a chat-template control token."""
    return (len(piece) > 2 and piece.strip() == piece
            and piece[0] in "<[" and piece[-1] in ">]")


def knows_suffix(tok, suffix: Suffix) -> bool:
    """Every control token of `suffix` is one token in this tokenizer."""
    return all(_control_id(tok, p) is not None
               for p in suffix if _is_control(p))


def encode_suffix(tok, suffix: Suffix) -> list[int]:
    """Token ids of `suffix`, piece by piece."""
    ids: list[int] = []
    for piece in suffix:
        tid = _control_id(tok, piece) if _is_control(piece) else None
        ids.extend([tid] if tid is not None
                   else tok.encode(piece, add_special_tokens=False))
    return ids


def _special_ids(tok) -> set[int]:
    ids = set(getattr(tok, "all_special_ids", None) or ())
    added = getattr(tok, "added_tokens_decoder", None)
    if isinstance(added, dict):
        ids.update(k for k in added if isinstance(k, int))
    return ids


def _text_tail(tok, prompt_ids: list[int], window: int = 64
               ) -> list[int]:
    """The prompt's last run of plain-text tokens: everything after the
    last special or added token (the chat template's control tokens),
    within `window` tokens. When the window holds no such token, the
    shortest ending that survives a decode/encode round trip."""
    special = _special_ids(tok)
    recent = prompt_ids[-window:]
    for i in range(len(recent) - 1, -1, -1):
        if recent[i] in special:
            return list(recent[i + 1:])
    while True:
        tail = list(prompt_ids[-window:])
        if tok.encode(tok.decode(tail), add_special_tokens=False) == tail:
            return tail
        if window >= len(prompt_ids):
            return list(prompt_ids)
        window *= 2


def answer_slots(tok, prompt_ids: list[int], labels: list[str]
                 ) -> list[int]:
    """Token id of each label as the model's next token after the prompt.

    Each label must be exactly one token in context (appended to the
    prompt's plain-text ending), leave that ending's tokens unchanged, and
    decode back to itself. Raises SlotError otherwise."""
    tail = _text_tail(tok, prompt_ids)
    tail_text = tok.decode(tail) if tail else ""
    ids: list[int] = []
    for label in labels:
        whole = list(tok.encode(tail_text + label, add_special_tokens=False))
        if whole[:len(tail)] != tail:
            raise SlotError(
                f"Answer boundary changes tokenization for slot {label!r}. "
                "The prompt tail is not letter-stable.")
        new = whole[len(tail):]
        if len(new) != 1:
            raise SlotError(
                f"Option label {label!r} is not a single token after the "
                f"prompt (tokenizes to {len(new)}). Cannot read out "
                "reliably.")
        probe = tok.decode(new).strip()
        if probe != label:
            raise SlotError(
                f"Option label {label!r} does not round-trip through the "
                f"tokenizer (its token decodes to {probe!r}). Cannot read "
                "out reliably.")
        ids.append(new[0])
    return ids
