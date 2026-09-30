#!/usr/bin/env python3
"""Does the decision prompt end at the answer on this model? A check of
tokenizers and chat templates, no GPU, with a report a non-programmer can
read.

For each model it builds the real decision prompt the way the server does
(same system prompt, payload and thinking-off switches), then reports:

- how the prompt ends, and whether answer_slot.py had to close an open
  reasoning block or output channel;
- whether all 26 option letters can be read (one token each, at the right
  token in context);
- the wide-direct capacity (options readable in one pass);
- which empty reasoning blocks the server's self-check could fall back
  to, for models that open one by themselves.

It can't see what the model wants to write next; the server's
answer-slot self-check measures that on the loaded model.

    python tools/check_tokenizers.py --report REPORT.md
    python tools/check_tokenizers.py --models Qwen/Qwen3-8B openai/gpt-oss-20b

Downloads only tokenizer and chat-template files (a few MB per model; no
weights) into the Hugging Face cache. Needs `transformers`; Mistral models
that ship only `tekken.json` also need `mistral_common`. DeepSeek V3.2 and
V4 render with the encoding script shipped in their repos, the same code
vLLM carries as its own tokenizer for them.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Six families by use in 2025-2026, plus GLM. Ungated mirrors stand in for
# gated repos (same tokenizer files).
DEFAULT_MODELS = [
    ("Qwen", "Qwen/Qwen3-8B"),
    ("Qwen", "Qwen/Qwen3.5-9B"),
    ("Qwen", "Qwen/Qwen3.6-35B-A3B-FP8"),
    ("Llama", "unsloth/Llama-3.3-70B-Instruct"),
    ("Llama", "unsloth/Llama-4-Scout-17B-16E-Instruct"),
    ("Gemma", "unsloth/gemma-3-4b-it"),
    ("Gemma", "google/gemma-4-E4B-it"),
    ("Gemma", "google/gemma-4-26B-A4B-it"),
    ("DeepSeek", "deepseek-ai/DeepSeek-R1-0528"),
    ("DeepSeek", "deepseek-ai/DeepSeek-V3.2"),
    ("DeepSeek", "deepseek-ai/DeepSeek-V4-Flash"),
    ("Mistral", "mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
    ("Mistral", "mistralai/Magistral-Small-2509"),
    ("Mistral", "mistralai/Ministral-3-14B-Reasoning-2512"),
    ("Mistral", "mistralai/Mistral-Small-4-119B-2603"),
    ("GPT-OSS", "openai/gpt-oss-20b"),
    ("GLM", "zai-org/GLM-5.3-Flash"),
]

FILES = ["tokenizer*", "*.jinja", "chat_template*",
         "special_tokens_map.json", "tekken.json", "config.json",
         "generation_config.json", "*.model", "vocab*", "merges.txt",
         "encoding/*.py"]

# what vLLM's ChatCompletionRequest.build_chat_params passes to the
# template for the server's reasoning_effort="none"
TEMPLATE_KWARGS = dict(add_generation_prompt=True,
                       continue_final_message=False, documents=None,
                       reasoning_effort="none", enable_thinking=False)


def load_overlay():
    """The overlay's decisions package, through the test stubs (no vLLM
    install needed)."""
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from vllm_stubs import install
    install()
    gen = importlib.import_module("vllm.entrypoints.generate")
    gen.__path__ = [os.path.join(REPO, "overlay", "vllm", "entrypoints",
                                 "generate")]
    serving = importlib.import_module(
        "vllm.entrypoints.generate.decisions.serving")
    slot = importlib.import_module(
        "vllm.entrypoints.generate.decisions.answer_slot")
    wide = importlib.import_module(
        "vllm.entrypoints.generate.decisions.backends.large_choice")
    return serving, slot, wide.wide_direct_markers


def messages(serving):
    opts = [SimpleNamespace(description=d) for d in
            ["The claim is supported", "The claim is contradicted",
             "Not enough evidence", "The claim is off topic"]]
    return [
        {"role": "system", "content": serving.DECISION_SYSTEM},
        {"role": "user", "content": serving._user_payload(
            "The sky was clear all day.",
            "Does the evidence support: it rained?", opts)},
    ]


def render(repo, msgs):
    """(tokenizer, prompt ids, how it was rendered). Like the server: a
    template that refuses the thinking-off switch renders without it."""
    try:
        return _render(repo, msgs, thinking_off=True)
    except Exception:
        tok, ids, how = _render(repo, msgs, thinking_off=False)
        return tok, ids, how + ", refuses thinking-off"


def _render(repo, msgs, thinking_off):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    d = snapshot_download(repo, allow_patterns=FILES)
    has = set(os.listdir(d))
    kwargs = dict(TEMPLATE_KWARGS)
    if not thinking_off:
        kwargs.pop("reasoning_effort")
        kwargs.pop("enable_thinking")
    if "tekken.json" in has and "tokenizer_config.json" not in has:
        from transformers.tokenization_mistral_common import (
            MistralCommonBackend)
        tok = MistralCommonBackend.from_pretrained(d)
        # vLLM passes reasoning_effort to Mistral tokenizers from v15 on
        version = int(str(tok.tokenizer.instruct_tokenizer.tokenizer
                          .version).rsplit("v", 1)[-1])
        extra = ({"reasoning_effort": "none"}
                 if thinking_off and version >= 15 else {})
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      tokenize=True, return_dict=False,
                                      **extra)
        return tok, list(ids), "mistral_common"
    tok = AutoTokenizer.from_pretrained(d)
    scripts = glob.glob(os.path.join(d, "encoding", "encoding_ds*.py"))
    if scripts and not tok.chat_template:
        spec = importlib.util.spec_from_file_location("ds_encoding",
                                                      scripts[0])
        enc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enc)
        text = enc.encode_messages(msgs, thinking_mode="chat")
        return tok, tok.encode(text, add_special_tokens=False), \
            os.path.basename(scripts[0])
    text = tok.apply_chat_template(msgs, tokenize=False, **kwargs)
    return tok, tok.encode(text, add_special_tokens=False), "chat template"


def check(repo, family, serving, slot, wide_direct):
    row = {"family": family, "repo": repo}
    try:
        tok, ids, how = render(repo, messages(serving))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        return row
    row["render"] = how
    row["ends"] = tok.decode(ids[-6:])
    closing = slot.closing_suffix(tok.decode(ids[-16:]))
    row["closed_with"] = "".join(closing) if closing else None
    if closing:
        ids = ids + slot.encode_suffix(tok, closing)
    letters = [chr(65 + i) for i in range(26)]
    try:
        slot.answer_slots(tok, ids, letters)
        row["letters"] = "all 26 read"
    except slot.SlotError as e:
        row["letters"] = f"refused: {e}"
    try:
        row["wide_direct"] = len(wide_direct(tok, tuple(letters)))
    except Exception as e:
        row["wide_direct"] = f"error: {e}"
    row["fallbacks"] = [
        "".join(b) for b in slot.EMPTY_REASONING_BLOCKS
        if slot.knows_suffix(tok, b)]
    return row


def verdict(row):
    if "error" in row:
        return "not checked"
    if not row["letters"].startswith("all"):
        return "REFUSED"
    if row["closed_with"]:
        return "ready (ending closed)"
    return "ready"


def report(rows) -> str:
    out = [
        "# Tokenizer and chat-template check",
        "",
        "Built the real decision prompt for each model, the way the server",
        "does, and checked whether the answer letter comes next. No GPU: this",
        "checks the tokenizer and chat template, not the model's weights.",
        "",
        "| Family | Model | Verdict | Thinking-off switch | "
        "Prompt ends with | Closed with | Letters | Wide-direct capacity |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ends = repr(r.get("ends", "")).replace("|", "\\|")
        closed = repr(r["closed_with"]).replace("|", "\\|") \
            if r.get("closed_with") else "-"
        switch = ("-" if "render" not in r else "refused, rendered without"
                  if "refuses" in r["render"] else "accepted")
        out.append(
            f"| {r['family']} | `{r['repo']}` | {verdict(r)} | {switch} | "
            f"`{ends}` | "
            f"`{closed}` | {r.get('letters', r.get('error', ''))} | "
            f"{r.get('wide_direct', '-')} |")
    out += [
        "",
        "## How to read it",
        "",
        "- **ready**: the prompt ends where the model's next token is its",
        "  answer letter, and every letter can be read.",
        "- **ready (ending closed)**: the chat template left a reasoning",
        "  block or output channel open; the server closes it (column",
        "  *Closed with*) before reading.",
        "- **REFUSED**: the server would refuse these requests; the",
        "  *Letters* column says why.",
        "- **Thinking-off switch**: the server asks the template for no",
        "  reasoning. A template that refuses (a reasoning-only model) is",
        "  rendered without it; the self-check below then decides.",
        "- **Wide-direct capacity**: how many options one pass can read",
        "  (26 letters plus the letter pairs this tokenizer keeps as one",
        "  token). Above that, the server uses two-stage.",
        "",
        "Some models open a reasoning block by themselves even when the",
        "template ends at the answer (DeepSeek-R1, possibly the Mistral",
        "reasoning models). Only the loaded model shows that: the server's",
        "self-check measures it on the first request and, if needed, adds",
        "an empty reasoning block the tokenizer supports:",
        "",
    ]
    for r in rows:
        if "fallbacks" in r:
            fb = ", ".join(f"`{f!r}`" for f in r["fallbacks"]) or "none"
            out.append(f"- `{r['repo']}`: {fb}")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="*",
                    help="Hugging Face repos (default: the built-in list)")
    ap.add_argument("--report", help="write the Markdown report here")
    args = ap.parse_args(argv)
    models = ([("-", m) for m in args.models] if args.models
              else DEFAULT_MODELS)
    serving, slot, wide_direct = load_overlay()
    rows = []
    for family, repo in models:
        row = check(repo, family, serving, slot, wide_direct)
        rows.append(row)
        print(f"{verdict(row):22s} {repo}", flush=True)
    text = report(rows)
    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"report: {args.report}")
    else:
        print(text)
    return 0 if all(verdict(r).startswith("ready") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
