# v1/decisions: Typed Decisions for vLLM

This repository proposes a first-class vLLM endpoint, **/v1/decisions**. It unifies the existing typed-decision (Jev-compatible) backends under one API, and is designed so the protocol is easy to extend in future open-source work.

A typed decision returns a probability distribution over a fixed set of options instead of generated text. You send the input (the evidence) and one or more typed questions. For each question, the server reads the answer from the model's logits in a single forward pass and returns calibrated probabilities.

/v1/decisions speaks OpenAI's Decisions format, so a client built with the OpenAI SDK can point at a self-hosted model. Everything OpenAI's format has no field for (per-request calibration, backend selection and options, a seed, and diagnostics) goes in an `extra` block. Backends are pluggable, and so are question types (see [Extending the API](#extending-the-api)). /v1/systemone speaks the request format of TypeSafe's Jev API, so existing Jev clients can point at a self-hosted model too. Both are answered by the same code path: the same question gives the same probabilities on either endpoint.

The endpoint is proposed for upstream vLLM. This repository contains a patch with all the changes demonstrating the
reference implementation, built as an overlay on stock vLLM **v0.30.0**.

Three backends are included:

* **`logit`**: the default, for any generative checkpoint vLLM serves. It
renders the question with the model's own chat template and reads the
next-token log-probabilities over the option labels at the answer
position. It needs no training and no extra model, so the chat model
you already serve answers decisions too. At startup it checks that the
model's next token really is an option label, and fits a calibration
temperature.
* **`encoder`**: for Laya decision models. One pooling pass scores every
option. It needs the vendored Laya pooling patch from
[#58429](https://github.com/vllm-project/vllm/pull/58429).
* **`canvas`**: for DiffusionGemma. One read-only canvas denoise gives a
distribution over the option letters at the answer slot. It is seeded,
not deterministic, and its accuracy isn't measured yet. It needs the
vendored patch from [#57250](https://github.com/vllm-project/vllm/pull/57250).

The server picks a backend at startup. It uses `VLLM_TYPED_DECISIONS_BACKEND`
if set. Otherwise it uses the backend that claims the served model's
architecture, and falls back to `logit`. A request can choose another
loaded backend with `extra.backend`.

A new backend is one class: a `name`, the architectures it serves, and a
`read(question, request_id)` method that returns a score per option.
Register it by name or through the `vllm.decision_backends` entry point.
Question types are pluggable the same way. See
[Extending the API](#extending-the-api).



## Results (Logit Backend)

Stock weights, no training or fine-tuning. The server renders each
question with the model's own chat template and reads its next-token
probabilities over the option labels. Measured September 2026 on vLLM
v0.30.0 with this package, 2 consumer GPUs (RTX 3090 Ti + 3090).

**Accuracy on JevBench public (231 tasks), Qwen3.8-27B INT4:**

||easy|original|hard|
|-|-:|-:|-:|
|This, `/v1/systemone`|1.000|0.931|0.730|
|Jev hosted API|1.000|0.986|0.730|

**Speed, JevBench board method** (one request at a time, caller wall
time, the board's formula and self-hosted adjustment of latency ×2 +
0.15 s): p50 0.164 s, p95 0.455 s, speed score **83.0** (91.3
unadjusted). Jev's hosted API scores 83.3. `/v1/decisions` runs the same code path
but was not timed separately.

**Many questions about one input.** Put all questions in one request and
the shared input is prefilled once through vLLM's prefix cache. On
Qwen3-4B (BF16) with a ~4.2k-token state and 16 questions:

||per decision|decisions/s|
|-|-:|-:|
|16 one-question requests, one at a time|25 ms|~40|
|One 16-question request|**6 ms**|**157**|

About 4× faster per decision, with 93–99.7% of input tokens served from
cache. This needs a model whose prefix cache works in vLLM: on hybrid
(Mamba/GDN) models such as Qwen3.8-27B, vLLM v0.30.0 reuses no prefix
even with `--enable-prefix-caching --mamba-cache-mode align`, so each
question re-reads the state (see [Performance](#performance)).

**What this means**

* The chat model you already serve answers typed decisions too: no
second model, no extra VRAM, no extra service.
* It works on any generative checkpoint vLLM serves; the startup
self-check refuses models whose next token isn't an option label.
* Every answer carries raw option logits and an audit record, so
probabilities can be recalibrated, compared or logged downstream.
* These numbers recorded below are from off-the-shelf models without any decision post-training.
* It's theoretically possible to post-train a model for more accurate decision outputs while also keeping its chat output, but finding that balance requires testing.

## Contents

|Path|What it is|
|-|-|
|`overlay/vllm/entrypoints/generate/decisions/`|The endpoint package: routers, request/response models, serving, backends|
|`api-router-wire-v0.30.patch`|The wiring patch (three files in `vllm/entrypoints/`)|
|`install.sh`|Copies the package and applies the patch in an installed vLLM|
|`patches/00572-diffusiongemma-structured.patch`|Engine support needed by the `canvas` backend (vendored from [#57250](https://github.com/vllm-project/vllm/pull/57250), plus a `diffusion_seed` field that is validated and stored but not yet used)|
|`patches/00584-vllm-laya-pooling.patch`|Laya pooling model needed by the `encoder` backend (vendored from [#58429](https://github.com/vllm-project/vllm/pull/58429))|
|`patches/max-logprob-token-ids.patch`|Optional: raises vLLM's 128 token-id cap on a restricted read to 600, so one-pass `exact` reads cover up to 255 options (see [Read limits](#read-limits)). Not needed with `logprobs: top-k`|
|`overlay/.../decisions/calibration_data/`|JevBench public (231 questions, MIT, credit in `NOTICE`), the default set for [startup calibration](#startup-calibration)|
|`tools/`|Live test suites (`test_decisions.py`, `test_systemone.py`), `check_tokenizers.py` (answer-slot check on real tokenizers, no GPU), `capture_logits.py` (raw option logits for fitting T by hand)|
|`tests/`|Offline tests (no GPU, no vLLM install needed)|

## Quickstart

Install into an image or environment with vLLM 0.30.0, from a clone of
this repository (the script needs `patch`). Tested from a clean clone on
the `vllm/vllm-openai:v0.30.0` image:

```bash
./install.sh            # finds vLLM via `import vllm`; exits non-zero on any failure
```

`install.sh` is idempotent. To install by hand instead, apply the patch from
the directory that **contains** the `vllm/` package, then copy the package:

```bash
SITE=$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')
(cd "$SITE" && patch -p1 < /path/to/api-router-wire-v0.30.patch)
cp -r overlay/vllm/entrypoints/generate/decisions "$SITE/vllm/entrypoints/generate/"
```

Start the server with the endpoints enabled:

```bash
VLLM_ENABLE_TYPED_DECISIONS=1 vllm serve Qwen/Qwen3.5-0.8B --max-logprobs 128
```

`--max-logprobs 128` lets one `exact` read cover up to 128 options; with
`logprobs: top-k` no flag is needed (see [Read limits](#read-limits)). At startup the server checks the model and
fits its calibration, which takes seconds on a small model and about two
minutes on Qwen 3.8 27B. Until then decision requests return 503 with
`Retry-After`; `/health` is up earlier.

Ask a question:

```bash
curl -s localhost:8000/v1/decisions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.5-0.8B",
  "input": "Ticket: My payouts have been failing for 3 days.",
  "questions": [
    {"type": "choice", "name": "department", "instructions": "Which team should handle this?",
     "choices": [{"value": "billing", "description": "Payments, refunds"},
                 {"value": "technical", "description": "Bugs, outages"}]}
  ]
}'
```

## Configuration

|Variable|Values|Effect|
|-|-|-|
|`VLLM_ENABLE_TYPED_DECISIONS`|`1`, `true`, `yes`|Registers both routes. Unset: the routes don't exist and no serving state is created.|
|`VLLM_TYPED_DECISIONS_BACKEND`|`name[:key=value,...]`|Startup backend and its constructor arguments. Unset: `canvas` on DiffusionGemma checkpoints, `logit` on everything else.|
|`VLLM_TYPED_DECISIONS_TEMPERATURE`|float > 0, unset by default|A hand-set calibration temperature for every answer that doesn't send its own. Setting it skips startup calibration. See [Calibration](#calibration).|
|`VLLM_TYPED_DECISIONS_CALIBRATION`|unset/`jevbench`, `on`, a file path, `off`|Startup calibration. Unset: on for `logit`, off for the other backends. See [Startup calibration](#startup-calibration).|
|`VLLM_TYPED_DECISIONS_CALIBRATION_DIR`|a directory, default `$HF_HOME/decisions-calibration`|Where saved calibration results live.|
|`VLLM_TYPED_DECISIONS_MIN_OPTION_MASS`|0–1, default `0.5`|The answer-slot self-check threshold (logit backend). `0` turns the check off. See [`logit`](#logit).|
|`DECISIONS_READ_RETRIES`|int ≥ 0, default `2`|Re-issues of a restricted read whose reported logprobs lack requested ids (a known transient engine gather defect; see the fork's `patch_decision_logprob_chunking`). `0` disables retrying.|
|`DECISIONS_READ_RETRY_BACKOFF_S`|float ≥ 0, default `0.25`|Seconds before retry n (scaled by n). A read still missing ids after the retries either falls back to a degraded generative read (logit `direct`/`wide-direct`, marked `degraded: true` in `meta`) or fails per question as before.|

There are no CLI flags; every setting is an environment variable.

Backend arguments are split from the name at the first `:` and separated by
`,`. Values are parsed as int, float or bool, otherwise kept as strings.
Unknown arguments fail at startup.

```bash
VLLM_TYPED_DECISIONS_BACKEND=canvas:samples=5,max_steps=2
```

A request can pick a different backend by name with the `backend` field.
Arguments can't be passed per request; a backend selected that way is built
with its defaults.

Engine requirements:

* Prefix caching must be on (the vLLM V1 default). Multi-question requests
share one state prefix; without the cache, each question prefills the
state again.
* The `encoder` backend needs `patches/00584` and a Laya checkpoint.
* The `canvas` backend needs `patches/00572` and a DiffusionGemma checkpoint.
* Both patches touch `docs/`, `examples/` and `tests/` as well as `vllm/`, so
apply them to a v0.30.0 source checkout.

## Limits

Every limit is a startup setting and applies to both endpoints.
Defaults: 255 options per choice question (Jev's limit), 10 score
levels, 64 questions per request. A request over a limit is refused
before any model work (400 on `/v1/decisions`, 422 on `/v1/systemone`),
and the message names the setting to change.

### Read limits

A one-pass `exact` read (see [Logprobs](#logprobs-exact-or-top-k)) asks
the engine for one logprob per option label, so it is bounded by vLLM
itself:

* `--max-logprobs` (default 20): the most logprobs one request may ask
for. `-1` means no limit.
* `MAX_LOGPROB_TOKEN_IDS` in `vllm/sampling_params.py` (128 in v0.30.0):
the most token ids one request may name.

The server reads both at startup and logs what fits, for example
`decision readouts: direct up to 26, wide-direct up to 128 (exact), top-k wide-direct up to 386, two-stage beyond (read limit 128: max_logprobs=600, token-id cap=128)`. `auto`
never picks a read that doesn't fit: past the limit an `exact` read
switches to a `top-k` wide-direct read (see [Logprobs](#logprobs-exact-or-top-k)),
which has no per-label limit, and only past the tokenizer's wide-direct
capacity does it use the two-stage read. A request that explicitly asks
for an `exact` one-pass read that doesn't fit gets a per-question error
naming the setting to raise.

A `top-k` read names no token ids, so these limits don't bound it:
`auto` uses one-pass wide-direct up to the tokenizer's capacity on any
setup, and `--max-logprobs` only sets the size of the window it reads
(bigger is closer to `exact`).

|Server setup|`exact` up to (then `auto` reads `top-k`)|One pass up to, `top-k`|
|-|-:|-:|
|stock vLLM, default `--max-logprobs`|20 (markers capped to 20, with a warning)|255 (window 20)|
|stock vLLM, `--max-logprobs 128` or more|128|255 (window 128)|
|`--max-logprobs 600` + `patches/max-logprob-token-ids.patch`|255 (the tokenizer's wide-direct capacity on Qwen)|255 (window 600)|

Two-stage is the last resort, past the tokenizer's wide-direct
capacity (386 on Qwen); 255 is the package's option limit, so on Qwen
`auto` never reaches it.

The optional patch is a one-line change to that constant; `install.sh`
doesn't apply it. An explicitly configured marker set larger than
`--max-logprobs` refuses to start.

|Variable|Default|Meaning|
|-|-|-|
|`VLLM_TYPED_DECISIONS_MAX_OPTIONS`|255|Options per choice question|
|`VLLM_TYPED_DECISIONS_MAX_SCORE_LEVELS`|10|Levels per score question|
|`VLLM_TYPED_DECISIONS_MAX_QUESTIONS`|64|Questions per request|
|`VLLM_TYPED_DECISIONS_MARKERS`|`A-Z`|Option markers for the direct read, in order. Comma-separated; `X-Y` expands a character range|
|`VLLM_TYPED_DECISIONS_SHORTLIST`|16|Finalists kept for stage 2 of the two-stage read|

### How options are read (logit backend)

* **Direct read (up to one option per marker, 26 by default).** One
forward pass. Every marker must be a single token.
* **Wide-direct read (more options than markers).** One forward pass,
with extra labels made of letter pairs the tokenizer keeps as single
tokens. Its capacity depends on the tokenizer (255 on Qwen) and on
the [read limits](#read-limits).
* **Two-stage read (beyond wide-direct's capacity).** Stage 1 scores
each option independently (yes/no over the shared, prefix-cached
state); the top `SHORTLIST` options go through a direct read; the
finalists share the stage-1 mass. Costs k+1 reads.

The default, `auto`, picks the first read that fits: direct, then
wide-direct (`exact` within the read limits, `top-k` past them), then
two-stage. In paired A/B runs on 32 to 255 options,
wide-direct was more accurate than two-stage on Qwen3-4B (+0.12 pooled,
97.5% CI +0.07 to +0.18) and level on Qwen3.8-27B at 32 and 64 options
(+0.06 at 124). It takes one pass instead of k+1, and on the 4B it
answered about 2× faster at 64 and 255 options.

Pick one per request with `"extra": {"backend_options": {"readout": "auto" | "direct" | "wide-direct" | "two-stage"}}`. (A two-step "prefixed" read
was also tried; it lost to wide-direct and was removed.)

#### Logprobs: `exact` or `top-k`

How each read gets the markers' log-probabilities:

* **`exact`**: the engine returns the log-probability of
exactly the option markers (`logprob_token_ids`). Every option is read
exactly, but a one-pass read is bounded by the [read limits](#read-limits),
and under speculative decoding (MTP) the engine returns incomplete
reads (vLLM issue 42592); those fall back to a degraded read.
* **`top-k`** (the default): the engine returns its plain top-k list,
k = `--max-logprobs` (20 on stock vLLM). A marker outside the list
scores the list's lowest log-probability, an upper bound on its true
value, and is named in `meta.floored`; `option_mass` then counts only
the markers actually read. No per-label limit, so `auto` uses
wide-direct up to the tokenizer's capacity on a stock server, and it
works under speculative decoding.

Set the server default with
`VLLM_TYPED_DECISIONS_BACKEND=logit:logprobs=exact`, or per request with
`"extra": {"backend_options": {"logprobs": "exact"}}`. The startup log names the
window: `logit logprobs: top-k (window 20 = --max-logprobs); ...`.

Measured on Qwen3.8-27B INT4, stock vLLM v0.30.0 with the default
`--max-logprobs` (window 20), 100 questions per size, accuracy against
the correct answer:

|Options|`exact` (two-stage)|`top-k` (wide-direct)|`exact` wide-direct, patched server|
|-|-:|-:|-:|
|32|95 (5.2 s)|94 (0.6 s)|95|
|64|85 (9.8 s)|95 (1.2 s)|94|
|128|83 (19.3 s)|90 (2.1 s)|92|
|255|62 (37.9 s)|84 (4.0 s)|81|

The correct option was inside the 20-wide window on every question. On
JevBench (2 to 6 options) the two gave the same answer on all 231
questions (199/231 correct), with probabilities within 0.00002.

What each costs:

* `top-k` with a small window ties every option outside it at the
window's floor. The top answer is unaffected, but those options'
probabilities are overstated: calibration error at 255 options was
0.141 with a 20-wide window against 0.066 for `exact`. With a 256-wide
window it matched `exact` (probabilities within 0.002).
* `exact` under speculative decoding: with MTP on and chat generating
at the same time, the engine lost requested logprobs on 123 of 231
reads (vLLM issue [#42592](https://github.com/vllm-project/vllm/issues/42592),
fix pending in [#44727](https://github.com/vllm-project/vllm/pull/44727)).
Every request still got an answer, but those answers came from the
degraded fallback (4 engine passes, `degraded: true`). Two-stage
`exact` reads have no fallback and fail per question. With MTP and no
other traffic, `exact` reads were clean. `top-k` was clean under the
same load: 331/331 answers, one pass each, none degraded.

Use `top-k` on a stock server, with speculative decoding, or past 20
options without raising `--max-logprobs`. Use `exact` when you need the
true score of every option and the server has no speculative decoding.

The encoder and canvas backends use a direct read only, so they refuse
more options than markers.

## `POST /v1/decisions`

OpenAI's Decisions format (OpenAI's public beta of 2026-10-06), plus an
`extra` block for everything OpenAI's format has no field for. A client
built with the OpenAI SDK works unchanged (`client.decisions.create(...)`
with `base_url` pointing at the server); the `extra` block goes in
`extra_body`. Where OpenAI leaves behaviour open, the endpoint does what
vLLM's own PR for the API does
([#60465](https://github.com/vllm-project/vllm/pull/60465)). The
deliberate differences are listed in
[Differences from OpenAI and from upstream](#differences-from-openai-and-from-upstream).

### Request

```jsonc
{
  "model": "Qwen/Qwen3.8-27B",
  "input": "Ticket: My payouts have been failing for 3 days.",
  "questions": [
    {"type": "predicate", "name": "is_urgent",
     "instructions": "Does this convey urgency?"},
    {"type": "choice", "name": "department", "instructions": "Which team?",
     "choices": [{"value": "billing", "description": "Payments, refunds"},
                 {"value": "technical", "description": "Bugs, outages"}]},
    {"type": "score", "name": "frustration",
     "instructions": "How frustrated is the customer?",
     "levels": [{"label": "Calm"}, {"label": "Frustrated"},
                {"label": "Very angry", "description": "threatens to leave"}]}
  ],
  "extra": {"calibration_temperature": 1.2, "detail": "basic"}   // optional
}
```

|Field|Default|Meaning|
|-|-|-|
|`model`|required|Echoed in the response; `jev-latest` / `jev-preview` resolve to `jev-1.13.0`. The served checkpoint is whatever vLLM loaded|
|`input`|required|A string, or a list of user messages whose `content` is a string or `input_text` parts. Parts are joined with a newline, messages with a blank line. `input_image` is refused: no backend reads images yet|
|`questions`|required|1 to 200 questions (and at most the server's limit, 64 by default), answered in order|
|`safety_identifier`|none|Accepted (up to 128 characters) and ignored|
|`extra`|`{}`|Our settings, below|

**Question types.**

|Type|Fields|Answer fields|
|-|-|-|
|`predicate`|`instructions`, `name?`|`probability`: P(true)|
|`choice`|`instructions`, `name?`, `choices`: 2 to 255 of `{value, description?}`; a value is a string or a boolean, and `"true"` and `true` are different choices|`choice`: the most probable value; `probabilities`: `[{value, probability}]` in choice order; `confidence`|
|`score`|`instructions`, `name?`, `levels`: 2 to 10 of `{label, description?}`, lowest first|`score`: Σ i·pᵢ over level indices, can fall between levels; `probabilities`: `[{value: i, label, probability}]`; `confidence`|

A question may also be any registered plugin type
([Extending the API](#extending-the-api)): its own fields, checked by the
type, plus the optional `name`. Its answer is the type's own answer
fields with `name` and `type`, the same as on `/v1/systemone`. Jev's
`noul` is not accepted here; `predicate` is the yes/no type.

Each question is read as one of Jev's question types (a predicate as a
`noul` without criteria, a choice as a `choice`, a score as a `score`),
so it renders exactly as the same Jev question does and the saved
calibration applies. A choice without a description renders as its
value; a level renders as its label, or `label: description`.

**`extra` (request).**

|Field|Default|Meaning|
|-|-|-|
|`calibration_temperature`|the server's T|T > 0; probabilities are `softmax(scores / T)`. See [Calibration](#calibration)|
|`backend`|startup backend|`logit`, `encoder`, `canvas`, or a registered plugin|
|`backend_options`|none|Settings for that backend, validated by it. `logit`: `readout`, `logprobs`. `canvas`: `samples`, `max_steps`. `encoder`: none. A backend that takes none refuses any|
|`seed`|none|Seeds backends that sample (`canvas`); ignored by the others|
|`detail`|`"full"`|How much of each answer's `extra` to return: one level for both blocks, or a map per block, e.g. `{"audit": "full", "backend": "none"}`. Levels: `full`; `basic` (without per-option lists such as `option_logits`); `none` (leave the block out)|

**Validation.** As in #60465: unknown fields are refused everywhere
(ours go under `extra`), an explicit `"name": null` is refused, choice
values must be distinct by type and value, and names need not be
unique. Limits are in [Limits](#limits). An invalid request returns 400.

### Response

```jsonc
{
  "model": "Qwen/Qwen3.8-27B",
  "answers": [
    {"type": "predicate", "name": "is_urgent", "probability": 0.91, "extra": {…}},
    {
      "type": "choice", "name": "department", "choice": "billing",
      "probabilities": [{"value": "billing", "probability": 0.87},
                        {"value": "technical", "probability": 0.13}],
      "confidence": 0.74,
      "extra": {
        "backend": {
          "name": "logit",
          "option_logits": {"\"billing\"": -0.21, "\"technical\"": -2.11}
        },
        "audit": {
          "served_model": "Qwen/Qwen3.8-27B",
          "state_sha256": "3f89c2…",
          "confidence_formula": "normalized-peak-v1",
          "render_version": "2026-09-29.1",
          "calibration_temperature": 1.0,
          "temperature_source": "calibrated",
          "forward_passes": 1,
          "input_tokens": 103,
          "cached_input_tokens": 96,
          "option_mass": 0.98,
          "readout": "direct",
          "label_layout": "direct"
        }
      }
    },
    {"type": "refusal", "name": "frustration", "extra": {"error": "…"}}
  ],
  "usage": {
    "input_tokens": 309,
    "input_tokens_details": {"cached_tokens": 192, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 309
  },
  "extra": {"id": "decisions-6f1c…", "created": 1790000000}
}
```

* Answers come back in question order, each with its `name` (`null`
when the question had none).
* `confidence`: `(k·max − 1) / (k − 1)` over the calibrated
probabilities. 1.0 means all mass on one option; 0.0 means uniform.
OpenAI doesn't publish its formula; `audit.confidence_formula` names
ours.
* **Refusals.** Questions in one request run concurrently. A question
that fails (for example, more options than its read can hold) comes back
as a `refusal` answer, OpenAI's answer type for a declined question,
with the reason in `extra.error` (kept at every `detail`). The other
questions still answer. When every question fails, the request fails:
it returns the first failure in question order, with that failure's
status code.
* `usage` is OpenAI's shape. `input_tokens` sums the answers' prompt
tokens; `cached_tokens` is how many of those the engine served from its
prefix cache (0 when the backend can't tell). A multi-question request
over one input shows the saving directly: after the first question,
most of each prompt comes from the cache. `output_tokens` is 0: no
answer text is generated beyond a one-step decode (`logit`) or a canvas
read (`canvas`), which aren't counted.
* `extra.id` and `extra.created` identify the response.

**`extra` (answer)** has two blocks, trimmed by the request's `detail`:

* `backend`: what the backend did. Always `name` and `option_logits`
(the raw score per option, keyed by the option's id: a choice value as
JSON text, `true`/`false` for a predicate, `"0"`… for score levels;
`softmax(option_logits)` is the uncalibrated distribution), plus the
backend's own facts: the logit backend's `logprobs` and `topk_window` on
a `top-k` read, `floored` (options outside the window, scored at its
floor), `degraded` and `degraded_reason` on a fallback read; two-stage's
`stage1_scores` and `shortlist`, canvas's `seed`, `samples` and
`canvas_width`, the encoder's `prompt_source`, or a plugin's fields.
* `audit`: the same facts on every backend, to check how the answer was
produced:

|`audit` field|Meaning|
|-|-|
|`served_model`|The checkpoint that answered|
|`state_sha256`|SHA-256 of the input as read (the joined text)|
|`confidence_formula`|Which confidence statistic was used|
|`render_version`|Version of the prompt layout. It changes whenever the rendered prompt would change, which also invalidates saved calibration results|
|`calibration_temperature`|T actually applied (1.0 = raw)|
|`temperature_source`|Where that T came from: `request`, `server` (operator-set), `calibrated` (startup calibration) or `default` (1.0, never measured). See [Calibration](#which-t-an-answer-gets)|
|`forward_passes`|Engine passes for this answer (`canvas` with `samples=n` reports n)|
|`input_tokens`|Prompt tokens for this question, over every engine read it took (two-stage: k+1 reads; canvas: one per sample)|
|`cached_input_tokens`|How many of those prompt tokens the engine served from its prefix cache instead of recomputing. `null` when the backend can't tell|
|`option_mass`|Logit backend, direct and wide-direct reads: the share of the model's whole next-token probability that landed on any option (Σ exp(logprob) over the option tokens), before the probabilities are renormalized over the options. Near 1: the model clearly wanted to answer with an option. Low (say below 0.5): the options don't fit the question or the prompt confused the model, so treat the answer with suspicion. On a `top-k` read it counts only the options inside the window, so it can only understate. `null` for two-stage, encoder and canvas reads, and when the server runs with a `--logprobs-mode` other than `raw_logprobs`|
|`readout`|Which read produced the answer (`direct`, `wide-direct`, `two-stage`, `direct-degraded` for the fallback after an incomplete `exact` read, or a plugin's name)|
|`label_layout`|How the options were labelled (`direct` letters, or `merged-pairs` for wide-direct)|

### Differences from OpenAI and from upstream

Same as vLLM's PR (#60465): the request fields, refusing unknown
fields, the input joining, refusing images, the name rules, typed choice
values, 2 to 10 levels, the score as a weighted average of level
indices, OpenAI's usage shape, and answers in question order.

Deliberately different, each one additive (a client that ignores
`extra` sees a normal OpenAI response):

||OpenAI / #60465|This endpoint|Why|
|-|-|-|-|
|`extra` block|none / refused|accepted on the request, returned on answers and the response|backends, calibration and the audit need a place|
|Choices per question|not stated / 26 (255 in its schema)|255 (`VLLM_TYPED_DECISIONS_MAX_OPTIONS`); past the marker count the `logit` backend reads wide-direct, `top-k` past the [read limits](#read-limits)|measured: see [Logprobs](#logprobs-exact-or-top-k)|
|Questions per request|not stated / 200|200, and at most the server's limit (64 by default)|unchanged from 0.1|
|`confidence`|unpublished / p_top × option mass|normalized peak after calibration|same numbers as `/v1/systemone`|
|A failed question|a question may be declined as a `refusal` / whole request fails|`refusal` answer with `extra.error`; the request fails only when every question failed|one bad question shouldn't lose the rest|
|Prompt renders|unpublished / own renders|Jev's renders, byte for byte|the saved calibration stays valid|
|Invalid request|400 / 400|400|same|
|Images|accepted / refused|refused|no backend reads images yet|
|Question types|predicate, choice, score|those three, plus any registered plugin type, answered with the type's own fields|a plugin type is reachable on both endpoints|
|Backends, calibration, startup self-check|n/a / logit only, none|`logit`, `encoder`, `canvas`, plugins; startup calibration; answer-slot self-check|the point of this package|

## `POST /v1/systemone`

The Jev request and response format. A request is
`{model, state, questions}` (plus an optional `backend`), where
`questions` maps an id you choose to a typed question:

```jsonc
{
  "model": "jev-latest",
  "state": "Ticket: My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent":   {"type": "noul",   "instructions": "Does this convey urgency?",
                    "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
    "department":  {"type": "choice", "instructions": "Which team?",
                    "criteria": {"billing": "Payments, refunds", "technical": "Bugs, outages"}},
    "frustration": {"type": "score",  "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  },
  "extra": {"detail": "none"}   // optional, as on /v1/decisions
}
```

|Type|`criteria`|Answer fields|
|-|-|-|
|`noul`|optional `{"true": …, "false": …}` descriptions|`noul`: P(true)|
|`choice`|option id → description (or `null`), 2 to 255 options|`choice`, `probabilities` (id → p), `confidence`|
|`score`|ordered list of 2 to 10 level descriptions; level ids are `"0"`…`"k-1"`|`score`, `legend` (level id → description), `probabilities`, `confidence`|

Any registered plugin type is accepted too, and its answer comes back
whole (`type`, `probabilities`, `confidence`, the type's own fields,
`extra`), since it has no Jev shape. The server answers through
exactly the same path as `/v1/decisions`, then keeps Jev's fields: the
same question gives the same probabilities on both endpoints
(`tests/test_unify.py` and `tests/test_openai_wire.py` hold that as a
contract).

The request may carry the same `extra` block as `/v1/decisions`
(`calibration_temperature`, `backend`, `backend_options`, `seed`,
`detail`), and each answer carries `extra` (`audit`, `backend`) at that
`detail`, `full` by default; send `"extra": {"detail": "none"}` for
Jev's answer fields only. `backend` may be sent at the top level (as
before) or under `extra`, not both. The response also carries `id`
(`systemone-…`), `created`, `model`, `usage` and, when some question
failed, `partial_failures: {id: message}`.

### Differences from the Jev API

||Jev|This endpoint|
|-|-|-|
|Choice options|up to 255|up to 255 (`VLLM_TYPED_DECISIONS_MAX_OPTIONS`); more than the marker count uses the wide-direct read on the `logit` backend (read `top-k` past the [read limits](#read-limits) with `exact`)|
|Structured `instructions`|field references resolved natively|rendered as JSON; backtick references stay literal text|
|`model`|selects a Jev model|a label only. `jev-latest` and `jev-preview` are echoed as `jev-1.13.0`; other ids are echoed unchanged. The served checkpoint is whatever vLLM loaded.|
|Calibration|server-side|server-side: the operator's T or the startup calibration (see [Calibration](#calibration)); a request may override it with `extra.calibration_temperature`|
|Partial failure|whole request fails|successful answers return; failed ones are listed in `partial_failures`|
|Question types|noul, choice, score|those three, plus any registered plugin type|
|Extra response fields|none|`id`, `created`, and each answer's `extra` (unless `detail` is `none`)|
|Settings at the top level|n/a|`calibration_temperature`, `seed`, `backend_options` are refused with 422: send them under `extra`|

Validation errors return 422.



## Backends

A backend turns one compiled question into a score per option. The
server turns those scores into the answer (temperature, confidence, the
question type's fields, `audit`), so backends can be swapped without
changing the API.

|Backend|Model|Mechanism|Deterministic|Selected automatically for|
|-|-|-|-|-|
|`logit`|any generative checkpoint|option-letter logprobs at the answer position|yes|every model no other backend claims|
|`encoder`|Laya (`token_classify` pooling)|one pooling pass; the pooler scores each option|yes|`LayaForDecision`|
|`canvas`|DiffusionGemma|one read-only canvas denoise; distribution over option letters at the answer slot|no; seeded|`DiffusionGemmaForBlockDiffusion`|

Startup selection: `VLLM_TYPED_DECISIONS_BACKEND` if set, otherwise the
backend that claims the served model's architecture (a plugin's claim wins
over a built-in's), otherwise `logit`. A request can pick another loaded
backend with `backend`.

### `logit`

The server renders the question with the served chat template, with thinking
switched off:

1. The state.
2. The criterion.
3. The options as letters `A`–`Z` with their descriptions.

It then runs one forward pass that gathers logprobs only for the option-letter
tokens. The restricted softmax over those logprobs is the option
distribution; no answer token is sampled.

**The prompt must end where the model's next token is its answer.** Chat
templates differ here, so the server does three things
(`answer_slot.py`):

* **Closes endings left open.** Some templates leave a reasoning block or
output channel open whatever the thinking switch says. GLM-5.x ends on
`<think>`, closed with `</think>`. GPT-OSS ends on `<|start|>assistant`,
closed with `<|channel|>final<|message|>`. A template that refuses the
thinking-off switch (Ministral-3 Reasoning) is rendered without it.
* **Checks the loaded model once, at startup.** A self-check
asks an easy question and measures how much of the model's next-token
probability lands on the option letters. Below
`VLLM_TYPED_DECISIONS_MIN_OPTION_MASS` (default 0.5), the model is about
to write something else first. The server then tries each empty
reasoning block the tokenizer supports. It tries each marker as a
code span - the empty Think tag pair, `[THINK][/THINK]` - and keeps the first that works. This handles models that open a reasoning block by themselves, such as DeepSeek-R1. If none works, every logit read fails with a 503 that says why. While the check (and then [startup calibration](#startup-calibration)) runs, decision requests get a 503 "starting" with `Retry-After`.

* **Reads each letter at its token in context.** The letter's token is
the one that follows the prompt's last plain-text run. It must be one
token, leave the prompt's tokens unchanged, and decode back to the
letter; otherwise the question is refused.

Prompts longer than the model's context are refused, not truncated.

`tools/check_tokenizers.py` runs the first and last of these on real
tokenizers without a GPU. In September 2026 all 17 checked models from
Qwen, Llama, Gemma, DeepSeek, Mistral, GPT-OSS and GLM were ready; GLM-5.3
and GPT-OSS needed their endings closed. Qwen, Llama 3.1, Gemma 4 and
GPT-OSS have since been run live (see [Status](#status)).

This is the default because it works on any generative checkpoint. The same
server can therefore handle chat and decision traffic from one model (see
[Results](#results)). The prompt layout follows SemIf's
`direct-options-v1`, and `tests/golden/` pins it byte for byte.

|`backend_options`|Default|Meaning|
|-|-|-|
|`readout`|`auto`|See [How options are read](#how-options-are-read-logit-backend)|
|`logprobs`|server default (`exact`)|`exact` or `top-k`; see [Logprobs](#logprobs-exact-or-top-k)|

### `encoder`

For Laya checkpoints (`LayaForDecision`, served as a `token_classify`
pooling model through `patches/00584`). Selected automatically for them.

1. The checkpoint's own prompt builder (`laya_prompts.py`) renders the
question using `max_len` and `head_max_len` from the checkpoint's
`laya_config`.
2. One pooling pass returns a score per option.

Laya answers noul, choice and score questions; other question types are
refused per question.

|Startup argument|Default|Meaning|
|-|-|-|
|`truncate_state`|`false`|`false`: a state longer than `max_len` is refused. `true`: Laya's upstream behaviour (keeps the tail of list states).|

### `canvas`

For DiffusionGemma checkpoints (`DiffusionGemmaForBlockDiffusion`), using
the structured-diffusion engine path from `patches/00572`. Selected
automatically for them.

1. The server renders the same prompt as `logit`.
2. It seeds the canvas with a noise token at the answer slot, followed by
the turn-close marker.
3. It runs one read-only denoise and takes the logprobs of each option
letter at that slot.
4. With `samples=n`, it repeats the read with n noise draws from one seed
and averages the distributions. `extra.backend.samples` reports the
per-option standard error and the agreement rate.

The seed used is always reported in `extra.backend.seed`. The seed fixes
the canvas noise; the engine's denoise sampling itself is not seeded, so
individual reads still vary. For stable numbers use `samples=n`, which
averages n noise draws and reports the per-option standard error.

|Startup argument|`backend_options`|Default|Meaning|
|-|-|-|-|
|`canvas_width`|–|64|Canvas length for the request; at most the served canvas length|
|`samples`|`samples` (1–64)|1|Noise draws averaged per question|
|`max_steps`|`max_steps` (1–64)|1|Denoise steps per read|
|`turn_close`|–|106|Turn-close token id|
|`pad`|–|0|Pad token id|

## Extending the API

Both halves of a decision are pluggable, and a plugin needs no change to
the core: a package registers itself at import, directly or through an
entry point the server loads at startup.

**A new backend** (a new kind of decision model): a class with a `name`,
the model `architectures` it should serve by default, an optional
`options_model` for per-request `backend_options`, and
`read(question, request_id)` returning a score per option. It reaches the
engine only through the `BackendHost` it's constructed with (render the
standard prompt, restricted read, generate, pool). Whatever it reports in
`meta` appears under `extra.backend`. Register with `register_backend` or
the `vllm.decision_backends` entry-point group. Full guide:
[`backends/README.md`](overlay/vllm/entrypoints/generate/decisions/backends/README.md).

**A new question type**: a `QuestionType` with a `name`, a pydantic
`model` for the question JSON, `options(question)` returning the closed
option set, and `answer(probabilities, options)` returning the type's
answer fields. Once registered, the type is accepted on both
endpoints: `/v1/decisions` takes it next to OpenAI's three types (plus
OpenAI's optional `name`), and `/v1/systemone` next to Jev's. Both
return its whole answer: `type`, `probabilities`, `confidence`, the
type's own fields, and `extra`. The names `predicate` and `refusal` are
reserved by OpenAI's format. `jev = True` (with `jev_fields`) is only
for Jev's own types, which `/v1/systemone` cuts down to Jev's fields.
An OpenAI SDK client reads a plugin answer's fields as attributes
(`answer.label`); the SDK doesn't type-check answers it doesn't know.
Register with `register_question_type` or the
`vllm.decision_question_types` entry-point group. The built-in noul,
choice and score in `question_types.py` are complete examples.

The one invariant: every question type reduces to a probability
distribution over a closed option set, since that is what every backend
reads. `tests/test_modularity.py` adds a question type and a backend the
way a third-party package would and checks the whole request path.

## Calibration

The API applies one temperature T to the backend's scores:
`probabilities = softmax(scores / T)`. T never changes which option ranks
first, so accuracy is unaffected. It changes how confident the
distribution is. A well-chosen T makes a stated 80% mean "right about 80%
of the time".

A good T depends on the checkpoint, its quantization, the backend and the
workload. The server therefore fits its own T at startup.

### Which T an answer gets

Highest first. The same rule applies to both endpoints, every backend
and every readout.

|#|Source|`audit.temperature_source`|
|-|-|-|
|1|The request's `extra.calibration_temperature`|`request`|
|2|`VLLM_TYPED_DECISIONS_TEMPERATURE`, set by the operator (startup calibration is then skipped)|`server`|
|3|The startup calibration result|`calibrated`|
|4|1.0, raw probabilities (calibration off or failed)|`default`|

A calibration that found no real gain gives T = 1.0 with source
`calibrated`: measured, no correction needed. That is different from
`default`, which means never measured.

Every answer reports the T applied and its source in
`extra.audit.calibration_temperature` and `temperature_source`. The raw
scores are in `extra.backend.option_logits`; `softmax(option_logits)` is
the uncalibrated distribution.

### Startup calibration

`VLLM_TYPED_DECISIONS_CALIBRATION` controls it:

|Value|At startup|
|-|-|
|unset or `jevbench`|Calibrate on the bundled JevBench public set, if the backend calibrates by default (`logit` does; `encoder` and `canvas` don't)|
|`on`|Calibrate on JevBench on any backend|
|a file path|Calibrate on the operator's own questions, on any backend|
|`off`|No calibration; T = 1.0, source `default`|

What happens:

1. The server starts. On the `logit` backend, the answer-slot self-check
runs first (see [`logit`](#logit)).
2. If a saved result matches this deployment, the server loads it and is
ready in about a second.
3. Otherwise it runs the question set through the loaded backend with
its default settings, inside the server (no HTTP), 8 questions at a
time, and records each question's raw option scores and correct option.
4. **Fit T.** T is the value in [0.05, 20] that makes the correct
answers most likely: it minimizes the mean negative log-likelihood
`-mean(log softmax(scores / T)[correct])`, found by golden-section search.
5. **Test T on questions it didn't see.** The questions are split into
5 folds (questions that share a `group` stay in the same fold). For each
fold, T is fitted on the other four and applied to this one, so every
question gets a calibrated answer from a T that never saw it.
6. **Measure the error both ways.** Calibration error (ECE) sorts
answers into 10 equal-width bins by the top option's probability and
takes the weighted mean gap between stated confidence and actual
accuracy: `ECE = sum over bins of (bin size / N) * |accuracy - confidence|`.
It is computed for the raw answers (T = 1.0) and for the held-out
calibrated answers.
7. **Keep T only if the gain is clearly real.** Each ECE gets a 95%
interval: resample the question groups with replacement 1,000 times,
recompute ECE each time, and take the 2.5th and 97.5th percentiles. T is
kept only if the calibrated interval's top is below the raw interval's
bottom, i.e. the two intervals don't overlap. Otherwise the server uses
T = 1.0, source `calibrated` (measured, no correction needed). The T kept
is the one fitted on all questions in step 4.
8. The server saves the result and logs one line, for example
`decision calibration: T=9.51 kept (ECE 0.318 -> 0.122, 231/231 questions, jevbench, 21 s)`.

**Until the startup work finishes, decision requests get a 503** with
`Retry-After: 5` and the message `decision server is starting (self-check / calibration in progress); retry shortly`. vLLM's `/health` is up
before that, so scripts that wait for the server should poll a decision
request, not `/health`.

**Saved results.** They are small JSON files in
`VLLM_TYPED_DECISIONS_CALIBRATION_DIR` (default
`$HF_HOME/decisions-calibration`, which survives container rebuilds when
the Hugging Face cache is mounted). A result is reused only for the same
model (name and revision), quantization, backend and its startup
arguments, prompt version (`render_version`) and question file content.
If any of those change, the server calibrates again. To force a re-run,
delete the file. Each file holds T, whether it was kept, the raw and
calibrated ECE with their intervals, the questions used and skipped, the
key, the date and the vLLM version.

**Operator question file.** JSONL, one question per line:

```jsonc
{"state": "...", "question": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}},
 "expected": "a", "group": "optional-id", "id": "optional"}
```

`question` is a typed question as in a `/v1/systemone` request. `expected` is the correct
option id; for `noul`, `yes`/`no` or `true`/`false`; for `score`, the
level number. Questions that share a `group` stay in the same
cross-validation fold. At least 50 questions must come back with an
answer; the closer they are to real traffic, the better the T.

**Failures.** The server always starts. On any of these it uses T = 1.0,
source `default`, and logs a warning naming the cause:

* the question file is missing, unreadable or malformed;
* fewer than 50 questions came back with an answer;
* the fitted T landed at the edge of the range (0.05 or 20), which points
to broken scores rather than a real T;
* the self-check refused the model. Decision requests are refused with
a 503 anyway, and calibration is skipped.

### Measured results

First-start fits on the bundled JevBench set (231 questions), vLLM
v0.30.0, `logit` backend:

|Model|Fitted T|Kept|ECE raw → calibrated|Calibration time|
|-|-:|-|-|-:|
|Qwen3-4B (BF16)|9.51|yes|0.318 → 0.122|21 s|
|Qwen3.8-27B (INT4)|0.93|no, so T = 1.0|0.037 → 0.033|120 s|
|Llama 3.1 8B|2.58|no, so T = 1.0|0.166 → 0.067|34 s|
|Gemma 4 E4B|3.53|yes|0.185 → 0.070|24 s|

The 4B is strongly overconfident: it states near-certainty on questions
it gets right less than half the time. The 27B is already close to
calibrated. Llama's gain looks large, but at 231 questions its intervals
overlap, so it is not kept.

### Known limits

* **T fits questions like the calibration set.** On very different
questions it is only approximate, and the fitted value moves with the
set. The 4B fits about 7 on a harder authored set and 9.5 on JevBench.
Operators whose traffic looks unlike JevBench should use their own
file.
* **JevBench confidence becomes partly self-graded** when the server was
calibrated on JevBench. Accuracy is unaffected. Published calibration
numbers should say which set T was fitted on.
* **One T per server.** T is fitted for the server's default backend and
its default options. Requests that pick another backend or pass other
`backend_options` (such as a different readout) get the same T.
* **Close calls move slightly between runs.** With the prefix cache on
(the default), a close call's probabilities can shift by up to about
0.2 on small models, depending on what was already cached. A
re-calibration can therefore give a T that differs in the second
decimal. It is harmless, but exact repeatability needs the cache off.
* **`encoder` and `canvas` are unmeasured.** They support calibration
(`on` or a file path), but nobody has measured whether they need it;
`canvas` with `samples=n` also costs n reads per question.

### Fitting T by hand

To fit on a set that isn't in the question-file format, or offline:

1. Capture raw option logits for a labelled set with
`tools/capture_logits.py`.
2. Fit T with SemIf's `calibrate.py` (group-disjoint cross-validation).
3. Set the result as `VLLM_TYPED_DECISIONS_TEMPERATURE`, or send it per
request.



## Performance

Measured on one machine: RTX 3090 Ti + 3090, tensor-parallel, stock vLLM
v0.30.0 with this package installed by `install.sh`, prefix cache on, no
speculative decoding or special KV settings. Absolute numbers will differ
on other hardware.

**Qwen3.8-27B INT4, TP=2** (`--max-logprobs 600` plus the optional
token-id patch; startup calibration fitted T = 0.94, not kept, so T = 1.0):

* JevBench board method, 231 public tasks one at a time: first request
0.190 s, p50 0.164 s, p95 0.455 s; speed score 91.3 unadjusted, 83.0
with the board's self-hosted adjustment; accuracy 196/231 (84.8%).
* Multi-question requests on a very short shared state (~18 tokens):
8 questions 141 ms per decision, 16 questions 127 ms (7.9 decisions/s)
at concurrency 1. Concurrency 4 was slower (2.0–2.3 decisions/s) on
this uneven GPU pair.
* One question on a 32,302-token state: 25.5 s uncached, answered
correctly.
* Prefix cache: reused in 784-token blocks, the same with the default
cache mode and `--mamba-cache-mode align`. A repeat question on a
32k-token state took 0.79 s against 24.8 s for the first (measured with
`--max-logprobs 128`, no patch). An earlier run on this machine showed
0% reuse; that does not reproduce, and the cause is not known.
* Large choice questions and speculative decoding: see
[Logprobs](#logprobs-exact-or-top-k).

**Qwen3-4B BF16, one GPU** (`--max-logprobs 128`), 16 questions over one
~4.2k-token state:

||per decision|decisions/s|input from cache|
|-|-:|-:|-:|
|One question per request, one at a time|25 ms|~40|—|
|16 questions per request, concurrency 1|6 ms|156.7|93–99.7%|
|16 questions per request, concurrency 4|15 ms|73.1|—|

The per-decision numbers for batched requests are not the board method;
the board times one request with one question.

**Other results**

* On SemIf's 144-decision set with SemIf's evaluator, the 27B INT4
checkpoint scored 0.9424 balanced accuracy. SemIf's published EXL3 run
of the same model family scored 0.9579, and a 4B BF16 baseline scored
0.813. One run per configuration, so the 1.6% gap to EXL3 is
consistent with quantization but doesn't isolate it.
* On JevBench, Qwen3.8-Flash-Next (125B MoE) answered 231/231 on the
public set and scored 0.856 on the hard tier.
* On a separate SemIf eval render, the fitted T was 0.505, taking ECE from
0.073 to 0.016 out-of-fold.

## Lineage

* [**SemIf**](https://github.com/TheoLeeCJ/SemIf-OpenJev) (TheoLeeCJ)
established the decision-native logit readout (`direct-options-v1`) and
its evaluation method. In SemIf's benchmark, direct option logits beat
a native reranker (Qwen3-Reranker-4B) as a general decision baseline
(0.845 vs 0.560 on its TypeSafe subset) and were 5.2× faster than
generating a compact JSON answer. That finding is why `logit` is the
default.
* [**Jev**](https://docs.typesafe.ai) (TypeSafe) defined the typed wire
format: `noul`/`choice`/`score`, questions by id.
* [**SimpleJev**](https://github.com/featherless-ai/simple-jev) (Featherless)
is a small standalone reference server for the same format.
* [**diffgemma**](https://github.com/mmastrac/diffgemma) (mmastrac) originated
the canvas readout ([issue #21](https://github.com/mmastrac/diffgemma/issues/21)):
seed the answer template into the canvas with label slots as noise, read
each question's distribution at its slot, average several noise draws.
* [#57250](https://github.com/vllm-project/vllm/pull/57250) ported that into
vLLM as the `structured_server.py` example.
* [#58429](https://github.com/vllm-project/vllm/pull/58429) added Laya
pooling models and a `--backend laya` path to the same example server.
* **gev** (dglazkov) also serves the Jev API for DiffusionGemma-class models,
using autoregressive single-token logprobs rather than canvas reads.

This repository moves those readouts from a separate fronting server into
vLLM's API server. They then share the engine's batching, prefix cache,
tensor parallelism and quantization support.

## Status

Tested on vLLM v0.30.0:

* `logit`: Qwen3.8-27B (INT4), Qwen3.8-Flash-Next (125B MoE) and
Qwen3.5-0.8B. In September 2026, a sweep over the JevBench public tasks
on this code (share of tasks where the most likely option is the
expected answer; a sanity check, not the official JevBench score):

|Model|easy|original|hard|
|-|-:|-:|-:|
|Qwen3-4B (BF16)|1.000|0.833|0.423|
|Qwen3.8-27B (INT4)|1.000|0.931|0.757|
|Gemma 4 E4B|1.000|0.972|0.540|
|Llama 3.1 8B|0.979|0.764|0.423|
|GPT-OSS 20B|1.000|0.681|0.486|

  All five passed the answer-slot self-check. DeepSeek-R1-Distill 1.5B is
refused by the self-check (a 503 that says why). Ministral-3 3B doesn't
boot on vLLM v0.30.0 (its Pixtral config), so it is untested.
These come from an earlier sweep than [Results](#results), so the 27B's
hard score differs slightly (0.757 vs 0.730).
Calibration results for four of these are under
[Measured results](#measured-results).

* `encoder`: a Laya checkpoint.
* `canvas`: ran end-to-end on diffusiongemma-26B-nvfp4, but that was before
the readout was corrected to one answer slot per question. Its accuracy
still needs to be measured.

See `CHANGELOG.md`.

Porting to vLLM main is the next step; `overlay/README.md` lists the known
differences.

License: Apache-2.0.
