# Changelog

## Unreleased

**Breaking (0.2.0): `/v1/decisions` now takes and returns OpenAI's
Decisions format.** Clients of the old `/v1/decisions` body must move to
OpenAI's request, or to `/v1/systemone`, which keeps the Jev format.

- `/v1/decisions` request: `model`, `input` (a string, or user messages
  of `input_text` parts), `questions[]` (`predicate`, `choice` with string
  or boolean values, `score` with 2-10 levels), `safety_identifier`
  (accepted, ignored). Unknown fields, images and an explicit
  `name: null` are refused with HTTP 400 (was 422), as in vLLM's draft
  #60465. Input is joined as there: parts with a newline, messages with
  a blank line.
- `/v1/decisions` response: `model`, `answers` in question order,
  OpenAI's `usage` shape; `id` and `created` move to the top-level
  `extra`. A failed question is a `refusal` answer with `extra.error`
  (was `partial_failures`); if every question fails, the request fails.
- All settings OpenAI has no field for move under one `extra` block:
  `calibration_temperature`, `backend`, `backend_options`, `seed`, and
  `detail` (was the request's `extra`: full | basic | none, or per
  block; default `full`). Answers carry their audit and backend blocks
  in `extra`, as before.
- Removed: the one-question shorthand, the question map, `state`, and
  the top-level settings on `/v1/decisions`.
- `/v1/systemone` is unchanged for Jev clients and gains the same
  `extra` request block. `calibration_temperature`, `seed` and
  `backend_options` are refused at its top level (send them under
  `extra`); `backend` is accepted at the top level or under `extra`,
  not both.
- Each OpenAI question renders as the equivalent Jev question, byte for
  byte, so the saved temperature calibration still applies.
- Choice questions take up to 255 choices (the server's limit), not
  #60465's 26.
- `tools/capture_logits.py` posts to `/v1/systemone`;
  `tools/test_decisions.py` tests the OpenAI wire, and the OpenAI SDK
  parse when `openai` is installed.
- Internal: `DecisionsRequest` is now `DecisionsQuery`, and
  `ServingDecisions.create_decisions` is now `answer_query`;
  `create_decisions` takes the OpenAI request.

- Logit backend: `logprobs` option, `top-k` (default) or `exact`. `top-k`
  reads the markers from the engine's plain top-k window
  (k = `--max-logprobs`, no `logprob_token_ids`); a marker outside it
  scores the window's lowest logprob and is listed in `meta.floored`.
  No per-label limit, so `auto` reaches wide-direct capacity on a stock
  server, and it works under speculative decoding (vLLM issue 42592
  makes `exact` reads come back incomplete under MTP load). Switch back
  via `VLLM_TYPED_DECISIONS_BACKEND=logit:logprobs=exact`, per request via
  `backend_options.logprobs`.
- `auto` with `exact`: past the engine's read limit, the wide-direct
  read switches to `top-k` instead of falling back to two-stage. On a
  stock 27B server (window 20) that was more accurate at 64-255 options
  (95/90/84 vs 85/83/62 of 100) and about 10x faster. Two-stage is now
  only the last resort, past the tokenizer's wide-direct capacity.
- Fix: the degraded fallback scored a marker outside the window as
  -inf, and the response failed JSON encoding (HTTP 400 "Out of range
  float values"). It now takes the window's lowest logprob, and any
  non-finite value in `extra.backend` is sent as `null`.

## 0.1.0 — first public release

Typed decisions for vLLM v0.30.0, as an overlay on the API server.

- A restricted read whose reported logprobs lack requested ids is
  retried (`DECISIONS_READ_RETRIES`, default 2, with a
  `DECISIONS_READ_RETRY_BACKOFF_S` backoff): the engine's gather can
  transiently drop requested ids under chunked-prefill contention.
  A logit `direct`/`wide-direct` read that exhausts the retries falls
  back to one degraded generative read scored from the top-k window
  (`meta.degraded: true`, `readout: "direct-degraded"`); other reads
  fail per question as before.

- Backends may add an optional `read_many(questions, request_id)` that
  answers several questions of one request from shared engine requests;
  the server calls it once per multi-question request and reads whatever
  it leaves with `read`. The host gains `render_joint(questions)`.
- `canvas` implements it: the joint canvas read from the upstream
  structured-diffusion example. One prompt asks every question, the
  canvas holds `1: A`, `2: A`, ... with noise at each letter, and one
  read-only denoise scores all of them (`extra.audit.readout: "joint"`).
  Questions past the canvas width, or with max_steps > 1, are read one by
  one as before. Accuracy against per-question reads is not yet measured.

- `POST /v1/decisions`: questions by id with pluggable types (`noul`,
  `choice`, `score` built in), per-request temperature, backend choice
  and options, seed, a one-question shorthand, and an `extra` block with
  raw option logits and an audit record. Partial failures are reported
  per question.
- `POST /v1/systemone`: the Jev request and response format, answered
  through `/v1/decisions` and trimmed to Jev's fields.
- Backends: `logit` (any generative checkpoint; direct, wide-direct and
  two-stage reads, chosen by `auto` within the engine's read limits),
  `encoder` (Laya, needs `patches/00584`) and `canvas` (DiffusionGemma,
  needs `patches/00572`). Backends and question types are plugins.
- Startup: an answer-slot self-check (logit) and a temperature fit on
  the bundled JevBench public set or the operator's own questions, with
  saved results reused across restarts. Decision requests return 503
  with `Retry-After` until both finish.
- Optional `patches/max-logprob-token-ids.patch` raises vLLM's 128
  token-id cap so one-pass reads cover up to 255 options.

Known limits: see the README's Calibration section and Read limits.
Accuracy of the `canvas` backend is not yet measured.
