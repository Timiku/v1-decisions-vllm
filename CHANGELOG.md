# Changelog

## 0.1.0 — first public release

Typed decisions for vLLM v0.30.0, as an overlay on the API server.

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
