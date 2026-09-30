# Overlay layout

Files here mirror their paths inside the `vllm/` package of vLLM v0.30.0.
The only one is the new `vllm/entrypoints/generate/decisions/` package; no
upstream file has that path. Upstream files are changed only by the wiring
patch below, never shipped as copies.

## The wiring patch

`api-router-wire-v0.30.patch` (repo root) changes three upstream files:

| File | Change |
|---|---|
| `entrypoints/generate/api_router.py` | Registers `/v1/decisions` and `/v1/systemone`; builds `ServingDecisions` and `ServingSystemOne`; loads backend plugins; reads `VLLM_TYPED_DECISIONS_BACKEND` |
| `entrypoints/launchers/api_server/routers.py` | Registers the generate routes when the flag is set, even if the model isn't a generate model |
| `entrypoints/launchers/api_server/app_state.py` | Same condition for serving-state initialization |

The two launcher changes are needed for the `encoder` backend: Laya is a
pooling model, and without them vLLM never reaches the generate routes. All
three changes are gated on `VLLM_ENABLE_TYPED_DECISIONS`.

Paths in the patch start with `vllm/`. Apply it with `patch -p1` from the
directory that **contains** the `vllm/` package (`site-packages` or
`dist-packages`), not from inside it.

## Import fallback

The modules import `vllm.entrypoints.serve.engine.protocol` and fall back to
`vllm.entrypoints.openai.engine.protocol`. Stock v0.30.0 uses the first
path; some forks moved the module to the second.

## Porting to vLLM main

Known differences from v0.30.0, as of 2026-09-25:

- `ErrorResponse`, `OpenAIBaseModel` and `BaseServing` moved within
  `entrypoints/serve/engine/` after 0.30.0. The import fallback covers one
  direction only; check both.
- The `register_*_api_router(app)` pattern in
  `entrypoints/generate/api_router.py` is unchanged.
- `patches/00572` and `patches/00584` need rebasing. The 0.30 rebase of
  00572 had to follow two renames:
  - `_resolve_allow_missing_mm_embeddings` → `_resolve_mm_embedding_inputs`
  - flashinfer's `use_dedicated_xqa` → `use_xqa`

## Engine notes

- **Speculative decoding.** The readout logs a one-time warning instead of
  refusing. Restricted-logprob gathers under speculative decoding have had
  shape bugs ([#42592](https://github.com/vllm-project/vllm/issues/42592)
  and related). Compare logits with and without speculative decoding on your
  stack; if they differ, change the warning to a refusal.
- **Prefix caching.** Multi-question requests rely on it to prefill the
  shared state once. It is on by default in vLLM V1; don't disable it.
