# Writing a decision backend

A backend turns one `CompiledQuestion` (`protocol.py`) into a score for each
of the question's option ids. Three ship in this package: `logit`, `encoder`
and `canvas`. `logit_backend.py` is the smallest complete one; read it first.

A backend never sees the request's question type or the answer format: the
question type has already reduced the question to a closed option set, and
the server turns your scores into the answer. So a new kind of decision
model only has to produce one score per option.

## Interface

```python
from pydantic import BaseModel, ConfigDict

from vllm.entrypoints.generate.decisions.backends import (
    BackendError, BackendResult, cached_tokens, register_backend,
    restricted_softmax)


class MyOptions(BaseModel):              # optional: per-request settings
    model_config = ConfigDict(extra="forbid")
    threshold: float = 0.5


class MyBackend:
    name = "my-backend"                  # selection key and extra.backend.name
    architectures = ("MyModelForDecision",)   # served by default; may be ()
    options_model = MyOptions            # or None: the backend takes no options
    calibrate_by_default = False         # True: startup calibration runs on
                                         # JevBench unless the operator says off

    def __init__(self, host, **kwargs):  # kwargs: from VLLM_TYPED_DECISIONS_BACKEND
        self.host = host

    async def read(self, question, request_id):
        opts = question.backend_options or MyOptions()   # already validated
        prompt = await self.host.render(question)
        found, result = await self.host.restricted_read(
            prompt.engine_input, prompt.slot_ids, request_id)
        scores = {o.id: found[t]
                  for o, t in zip(question.options, prompt.slot_ids)}
        return BackendResult(
            option_logits=scores,
            probabilities=restricted_softmax(scores),
            forward_passes=1,
            meta={"input_tokens": prompt.input_tokens,
                  "cached_input_tokens": cached_tokens(result),
                  "readout": "my-read",
                  "threshold": opts.threshold})


register_backend("my-backend", MyBackend)
```

## What the host gives you (`host.py`)

Use nothing else from the server; this surface is what stays stable.

| Member | What it is |
|---|---|
| `host.model_config`, `host.architectures`, `host.tokenizer`, `host.limits` | Facts about the served model and the server's limits |
| `await host.render(question, labels=None)` | The standard decision prompt (chat template, thinking off, markers checked). Returns `engine_input`, `input_tokens`, `slot_ids` (each option's marker token) and `prompt_ids`. Raises `BackendError` when the prompt can't be read reliably |
| `await host.render_joint(questions)` | One prompt asking every question (same state), answered as `N: letter` lines. Returns `engine_input`, `input_tokens`, `prompt_ids`. For `read_many` |
| `await host.restricted_read(engine_input, token_ids, request_id)` | One pass returning the next-token logprob of exactly those token ids, plus the `RequestOutput` |
| `await host.generate(engine_input, sampling_params, request_id)` | One engine request with your own sampling params; the final `RequestOutput` |
| `await host.pool(prompt_ids, pooling_params, request_id)` | One pooling request (encoder models); the final output |

You don't have to use `render`: a model trained on its own format builds
its own prompt (the encoder does, for Laya).

## Rules

- **Cover every option.** Return a score for every id in
  `question.options`. If one can't be read, raise
  `BackendError(message, option_ids)`: the question is reported in
  `partial_failures` and the request's other questions still answer.
- **Return raw scores on a log scale** (logprobs, logits, or the log of
  averaged probabilities). Don't apply a temperature: the server applies
  the calibration temperature once, the same way for every backend:
  `softmax(option_logits / T)`. Startup calibration works on any backend
  through the same `read`: set `calibrate_by_default = True` to calibrate
  unless the operator turns it off, or leave it `False` so it runs only
  when `VLLM_TYPED_DECISIONS_CALIBRATION` is `on` or a file path. Only
  `logit` sets it today.
- **Report what you did in `meta`.** It appears under the answer's
  `extra.backend`, next to `name` and `option_logits`. These standard keys
  are lifted into `extra.audit` instead, with the same meaning on every
  backend:
  - `input_tokens` (required): prompt tokens over every engine request the
    question took;
  - `cached_input_tokens`: of those, how many came from the prefix cache
    (`cached_tokens(result)`; `add_known` sums several reads);
  - `option_mass`: only if your scores are full-vocabulary logprobs of the
    option tokens (`option_mass(host, scores)`);
  - `readout`, `label_layout`: names for how you read and labelled.
- **Per-option facts** in `meta` (a dict keyed by option ids) are dropped
  automatically when a request asks for `extra: "basic"`.
- **Count passes honestly.** `forward_passes` is the number of engine
  requests one read took.
- **Errors.** Raise `BackendError` for anything the caller can act on.
  Other exceptions are logged and reported the same way (a 500 if every
  question failed). `asyncio.CancelledError` must propagate.

## Reading several questions at once (optional)

A request's questions share one state. A backend that can answer several
of them from one engine request adds:

```python
    async def read_many(self, questions, request_id):
        # -> one entry per question, in order:
        #    BackendResult, BackendError, or None ("read this one with read")
```

The server calls it once per request that has more than one question,
then calls `read` for every question it left as `None`. So `read_many`
can take the questions it handles well and decline the rest, or decline
all of them. Each `BackendResult` is one question's read, under the same
rules as `read`'s; split the shared request's `input_tokens` and
`cached_input_tokens` across the questions so the request's usage adds up.
If `read_many` raises, the server logs it and reads every question with
`read`. `canvas` implements it (the joint canvas read); `logit` and
`encoder` don't: their reads are one question per sequence.

## Registration and selection

- **Direct.** `register_backend(name, cls)` at import time. Registering a
  name twice raises; don't use a built-in name (`logit`, `encoder`,
  `canvas`).
- **Entry point.** Declare an entry point in the `vllm.decision_backends`
  group that resolves to a function calling `register_backend`. The server
  calls every such function at startup. A plugin that fails to load is
  logged and skipped.

At startup the server uses `VLLM_TYPED_DECISIONS_BACKEND` if set,
otherwise the backend whose `architectures` include the served model's
architecture (a plugin's claim wins over a built-in's), otherwise `logit`:

```
VLLM_TYPED_DECISIONS_BACKEND=my-backend:threshold=0.7,verbose=true
```

The name ends at the first `:`; arguments are `key=value` pairs separated
by `,`, passed to `__init__`. A request can pick a loaded backend with
`"backend": "my-backend"` (built once, with default arguments) and send
`"backend_options": {...}`, validated against your `options_model`. A
backend without an `options_model` refuses any options.

## Question types

Question types are the other plug-in point (`question_types.py`): a type
defines its JSON shape, the closed option set it compiles to, and its
answer fields. A backend works with every type unless it can't (the Laya
encoder only knows noul, choice and score, and refuses others per
question).
