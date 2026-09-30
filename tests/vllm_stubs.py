"""Offline stubs for the vLLM imports the decisions overlay performs.

Installed by conftest when real vLLM is absent, so the package imports
and the request/compile/readout logic runs under plain pytest.
"""
import sys
import types
import uuid
from http import HTTPStatus


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # allow "vllm.entrypoints.generate" style imports
    sys.modules[name] = mod
    return mod


class _Logger:
    def __getattr__(self, item):
        return lambda *a, **k: None

    def exception(self, *a, **k):
        pass


def init_logger(name):
    return _Logger()


def random_uuid() -> str:
    return uuid.uuid4().hex


class OpenAIBaseModel:
    """Minimal pydantic BaseModel with extra=forbid semantics matching
    vLLM's model (we allow extra fields so the systemone override test
    exercises the model_validator, not the meta model)."""
    import pydantic

    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


def install() -> None:
    if "vllm" in sys.modules and getattr(
            sys.modules["vllm"], "__decision_tests_real__", False):
        return

    vllm = _module("vllm")
    vllm.__decision_tests_real__ = False

    logger_mod = _module("vllm.logger")
    logger_mod.init_logger = init_logger

    utils_mod = _module("vllm.utils")
    utils_mod.random_uuid = random_uuid

    # vllm entrypoints protocol: OpenAIBaseModel/ErrorResponse/ErrorInfo
    proto = _module("vllm.entrypoints")
    _module("vllm.entrypoints.generate")
    _module("vllm.entrypoints.serve")
    _module("vllm.entrypoints.serve.engine")
    serve_proto = _module("vllm.entrypoints.serve.engine.protocol")

    import pydantic

    class _OpenAIBaseModel(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow")

    class ErrorInfo(_OpenAIBaseModel):
        message: str
        type: str
        param: str | None = None
        code: int

    class ErrorResponse(_OpenAIBaseModel):
        error: ErrorInfo

    serve_proto.OpenAIBaseModel = _OpenAIBaseModel
    serve_proto.ErrorResponse = ErrorResponse
    serve_proto.ErrorInfo = ErrorInfo

    # serving.py also imports chat protocol / models serving / BaseServing /
    # renderer bits; stub enough for import.
    _module("vllm.entrypoints.openai")
    chat_proto = _module("vllm.entrypoints.openai.chat_completion.protocol")

    class ChatCompletionRequest(_OpenAIBaseModel):
        model_config = pydantic.ConfigDict(extra="allow")

        def build_chat_params(self, *a, **k):
            # real vLLM passes reasoning_effort into the template kwargs
            return _ChatParams(getattr(self, "reasoning_effort", None))

    chat_proto.ChatCompletionRequest = ChatCompletionRequest

    class _ChatParams:
        def __init__(self, reasoning_effort=None):
            self.reasoning_effort = reasoning_effort

        def with_defaults(self, kw):
            return self

    chat_proto.ChatParams = _ChatParams
    serve_proto.ChatCompletionRequest = ChatCompletionRequest

    models_serving = _module("vllm.entrypoints.openai.models.serving")

    class OpenAIServingModels:
        def model_name(self, _):
            return "test-model"

    models_serving.OpenAIServingModels = OpenAIServingModels

    _module("vllm.entrypoints.serve.utils")
    _module("vllm.entrypoints.serve.utils.request_logger")

    class RequestLogger:
        pass

    sys.modules["vllm.entrypoints.serve.utils.request_logger"].RequestLogger = RequestLogger

    serving_mod = _module("vllm.entrypoints.serve.engine.serving")

    class BaseServing:
        def __init__(self, models=None, model_config=None, *a, **k):
            self.models = models
            self.model_config = model_config

        def create_error_response(self, message, status_code=HTTPStatus.BAD_REQUEST):
            # real vLLM also accepts an Exception (mapped to a status by
            # type); the stub keeps the message and maps plain ones to 500
            if isinstance(message, Exception):
                message, status_code = f"{message}", HTTPStatus.INTERNAL_SERVER_ERROR
            # strict like real vLLM: a plain int is a bug at the call site
            # (real vLLM does status_code.value and raises AttributeError)
            if not isinstance(status_code, HTTPStatus):
                raise TypeError(
                    f"status_code must be http.HTTPStatus, got {status_code!r}")
            return ErrorResponse(error=ErrorInfo(
                message=message, type="invalid_request_error",
                param=None, code=status_code.value))

        def _log_inputs(self, *args, **kwargs):
            pass

        def _base_request_id(self, raw_request, model_name):
            from vllm.utils import random_uuid
            return random_uuid()[:8]

    serving_mod.BaseServing = BaseServing

    api_utils = _module("vllm.entrypoints.serve.utils.api_utils")
    api_utils.with_cancellation = lambda **k: (lambda f: f)
    api_utils.check_request_id = lambda request_id: None

    # sampling / pooling / inputs stubs for backends
    sp = _module("vllm.sampling_params")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    sp.SamplingParams = SamplingParams
    sp.MAX_LOGPROB_TOKEN_IDS = 600

    pp = _module("vllm.pooling_params")

    class PoolingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    pp.PoolingParams = PoolingParams

    vi = _module("vllm.inputs")
    vi.TokensInput = dict
    vi.tokens_input = lambda ids, *a, **k: {"prompt_token_ids": list(ids)}

    # fastapi is a real dependency of the overlay routers; only needed if
    # routers are imported. Provide nothing - tests import the package
    # internals, not the routers.

    # renderers stub: serving.py imports ChatParams/TokenizeParams
    rend = _module("vllm.renderers")
    rend_params = _module("vllm.renderers.params")

    class ChatParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class TokenizeParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    rend_params.ChatParams = ChatParams
    rend_params.TokenizeParams = TokenizeParams

    # async engine dead error for the concurrency test
    _module("vllm")
    _module("vllm.v1")
    v1engine = _module("vllm.v1.engine")
    exceptions_mod = _module("vllm.v1.engine.exceptions")

    class EngineDeadError(Exception):
        pass

    exceptions_mod.EngineDeadError = EngineDeadError
    exceptions_mod.EngineGenerateError = Exception
    vllm.AsyncEngineDeadError = EngineDeadError  # legacy alias
