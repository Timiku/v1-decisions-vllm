# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router for the /v1/decisions typed-decision endpoint."""
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.generate.decisions.backends import (
    validation_error)
from vllm.entrypoints.generate.decisions.protocol import DecisionsRequest
from vllm.entrypoints.generate.decisions.serving import (
    ServingDecisions,
)
try:
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse
except ImportError:  # locked fork moved engine protocol to openai
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.logger import init_logger

router = APIRouter()

logger = init_logger(__name__)


def decisions(request: Request) -> ServingDecisions | None:
    return getattr(request.app.state, "serving_decisions", None)


@router.post(
    "/v1/decisions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.UNPROCESSABLE_ENTITY.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_decision(raw_request: Request):
    handler = decisions(raw_request)
    if handler is None:
        raise NotImplementedError(
            "The model does not support the Decisions API")

    raw_body = await raw_request.json()
    try:
        request = DecisionsRequest(**raw_body)
    except Exception as e:
        return JSONResponse(content=validation_error(str(e)).model_dump(),
                            status_code=422)
    result = await handler.create_decisions(request, raw_request)

    if isinstance(result, ErrorResponse):
        # Retry-After only helps while the server is STARTING; a
        # self-check refusal is permanent, retrying changes nothing.
        headers = ({"Retry-After": "5"}
                   if result.error.code == 503 and
                   "decision server is starting" in result.error.message
                   else None)
        return JSONResponse(content=result.model_dump(),
                            status_code=result.error.code,
                            headers=headers)
    return JSONResponse(content=result)


def register_decisions_api_router(app: FastAPI):
    app.include_router(router)
