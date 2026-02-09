"""HTTP API for querying annual return of Taiwan stocks."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from .market_data_client import FinMindClient
    from .advice_service import (
        AdviceConfigError,
        AdviceRateLimitError,
        AdviceRequest,
        AdviceUpstreamError,
        generate_advice,
    )
    from .tw_stock_return_service import (
        StockQueryError,
        UpstreamServiceError,
        compute_annual_return,
        resolve_stock,
    )
except ImportError:  # pragma: no cover - support direct script-style imports
    from market_data_client import FinMindClient
    from advice_service import (
        AdviceConfigError,
        AdviceRateLimitError,
        AdviceRequest,
        AdviceUpstreamError,
        generate_advice,
    )
    from tw_stock_return_service import (
        StockQueryError,
        UpstreamServiceError,
        compute_annual_return,
        resolve_stock,
    )


class AnnualReturnRequest(BaseModel):
    query: str


class ApiError(RuntimeError):
    def __init__(self, *, status_code: int, error_code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.details = details


logger = logging.getLogger(__name__)
app = FastAPI(title="TW Stock Annual Return API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ApiError)
async def api_error_handler(_, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "INVALID_QUERY",
            "message": "請提供有效的 query 字串。",
            "details": {"errors": exc.errors()},
        },
    )


@app.post("/api/v1/tw-stock/annual-return")
def get_annual_return(payload: AnnualReturnRequest) -> dict:
    client = FinMindClient.from_env()
    if client is None:
        raise ApiError(
            status_code=500,
            error_code="CONFIG_ERROR",
            message="缺少 FINMIND_API_TOKEN，無法查詢資料。",
            details=None,
        )

    try:
        resolved = resolve_stock(payload.query, client)
        result = compute_annual_return(resolved.stock_id, client)
    except StockQueryError as exc:
        raise ApiError(
            status_code=422,
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
        ) from exc
    except UpstreamServiceError as exc:
        raise ApiError(
            status_code=502,
            error_code="UPSTREAM_ERROR",
            message=exc.message,
            details=None,
        ) from exc

    logger.info("stock_resolve_success stock_id=%s query=%s", resolved.stock_id, payload.query)

    return {
        "query": payload.query,
        "resolved_stock_id": resolved.stock_id,
        "resolved_stock_name": resolved.stock_name,
        "price_date_latest": result.price_date_latest,
        "price_latest": result.price_latest,
        "price_date_base": result.price_date_base,
        "price_base": result.price_base,
        "annual_return": result.annual_return,
    }


@app.post("/api/v1/advice/generate")
def post_generate_advice(payload: AdviceRequest) -> dict:
    try:
        advice = generate_advice(payload)
    except AdviceConfigError as exc:
        raise ApiError(
            status_code=503,
            error_code="AI_CONFIG_ERROR",
            message=str(exc),
            details=None,
        ) from exc
    except AdviceRateLimitError as exc:
        raise ApiError(
            status_code=429,
            error_code="AI_RATE_LIMITED",
            message=str(exc),
            details=None,
        ) from exc
    except AdviceUpstreamError as exc:
        raise ApiError(
            status_code=502,
            error_code="AI_UPSTREAM_ERROR",
            message=str(exc),
            details=None,
        ) from exc

    logger.info(
        "advice_success risk=%s model=%s latency_ms=%s",
        advice.risk_level,
        advice.model_meta.model,
        advice.model_meta.latency_ms,
    )
    return advice.model_dump()
