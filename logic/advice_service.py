"""LLM-backed advice generation with schema validation and short-lived cache."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_ADVICE_CACHE: dict[str, tuple[datetime, "AdviceResponse"]] = {}
_DEFAULT_TIMEOUT_MS = 8000
_DEFAULT_TTL_SECONDS = 120


class AdviceConfigError(RuntimeError):
    """Raised when AI configuration is missing."""


class AdviceUpstreamError(RuntimeError):
    """Raised when upstream AI service fails."""


class AdviceRateLimitError(RuntimeError):
    """Raised when upstream AI service is rate-limited."""


class AdviceAction(BaseModel):
    title: str
    reason: str
    priority: str


class AdviceModelMeta(BaseModel):
    provider: str = "openai"
    model: str
    latency_ms: int


class AdviceRequestPosition(BaseModel):
    ticker: str
    name: Optional[str] = None
    weight: float
    expectedReturn: float
    volatility: float


class AdviceAllocationSuggestion(BaseModel):
    bucket: str
    currentWeight: float
    targetWeight: float
    delta: float


class AdviceRequestProfile(BaseModel):
    age: int = Field(ge=18, le=90)
    riskLevel: str
    taxRegion: str
    horizonYears: int = Field(ge=1, le=30)


class AdviceRequestPortfolio(BaseModel):
    totalUsd: float = Field(ge=0)
    expectedReturn: float
    volatility: float
    maxDrawdown: float
    positions: list[AdviceRequestPosition]
    allocationSuggestion: list[AdviceAllocationSuggestion]


class AdviceRequest(BaseModel):
    profile: AdviceRequestProfile
    portfolio: AdviceRequestPortfolio
    locale: str = "zh-TW"


class AdviceResponse(BaseModel):
    summary: str
    risk_level: str
    actions: list[AdviceAction]
    watchouts: list[str]
    disclaimer: str
    model_meta: AdviceModelMeta


@dataclass(frozen=True)
class _OpenAiConfig:
    api_key: str
    model: str
    timeout_seconds: float
    cache_ttl_seconds: int


def _utcnow() -> datetime:
    return datetime.utcnow()


def _load_config() -> _OpenAiConfig:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AdviceConfigError("Missing OPENAI_API_KEY.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    timeout_ms = int(os.getenv("OPENAI_TIMEOUT_MS", str(_DEFAULT_TIMEOUT_MS)))
    cache_ttl = int(os.getenv("ADVICE_CACHE_TTL_SEC", str(_DEFAULT_TTL_SECONDS)))
    return _OpenAiConfig(
        api_key=api_key,
        model=model,
        timeout_seconds=max(1.0, timeout_ms / 1000.0),
        cache_ttl_seconds=max(10, cache_ttl),
    )


def _cache_key(payload: AdviceRequest) -> str:
    serialized = json.dumps(payload.model_dump(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_messages(payload: AdviceRequest) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a cautious portfolio copilot. Output STRICT JSON only with keys: "
                "summary, risk_level, actions, watchouts, disclaimer. "
                "risk_level must be one of low/medium/high. actions max 3 items. "
                "No guaranteed return language. Keep locale based on input locale."
            ),
        },
        {
            "role": "user",
            "content": payload.model_dump_json(),
        },
    ]


def _extract_content(parsed: dict[str, Any]) -> str:
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AdviceUpstreamError("AI response missing choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        joined = "".join(text_parts).strip()
        if joined:
            return joined
    raise AdviceUpstreamError("AI response content is empty.")


def _call_openai(payload: AdviceRequest, config: _OpenAiConfig) -> AdviceResponse:
    body = {
        "model": config.model,
        "temperature": 0.2,
        "messages": _build_messages(payload),
        "response_format": {"type": "json_object"},
    }
    request = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = _utcnow()
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code == 429:
            raise AdviceRateLimitError("OpenAI rate limit reached.") from exc
        raise AdviceUpstreamError(f"OpenAI HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise AdviceUpstreamError(f"OpenAI unreachable: {exc}") from exc

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise AdviceUpstreamError("OpenAI returned malformed JSON.") from exc

    content = _extract_content(parsed)
    try:
        advice_payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AdviceUpstreamError("Model content is not valid JSON.") from exc

    latency_ms = max(0, int((_utcnow() - started).total_seconds() * 1000))
    advice_payload["model_meta"] = {
        "provider": "openai",
        "model": config.model,
        "latency_ms": latency_ms,
    }
    return AdviceResponse.model_validate(advice_payload)


def generate_advice(payload: AdviceRequest) -> AdviceResponse:
    config = _load_config()
    key = _cache_key(payload)
    cached = _ADVICE_CACHE.get(key)
    if cached and _utcnow() < cached[0]:
        logger.info("advice_cache_hit key=%s", key[:8])
        return cached[1]

    advice = _call_openai(payload, config)
    _ADVICE_CACHE[key] = (_utcnow() + timedelta(seconds=config.cache_ttl_seconds), advice)
    logger.info(
        "advice_generated model=%s latency_ms=%s actions=%s",
        advice.model_meta.model,
        advice.model_meta.latency_ms,
        len(advice.actions),
    )
    return advice
