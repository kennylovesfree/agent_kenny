"""Service helpers for resolving US stocks and computing annual return."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from .market_data_client import PricePoint
    from .tw_stock_return_service import StockQueryError, UpstreamServiceError
except ImportError:  # pragma: no cover - support direct script-style imports
    from market_data_client import PricePoint
    from tw_stock_return_service import StockQueryError, UpstreamServiceError


@dataclass(frozen=True)
class AnnualReturnResult:
    price_date_latest: str
    price_latest: float
    price_date_base: str
    price_base: float
    annual_return: float


_CACHE_TTL_SECONDS = 3600
_ANNUAL_RETURN_CACHE: dict[tuple[str, str], tuple[datetime, AnnualReturnResult]] = {}
_RETURN_SMOOTHING_WINDOWS: tuple[tuple[int, float], ...] = ((1, 0.6), (3, 0.3), (5, 0.1))
_MAX_TICKER_LENGTH = 12
_VALID_TICKER_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")


def _utcnow() -> datetime:
    return datetime.utcnow()


def normalize_query(query: str) -> str:
    cleaned = query.strip().upper()
    if cleaned.endswith(".US"):
        cleaned = cleaned[:-3]
    return cleaned


def _validate_ticker(ticker: str) -> None:
    if not ticker:
        raise StockQueryError("INVALID_QUERY", "query 不可為空白。")
    if len(ticker) > _MAX_TICKER_LENGTH:
        raise StockQueryError("INVALID_QUERY", "美股代碼長度不正確。", {"query": ticker})
    if any(ch not in _VALID_TICKER_CHARS for ch in ticker):
        raise StockQueryError("INVALID_QUERY", "美股代碼格式不正確。", {"query": ticker})
    if ticker[0] in {".", "-"}:
        raise StockQueryError("INVALID_QUERY", "美股代碼格式不正確。", {"query": ticker})


def fetch_us_price_history(ticker: str) -> list[PricePoint]:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    request = Request(url, headers={"User-Agent": "rebalance-copilot/1.0"})
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
    except URLError as exc:
        raise UpstreamServiceError(f"Failed to reach US market data source: {exc}") from exc

    if not body.strip() or "No data" in body:
        return []

    reader = csv.DictReader(io.StringIO(body))
    parsed: list[PricePoint] = []
    for row in reader:
        row_date = str(row.get("Date", "")).strip()
        close_raw = row.get("Close")
        if not row_date or close_raw in (None, ""):
            continue
        try:
            close = float(close_raw)
            date.fromisoformat(row_date)
        except (TypeError, ValueError):
            continue
        if close <= 0:
            continue
        parsed.append(PricePoint(data_id=ticker, date=row_date, close=close))

    return sorted(parsed, key=lambda r: r.date)


def _find_base_price(history: list[PricePoint], target_date: date) -> PricePoint | None:
    candidates: list[PricePoint] = []
    for point in history:
        try:
            point_date = date.fromisoformat(point.date)
        except ValueError:
            continue
        if point_date <= target_date:
            candidates.append(point)
    if not candidates:
        return None
    return candidates[-1]


def _compute_smoothed_annual_return(history: list[PricePoint], latest: PricePoint, fallback_base: float) -> float:
    latest_date = date.fromisoformat(latest.date)
    weighted_sum = 0.0
    weight_sum = 0.0

    for years, weight in _RETURN_SMOOTHING_WINDOWS:
        target_date = latest_date - timedelta(days=365 * years)
        base_point = _find_base_price(history, target_date)
        if base_point is None or base_point.close <= 0:
            continue
        horizon_return = (latest.close / base_point.close) ** (1.0 / years) - 1.0
        weighted_sum += horizon_return * weight
        weight_sum += weight

    if weight_sum > 0:
        return weighted_sum / weight_sum
    return (latest.close / fallback_base) - 1.0


def compute_annual_return(query: str) -> tuple[str, AnnualReturnResult]:
    ticker = normalize_query(query)
    _validate_ticker(ticker)

    today = date.today()
    cache_key = (ticker, today.isoformat())
    cached = _ANNUAL_RETURN_CACHE.get(cache_key)
    if cached and _utcnow() < cached[0]:
        return ticker, cached[1]

    history = fetch_us_price_history(ticker)
    if not history:
        raise StockQueryError("STOCK_NOT_FOUND", "查無此美股代碼。", {"query": ticker})

    latest = history[-1]
    target_base_date = today - timedelta(days=365)
    base = _find_base_price(history, target_base_date)
    if base is None:
        raise StockQueryError(
            "NO_PRICE_DATA",
            "找不到距今滿一年之前的基準收盤價。",
            {"ticker": ticker, "required_before": target_base_date.isoformat()},
        )
    if base.close <= 0:
        raise StockQueryError("NO_PRICE_DATA", "基準收盤價異常，無法計算報酬率。", {"ticker": ticker})

    annual_return = _compute_smoothed_annual_return(history, latest, base.close)
    result = AnnualReturnResult(
        price_date_latest=latest.date,
        price_latest=latest.close,
        price_date_base=base.date,
        price_base=base.close,
        annual_return=annual_return,
    )
    _ANNUAL_RETURN_CACHE[cache_key] = (_utcnow() + timedelta(seconds=_CACHE_TTL_SECONDS), result)
    return ticker, result
