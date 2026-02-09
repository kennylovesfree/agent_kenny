"""Service helpers for resolving Taiwan stocks and computing annual return."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

try:
    from .market_data_client import FinMindApiError, FinMindClient, FinMindUpstreamError, PricePoint
except ImportError:  # pragma: no cover - support direct script-style imports
    from market_data_client import FinMindApiError, FinMindClient, FinMindUpstreamError, PricePoint


class StockQueryError(RuntimeError):
    """Raised when user input does not pass strict validation rules."""

    def __init__(self, error_code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details


class UpstreamServiceError(RuntimeError):
    """Raised when FinMind upstream fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class ResolvedStock:
    stock_id: str
    stock_name: str


@dataclass(frozen=True)
class AnnualReturnResult:
    price_date_latest: str
    price_latest: float
    price_date_base: str
    price_base: float
    annual_return: float


_CACHE_TTL_SECONDS = 3600
_STOCK_INFO_CACHE: dict[str, object] = {"expires_at": datetime.min, "rows": []}
_ANNUAL_RETURN_CACHE: dict[tuple[str, str], tuple[datetime, AnnualReturnResult]] = {}


def _utcnow() -> datetime:
    return datetime.utcnow()


def normalize_query(query: str) -> str:
    cleaned = query.strip().upper()
    if cleaned.endswith(".TW"):
        cleaned = cleaned[:-3]
    return cleaned


def _get_stock_universe_cached(client: FinMindClient) -> list[dict[str, str]]:
    now = _utcnow()
    expires_at = _STOCK_INFO_CACHE["expires_at"]
    rows = _STOCK_INFO_CACHE["rows"]
    if isinstance(expires_at, datetime) and now < expires_at and isinstance(rows, list) and rows:
        return rows

    universe = client.get_taiwan_stock_info()
    _STOCK_INFO_CACHE["rows"] = universe
    _STOCK_INFO_CACHE["expires_at"] = now + timedelta(seconds=_CACHE_TTL_SECONDS)
    return universe


def resolve_stock(query: str, client: FinMindClient) -> ResolvedStock:
    raw_cleaned = query.strip()
    cleaned = normalize_query(query)
    if not cleaned:
        raise StockQueryError("INVALID_QUERY", "query 不可為空白。")

    try:
        universe = _get_stock_universe_cached(client)
    except FinMindUpstreamError as exc:
        raise UpstreamServiceError(str(exc)) from exc
    except FinMindApiError as exc:
        raise UpstreamServiceError(str(exc)) from exc

    for row in universe:
        stock_id = str(row["stock_id"]).strip().upper()
        if stock_id == cleaned:
            return ResolvedStock(stock_id=row["stock_id"], stock_name=row["stock_name"])

    matches = [row for row in universe if row["stock_name"] == raw_cleaned]
    dedup_by_stock_id: dict[str, dict[str, str]] = {m["stock_id"]: m for m in matches}
    unique_matches = list(dedup_by_stock_id.values())
    if not unique_matches:
        raise StockQueryError("STOCK_NOT_FOUND", "查無此台股代碼或名稱。", {"query": cleaned})
    if len(unique_matches) > 1:
        raise StockQueryError(
            "AMBIGUOUS_NAME",
            "名稱對應到多筆股票，請改用台股代碼。",
            {"query": cleaned, "matched_stock_ids": [m["stock_id"] for m in unique_matches]},
        )
    return ResolvedStock(stock_id=unique_matches[0]["stock_id"], stock_name=unique_matches[0]["stock_name"])


def compute_annual_return(stock_id: str, client: FinMindClient) -> AnnualReturnResult:
    today = date.today()
    cache_key = (stock_id, today.isoformat())
    cached = _ANNUAL_RETURN_CACHE.get(cache_key)
    if cached and _utcnow() < cached[0]:
        return cached[1]

    start_date = (today - timedelta(days=400)).isoformat()
    end_date = today.isoformat()
    target_base_date = today - timedelta(days=365)

    try:
        history = client.get_price_history(
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date,
            dataset="TaiwanStockPrice",
        )
    except FinMindUpstreamError as exc:
        raise UpstreamServiceError(str(exc)) from exc
    except FinMindApiError as exc:
        raise UpstreamServiceError(str(exc)) from exc

    if not history:
        raise StockQueryError("NO_PRICE_DATA", "查無可用價格資料。", {"stock_id": stock_id})

    latest = history[-1]
    base = _find_base_price(history, target_base_date)
    if base is None:
        raise StockQueryError(
            "NO_PRICE_DATA",
            "找不到距今滿一年之前的基準收盤價。",
            {"stock_id": stock_id, "required_before": target_base_date.isoformat()},
        )
    if base.close <= 0:
        raise StockQueryError("NO_PRICE_DATA", "基準收盤價異常，無法計算報酬率。", {"stock_id": stock_id})

    adjusted_base = _adjust_base_for_split(client, stock_id, base, latest)
    if adjusted_base <= 0:
        raise StockQueryError("NO_PRICE_DATA", "調整後基準收盤價異常，無法計算報酬率。", {"stock_id": stock_id})

    annual_return = (latest.close / adjusted_base) - 1.0
    result = AnnualReturnResult(
        price_date_latest=latest.date,
        price_latest=latest.close,
        price_date_base=base.date,
        price_base=adjusted_base,
        annual_return=annual_return,
    )
    _ANNUAL_RETURN_CACHE[cache_key] = (_utcnow() + timedelta(seconds=_CACHE_TTL_SECONDS), result)
    return result


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


def _adjust_base_for_split(client: FinMindClient, stock_id: str, base: PricePoint, latest: PricePoint) -> float:
    try:
        base_date = date.fromisoformat(base.date)
        latest_date = date.fromisoformat(latest.date)
    except ValueError:
        return base.close
    if latest_date <= base_date:
        return base.close

    try:
        events = client.get_split_events(
            data_id=stock_id,
            start_date=base_date.isoformat(),
            end_date=latest_date.isoformat(),
        )
    except (FinMindApiError, FinMindUpstreamError):
        return base.close

    factor = 1.0
    for event in events:
        try:
            event_date = date.fromisoformat(event.date)
        except ValueError:
            continue
        if event_date <= base_date or event_date > latest_date:
            continue
        if event.before_price <= 0 or event.after_price <= 0:
            continue
        factor *= event.after_price / event.before_price
    return base.close * factor
