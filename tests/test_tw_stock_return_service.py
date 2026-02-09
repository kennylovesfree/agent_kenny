from __future__ import annotations

import unittest
from dataclasses import dataclass

from logic.market_data_client import PricePoint
from logic import tw_stock_return_service as service_module
from logic.tw_stock_return_service import StockQueryError, compute_annual_return, resolve_stock


@dataclass
class FakeClient:
    stock_info: list[dict[str, str]]
    history: list[PricePoint]
    split_events: list[object] | None = None

    def get_taiwan_stock_info(self) -> list[dict[str, str]]:
        return self.stock_info

    def get_price_history(self, **_) -> list[PricePoint]:
        return self.history

    def get_split_events(self, **_) -> list[object]:
        return self.split_events or []


class TwStockReturnServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        service_module._STOCK_INFO_CACHE["rows"] = []
        service_module._STOCK_INFO_CACHE["expires_at"] = service_module.datetime.min
        service_module._ANNUAL_RETURN_CACHE.clear()

    def test_resolve_stock_by_id_success(self) -> None:
        client = FakeClient(stock_info=[{"stock_id": "2330", "stock_name": "台積電"}], history=[])
        resolved = resolve_stock("2330", client)
        self.assertEqual(resolved.stock_id, "2330")
        self.assertEqual(resolved.stock_name, "台積電")

    def test_resolve_stock_numeric_not_found(self) -> None:
        client = FakeClient(stock_info=[{"stock_id": "2330", "stock_name": "台積電"}], history=[])
        with self.assertRaises(StockQueryError) as ctx:
            resolve_stock("123", client)
        self.assertEqual(ctx.exception.error_code, "STOCK_NOT_FOUND")

    def test_resolve_stock_5_digit_etf_success(self) -> None:
        client = FakeClient(stock_info=[{"stock_id": "00662", "stock_name": "富邦NASDAQ"}], history=[])
        resolved = resolve_stock("00662", client)
        self.assertEqual(resolved.stock_id, "00662")

    def test_resolve_stock_alphanumeric_id_success(self) -> None:
        client = FakeClient(stock_info=[{"stock_id": "00865B", "stock_name": "國泰US短期公債"}], history=[])
        resolved = resolve_stock("00865B", client)
        self.assertEqual(resolved.stock_id, "00865B")

    def test_resolve_stock_alphanumeric_id_normalized_success(self) -> None:
        client = FakeClient(stock_info=[{"stock_id": "00865B", "stock_name": "國泰US短期公債"}], history=[])
        resolved = resolve_stock("00865b.tw", client)
        self.assertEqual(resolved.stock_id, "00865B")

    def test_resolve_stock_name_not_found(self) -> None:
        client = FakeClient(stock_info=[{"stock_id": "2330", "stock_name": "台積電"}], history=[])
        with self.assertRaises(StockQueryError) as ctx:
            resolve_stock("不存在公司", client)
        self.assertEqual(ctx.exception.error_code, "STOCK_NOT_FOUND")

    def test_compute_annual_return_success(self) -> None:
        client = FakeClient(
            stock_info=[],
            history=[
                PricePoint(data_id="2330", date="2025-01-01", close=500.0),
                PricePoint(data_id="2330", date="2026-02-07", close=600.0),
            ],
        )
        result = compute_annual_return("2330", client)
        self.assertEqual(result.price_base, 500.0)
        self.assertEqual(result.price_latest, 600.0)
        self.assertAlmostEqual(result.annual_return, 0.2, places=6)

    def test_compute_annual_return_adjusts_for_split(self) -> None:
        class SplitEvent:
            def __init__(self, date: str, before_price: float, after_price: float) -> None:
                self.date = date
                self.before_price = before_price
                self.after_price = after_price

        client = FakeClient(
            stock_info=[],
            history=[
                PricePoint(data_id="0050", date="2025-02-07", close=198.55),
                PricePoint(data_id="0050", date="2026-02-06", close=71.9),
            ],
            split_events=[SplitEvent("2025-06-18", 188.65, 47.16)],
        )
        result = compute_annual_return("0050", client)
        self.assertGreater(result.annual_return, 0)

    def test_compute_annual_return_no_base_price(self) -> None:
        client = FakeClient(
            stock_info=[],
            history=[PricePoint(data_id="2330", date="2026-01-30", close=600.0)],
        )
        with self.assertRaises(StockQueryError) as ctx:
            compute_annual_return("2330", client)
        self.assertEqual(ctx.exception.error_code, "NO_PRICE_DATA")


if __name__ == "__main__":
    unittest.main()
