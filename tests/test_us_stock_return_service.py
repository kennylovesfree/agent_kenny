from __future__ import annotations

import unittest
from datetime import date, timedelta
from unittest.mock import patch

from logic.market_data_client import PricePoint
from logic import us_stock_return_service as service_module
from logic.us_stock_return_service import StockQueryError, compute_annual_return


class UsStockReturnServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        service_module._ANNUAL_RETURN_CACHE.clear()

    def _build_history(self) -> list[PricePoint]:
        today = date.today()
        return [
            PricePoint(data_id="AAPL", date=(today - timedelta(days=365 * 5 + 5)).isoformat(), close=100.0),
            PricePoint(data_id="AAPL", date=(today - timedelta(days=365 * 3 + 5)).isoformat(), close=140.0),
            PricePoint(data_id="AAPL", date=(today - timedelta(days=365 + 5)).isoformat(), close=180.0),
            PricePoint(data_id="AAPL", date=(today - timedelta(days=1)).isoformat(), close=200.0),
        ]

    def test_compute_annual_return_success(self) -> None:
        with patch("logic.us_stock_return_service.fetch_us_price_history", return_value=self._build_history()):
            resolved_ticker, result = compute_annual_return("aapl.us")
        self.assertEqual(resolved_ticker, "AAPL")
        self.assertEqual(result.price_latest, 200.0)
        self.assertGreater(result.annual_return, 0)

    def test_compute_annual_return_invalid_query(self) -> None:
        with self.assertRaises(StockQueryError) as ctx:
            compute_annual_return("@@@")
        self.assertEqual(ctx.exception.error_code, "INVALID_QUERY")

    def test_compute_annual_return_stock_not_found(self) -> None:
        with patch("logic.us_stock_return_service.fetch_us_price_history", return_value=[]):
            with self.assertRaises(StockQueryError) as ctx:
                compute_annual_return("MSFT")
        self.assertEqual(ctx.exception.error_code, "STOCK_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
