from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from logic.market_data_client import FinMindApiError, PricePoint
from logic import tw_stock_return_service as service_module

try:
    from fastapi.testclient import TestClient
    from logic.api_server import AdviceConfigError, app
    HAS_FASTAPI = True
except ModuleNotFoundError:
    HAS_FASTAPI = False


@dataclass
class FakeClient:
    stock_info: list[dict[str, str]]
    history: list[PricePoint]
    raise_api_error: bool = False

    def get_taiwan_stock_info(self) -> list[dict[str, str]]:
        if self.raise_api_error:
            raise FinMindApiError("upstream error")
        return self.stock_info

    def get_price_history(self, **_) -> list[PricePoint]:
        if self.raise_api_error:
            raise FinMindApiError("upstream error")
        return self.history

    def get_split_events(self, **_) -> list[object]:
        if self.raise_api_error:
            raise FinMindApiError("upstream error")
        return []


class ApiServerTests(unittest.TestCase):
    def setUp(self) -> None:
        if not HAS_FASTAPI:
            self.skipTest("fastapi is not installed")
        service_module._STOCK_INFO_CACHE["rows"] = []
        service_module._STOCK_INFO_CACHE["expires_at"] = service_module.datetime.min
        service_module._ANNUAL_RETURN_CACHE.clear()
        self.client = TestClient(app)

    def test_api_annual_return_success(self) -> None:
        fake = FakeClient(
            stock_info=[{"stock_id": "2330", "stock_name": "台積電"}],
            history=[
                PricePoint(data_id="2330", date="2025-01-01", close=500.0),
                PricePoint(data_id="2330", date="2026-02-07", close=600.0),
            ],
        )
        with patch("logic.api_server.FinMindClient.from_env", return_value=fake):
            response = self.client.post("/api/v1/tw-stock/annual-return", json={"query": "台積電"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["resolved_stock_id"], "2330")
        self.assertEqual(payload["resolved_stock_name"], "台積電")
        self.assertAlmostEqual(payload["annual_return"], 0.2, places=6)

    def test_api_stock_not_found_returns_422_shape(self) -> None:
        fake = FakeClient(stock_info=[{"stock_id": "2330", "stock_name": "台積電"}], history=[])
        with patch("logic.api_server.FinMindClient.from_env", return_value=fake):
            response = self.client.post("/api/v1/tw-stock/annual-return", json={"query": "123"})
        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertEqual(payload["error_code"], "STOCK_NOT_FOUND")
        self.assertIn("message", payload)
        self.assertIn("details", payload)

    def test_api_alphanumeric_stock_id_normalized_success(self) -> None:
        fake = FakeClient(
            stock_info=[{"stock_id": "00865B", "stock_name": "國泰US短期公債"}],
            history=[
                PricePoint(data_id="00865B", date="2025-01-01", close=34.0),
                PricePoint(data_id="00865B", date="2026-02-07", close=35.0),
            ],
        )
        with patch("logic.api_server.FinMindClient.from_env", return_value=fake):
            response = self.client.post("/api/v1/tw-stock/annual-return", json={"query": "00865b.tw"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["resolved_stock_id"], "00865B")

    def test_api_upstream_error_returns_502_shape(self) -> None:
        fake = FakeClient(stock_info=[], history=[], raise_api_error=True)
        with patch("logic.api_server.FinMindClient.from_env", return_value=fake):
            response = self.client.post("/api/v1/tw-stock/annual-return", json={"query": "2330"})
        self.assertEqual(response.status_code, 502)
        payload = response.json()
        self.assertEqual(payload["error_code"], "UPSTREAM_ERROR")
        self.assertIn("message", payload)
        self.assertIn("details", payload)

    def test_api_missing_token_returns_500(self) -> None:
        with patch("logic.api_server.FinMindClient.from_env", return_value=None):
            response = self.client.post("/api/v1/tw-stock/annual-return", json={"query": "2330"})
        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["error_code"], "CONFIG_ERROR")

    def test_api_advice_generate_success(self) -> None:
        fake_advice = SimpleNamespace(
            risk_level="medium",
            model_meta=SimpleNamespace(model="gpt-4o-mini", latency_ms=100),
            model_dump=lambda: {
                "summary": "測試建議",
                "risk_level": "medium",
                "actions": [{"title": "降低集中", "reason": "單一持倉過高", "priority": "high"}],
                "watchouts": ["注意短期波動"],
                "disclaimer": "僅供參考",
                "model_meta": {"provider": "openai", "model": "gpt-4o-mini", "latency_ms": 100},
            },
        )
        payload = {
            "profile": {"age": 35, "riskLevel": "balanced", "taxRegion": "TW", "horizonYears": 10},
            "portfolio": {
                "totalUsd": 1000,
                "expectedReturn": 0.1,
                "volatility": 0.2,
                "maxDrawdown": 0.25,
                "positions": [],
                "allocationSuggestion": [],
            },
            "locale": "zh-TW",
        }
        with patch("logic.api_server.generate_advice", return_value=fake_advice):
            response = self.client.post("/api/v1/advice/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["risk_level"], "medium")
        self.assertIn("actions", body)

    def test_api_advice_config_error(self) -> None:
        payload = {
            "profile": {"age": 35, "riskLevel": "balanced", "taxRegion": "TW", "horizonYears": 10},
            "portfolio": {
                "totalUsd": 1000,
                "expectedReturn": 0.1,
                "volatility": 0.2,
                "maxDrawdown": 0.25,
                "positions": [],
                "allocationSuggestion": [],
            },
            "locale": "zh-TW",
        }
        with patch("logic.api_server.generate_advice", side_effect=AdviceConfigError("Missing OPENAI_API_KEY.")):
            response = self.client.post("/api/v1/advice/generate", json=payload)
        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertEqual(body["error_code"], "AI_CONFIG_ERROR")


if __name__ == "__main__":
    unittest.main()
