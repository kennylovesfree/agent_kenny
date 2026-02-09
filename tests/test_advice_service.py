from __future__ import annotations

import unittest
from unittest.mock import patch

from logic.advice_service import AdviceConfigError, AdviceRequest, AdviceResponse, generate_advice


def _sample_payload() -> AdviceRequest:
    return AdviceRequest.model_validate(
        {
            "profile": {"age": 35, "riskLevel": "balanced", "taxRegion": "TW", "horizonYears": 10},
            "portfolio": {
                "totalUsd": 1000,
                "expectedReturn": 0.08,
                "volatility": 0.16,
                "maxDrawdown": 0.2,
                "positions": [],
                "allocationSuggestion": [],
            },
            "locale": "zh-TW",
        }
    )


def _sample_response(provider: str, model: str) -> AdviceResponse:
    return AdviceResponse.model_validate(
        {
            "summary": "測試",
            "risk_level": "medium",
            "actions": [{"title": "調整", "reason": "測試", "priority": "medium"}],
            "watchouts": ["注意風險"],
            "disclaimer": "僅供參考",
            "model_meta": {"provider": provider, "model": model, "latency_ms": 10},
        }
    )


class AdviceServiceTests(unittest.TestCase):
    def test_generate_advice_uses_gemini_when_provider_set(self) -> None:
        payload = _sample_payload()
        with patch.dict(
            "os.environ",
            {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "x", "GEMINI_MODEL": "gemini-2.0-flash"},
            clear=False,
        ):
            with patch("logic.advice_service._call_gemini", return_value=_sample_response("gemini", "gemini-2.0-flash")):
                result = generate_advice(payload)
        self.assertEqual(result.model_meta.provider, "gemini")

    def test_generate_advice_gemini_missing_key_raises(self) -> None:
        payload = _sample_payload()
        with patch.dict("os.environ", {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": ""}, clear=False):
            with self.assertRaises(AdviceConfigError):
                generate_advice(payload)


if __name__ == "__main__":
    unittest.main()
