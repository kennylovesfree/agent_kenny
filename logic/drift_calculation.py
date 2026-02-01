"""
Pure, deterministic drift calculation.
No advice, no optimization.
"""
from __future__ import annotations

from typing import Dict, List


def calculate_drift(holdings: List[dict], target_allocation: List[dict]) -> Dict[str, dict]:
    """
    Returns a dict keyed by asset:
    {
      "ASSET": {"current": 0.55, "target": 0.50, "drift": 0.05}
    }
    """
    current_map = {h["asset"]: float(h["weight"]) for h in holdings}
    target_map = {t["asset"]: float(t["weight"]) for t in target_allocation}

    assets = sorted(set(current_map.keys()) | set(target_map.keys()))
    drift = {}
    for asset in assets:
        cur = current_map.get(asset, 0.0)
        tgt = target_map.get(asset, 0.0)
        drift[asset] = {
            "current": cur,
            "target": tgt,
            "drift": cur - tgt,
        }
    return drift
