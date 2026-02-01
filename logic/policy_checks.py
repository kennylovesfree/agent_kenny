"""
Rule-based gates only. Any uncertainty => Needs Human Review.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def check_policies(holdings: List[dict], constraints: dict) -> Tuple[bool, List[str], List[str]]:
    """
    Returns (passed, triggered_rules, review_flags)
    - passed: bool
    - triggered_rules: list of rule IDs that failed
    - review_flags: list of "Needs Human Review" reasons
    """
    triggered = []
    review_flags = []

    max_single = float(constraints.get("max_single_asset_pct", 1.0))
    max_equity = float(constraints.get("max_equity_pct", 1.0))
    no_leverage = bool(constraints.get("no_leverage", True))

    # Rule: No leverage (placeholder; we only flag if user says leverage exists)
    if not no_leverage:
        triggered.append("RULE_LEVERAGE_NOT_ALLOWED")

    # Rule: Max single asset weight
    for h in holdings:
        if float(h["weight"]) > max_single:
            triggered.append("RULE_MAX_SINGLE_ASSET")
            break

    # Rule: Max equity exposure (US_EQUITY + INTL_EQUITY in this demo)
    equity = sum(
        float(h["weight"]) for h in holdings if h["asset"] in {"US_EQUITY", "INTL_EQUITY"}
    )
    if equity > max_equity:
        triggered.append("RULE_MAX_EQUITY")

    # Uncertainty gate: if constraints are missing or unknown
    if "max_single_asset_pct" not in constraints or "max_equity_pct" not in constraints:
        review_flags.append("Needs Human Review: missing constraints")

    passed = len(triggered) == 0
    return passed, triggered, review_flags
