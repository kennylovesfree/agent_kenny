"""
CLI demo runner for Rebalance Copilot (MVP).
Reads mock data, runs deterministic checks, prints proposals and audit log.
"""
from __future__ import annotations

import json
from pathlib import Path

from drift_calculation import calculate_drift
from policy_checks import check_policies
from proposal_generator import generate_proposals


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_portfolio.json"
    raw = json.loads(data_path.read_text(encoding="utf-8"))

    holdings = raw.get("holdings", [])
    target = raw.get("target_allocation", [])
    constraints = raw.get("constraints", {})

    drift = calculate_drift(holdings, target)
    passed, triggered, review_flags = check_policies(holdings, constraints)
    proposals = generate_proposals(drift, review_flags)

    audit = {
        "input_snapshot": raw,
        "policy_results": {
            "passed": passed,
            "triggered_rules": triggered,
            "review_flags": review_flags,
        },
        "agent_reasoning": [
            "Computed drift between current and target allocations.",
            "Applied policy gates to constraints.",
            "Generated text-only proposals for human review.",
        ],
        "human_approval": {
            "status": "PENDING",
            "reviewer": "",
            "notes": "",
        },
    }

    print("=== Proposals ===")
    for idx, proposal in enumerate(proposals, start=1):
        print(f"{idx}. {proposal}")

    print("\n=== Audit Log (JSON) ===")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
