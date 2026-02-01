"""
Generate 1–3 textual proposals. No execution logic.
"""
from __future__ import annotations

from typing import Dict, List


def generate_proposals(drift: Dict[str, dict], review_flags: List[str]) -> List[str]:
    """
    Generate simple text proposals based on drift.
    If review_flags exist, prepend a Needs Human Review notice.
    """
    proposals = []

    header = "Needs Human Review: " + "; ".join(review_flags) if review_flags else None

    # Proposal 1: Highlight largest positive drift
    largest_over = max(drift.items(), key=lambda x: x[1]["drift"])
    if largest_over[1]["drift"] > 0:
        text = (
            f"Option A: Consider reducing {largest_over[0]} exposure "
            f"(current {largest_over[1]['current']:.2f} vs target {largest_over[1]['target']:.2f})."
        )
        proposals.append(text)

    # Proposal 2: Highlight largest negative drift
    largest_under = min(drift.items(), key=lambda x: x[1]["drift"])
    if largest_under[1]["drift"] < 0:
        text = (
            f"Option B: Consider increasing {largest_under[0]} exposure "
            f"(current {largest_under[1]['current']:.2f} vs target {largest_under[1]['target']:.2f})."
        )
        proposals.append(text)

    # Proposal 3: Neutral / Hold
    proposals.append("Option C: Hold current allocation; review later with human approval.")

    if header:
        proposals = [header] + proposals

    return proposals
