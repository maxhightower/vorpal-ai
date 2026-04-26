"""Built-in benchmark cases used by the evaluation harness."""

from __future__ import annotations

from typing import Any

# Each case is intentionally small and inspectable. ``expected`` describes
# properties the harness should preserve, not exact strings.
CASES: list[dict[str, Any]] = [
    {
        "name": "numerical_percentage_change",
        "question": "What is the percentage increase from 100 to 125?",
        "expected": {
            "claim_count": 1,
            "first_claim_type": "NUMERICAL",
            "first_verdict": "COMPUTED",
            "answer_must_contain": ["25"],
        },
    },
    {
        "name": "causal_without_evidence",
        "question": "Did the new ad campaign cause sales to increase?",
        "expected": {
            "first_claim_type": "CAUSAL",
            "verdicts_must_include": ["UNSUPPORTED"],
            "answer_must_contain": ["correlation", "causal"],
        },
    },
    {
        "name": "predictive",
        "question": "Will demand increase next quarter?",
        "expected": {
            "first_claim_type": "PREDICTIVE",
            "answer_must_contain": ["uncertainty", "prediction"],
            "verdicts_must_not_include": ["VERIFIED"],
        },
    },
    {
        "name": "procedural_without_policy",
        "question": "Is this action allowed under the policy?",
        "expected": {
            "first_claim_type": "PROCEDURAL",
            "verdicts_must_include": ["UNCLEAR"],
        },
    },
    {
        "name": "contradiction_remote_work",
        "question": "Can employees work remotely full-time?",
        "documents": [
            {
                "name": "DocA",
                "text": "Employees may work remotely up to five days per week.",
                "effective_date": "2024-01-01",
            },
            {
                "name": "DocB",
                "text": "As of March 2026, employees must work in office three days per week.",
                "effective_date": "2026-03-01",
            },
        ],
        "expected": {
            "answer_must_contain": ["office", "three"],
            "contradiction_expected": True,
        },
    },
    {
        "name": "unsupported_synthesis_best_option",
        "question": "Summarize this product and say whether it is the best option.",
        "expected": {
            "verdicts_must_include": ["UNSUPPORTED", "UNCLEAR"],
            "answer_must_contain": ["best"],
        },
    },
]
