"""Prompt construction for the reasoning layer.

Two principles shape everything here.

**Grounded, not guessing.** The model is handed the actual records the engine
looked at - the settlement, the ledger row, and every near-miss credit with its
real amount, date and narration. It is never asked to speculate about data it
cannot see. Every hypothesis it offers must cite record IDs that appear in its
input, which makes the output checkable rather than plausible-sounding.

**Refusal is a first-class answer.** The schema requires `sufficient_evidence`,
and the instructions say plainly that "there is not enough here to decide" is a
correct response that will not be penalised. Without that, a model asked to
explain an unmatched transaction will reliably invent a match - it reads the
task as "find the answer" rather than "judge whether an answer exists".

The system prompt is deliberately stable so it can be cached across batches.
"""

from __future__ import annotations

import json
from typing import Any

from core.config import MatchConfig
from core.normalize import paise_to_rupees

RECOMMENDED_ACTIONS = ["auto_resolve", "flag_duplicate", "escalate_to_human"]
CONFIDENCE_LABELS = ["high", "medium", "low"]


def system_prompt(cfg: MatchConfig) -> str:
    """Stable across every batch in a run, so it caches cleanly."""
    return f"""You are a reconciliation analyst reviewing exceptions from an automated \
three-way matching engine at a payment gateway. The three sources are:

- SETTLEMENT: what the gateway says it paid out (gross, fee, GST on fee, net, UTR)
- BANK: what actually landed in the merchant's bank account (credit, value date, narration)
- LEDGER: what the merchant's own order system expected (gross order value, status)

A deterministic engine has already run. It matched what it could and handed you \
only the cases it could not resolve, or resolved with low confidence. Its thresholds:

- Amounts match within {cfg.amount_tolerance_paise} paise \
(Rs {paise_to_rupees(cfg.amount_tolerance_paise)}).
- A credit may land on the settlement date or up to {cfg.date_window_days} days after, never before.
- A split settlement may span at most {cfg.max_split_legs} credits.
- When two or more credits satisfy both thresholds, the engine deliberately claims \
NEITHER rather than picking the closer one.

The engine is the authority on what matched. You are NOT deciding matches. Your job \
is to explain WHY a case did not resolve, and to recommend what a human should do. \
Nothing you write changes a match.

Common real causes, so you can recognise them:
- The bank credits NET while the ledger records GROSS. The gap equals fee + 18% GST on \
the fee. This is normal, not a discrepancy.
- Settlement lands T+1 or T+2, so a same-day comparison misses it.
- One settlement is paid out as several credits on different days.
- A partial refund means the credit is legitimately short of the settlement net.
- A webhook retry emits the same settlement twice against one real credit.
- Sub-rupee drift from currency rounding.
- Two unrelated transactions can have near-identical amounts and dates. This is the \
trap: they look like the same payment and are not.

RULES YOU MUST FOLLOW:

1. Cite only record IDs that appear in the case you are given. Never invent an ID, \
an amount or a date.
2. If the evidence does not identify a single answer, set sufficient_evidence to false \
and recommend escalate_to_human. This is a CORRECT and valued outcome. Do not \
manufacture a match to appear useful. A wrong auto_resolve on money is far more \
costly than an honest escalation.
3. A near-miss credit being the only nearby candidate is NOT evidence that it is the \
right one. If it sits outside the engine's thresholds, say so and say by how much.
4. Recommend auto_resolve only when the evidence uniquely and unambiguously explains \
the case with no competing reading.
5. Write the hypothesis for a finance operations person: plain English, specific \
numbers, no jargon about tiers or algorithms. One or two sentences.
6. Write amounts as "Rs 1,234.56" and use ASCII characters only. This text is \
printed to terminals that cannot always encode the rupee sign.

Return one investigation per case, in the same order you received them."""


INVESTIGATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "investigations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "case_id": {
                        "type": "string",
                        "description": "The case_id exactly as given in the input.",
                    },
                    "hypothesis": {
                        "type": "string",
                        "description": (
                            "One or two plain-English sentences explaining why this did "
                            "not reconcile, with specific amounts and dates."
                        ),
                    },
                    "sufficient_evidence": {
                        "type": "boolean",
                        "description": (
                            "True only if the evidence identifies a single answer. False "
                            "when the data genuinely cannot decide - this is a correct "
                            "and expected outcome."
                        ),
                    },
                    "confidence": {"type": "string", "enum": CONFIDENCE_LABELS},
                    "recommended_action": {"type": "string", "enum": RECOMMENDED_ACTIONS},
                    "evidence_cited": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Record IDs from THIS case that support the hypothesis.",
                    },
                    "rupee_impact": {
                        "type": "string",
                        "description": (
                            "The amount at stake as a plain decimal string, or 'unknown'."
                        ),
                    },
                },
                "required": ["case_id", "hypothesis", "sufficient_evidence", "confidence",
                             "recommended_action", "evidence_cited", "rupee_impact"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["investigations"],
    "additionalProperties": False,
}


def build_user_message(bundles: list[dict[str, Any]]) -> str:
    return (
        f"Investigate the following {len(bundles)} reconciliation "
        f"{'case' if len(bundles) == 1 else 'cases'}.\n\n"
        + json.dumps(bundles, indent=2, sort_keys=True)
    )
