"""Multi-turn investigation of the hardest exceptions.

The batched reasoner in `reasoner.py` explains a case from evidence it is handed.
This module goes further: for cases where the engine actually had a candidate in
front of it and declined, the agent gets **read-only tools** and several turns to
look for itself. It chooses what to query, sees real results, and decides when it
has enough - or when it does not and never will.

Why this exists
---------------
An agent that refuses to match something it was never allowed to look for is not
demonstrating judgement, it is restating its input. An agent that widens the
search past the engine's thresholds, finds the one candidate that exists, works
out that the amount gap is unexplained and the date is six days late, and *still*
declines - that is judgement, and it is checkable, because every query it ran is
in the audit trail.

Design constraints
------------------
* **Bounded.** A hard cap on turns. An agent that cannot conclude in a few steps
  is not converging, and an unbounded loop over money is not a feature.
* **Read-only.** Every tool is a question (see `tools.py`). No sequence of actions
  can change a reconciliation outcome.
* **Provider-agnostic.** The loop is expressed as schema-constrained JSON rather
  than vendor-specific tool-calling, so it runs identically on Anthropic and
  OpenAI. It also means the whole investigation is plain data - loggable,
  replayable, and cacheable.
* **Selective.** Only the hard cases get this treatment. Everything else takes
  the cheap batched path. Effort is matched to difficulty, the same way the
  matching engine spends its effort.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent.prompts import CONFIDENCE_LABELS, RECOMMENDED_ACTIONS
from agent.tools import TOOL_NAMES, BatchInvestigator, run_tool
from core.config import MatchConfig
from core.normalize import paise_to_rupees

MAX_TURNS = 4
MAX_TOKENS = 8000


INVESTIGATION_SYSTEM = """You are a reconciliation analyst investigating one \
exception that an automated matching engine could not resolve.

Unlike a quick review, you have tools and several turns. Use them. The engine \
searched only within its own thresholds; you can look wider and find out whether \
a plausible counterpart exists at all.

Engine thresholds, for context:
- amounts match within {tolerance} paise
- a credit may land on the settlement date or up to {window} days after, never before
- when two or more credits satisfy both, the engine claims NEITHER

Available actions:
- credits_near_settlement : every plausible credit near one settlement (start here)
- search_credits          : credits by amount range and/or date range
- get_credit              : full detail on one bank row
- get_settlement          : full detail on one settlement, plus its ledger order
- find_utr                : search settlements and credits by UTR fragment
- batch_summary           : size and date range of the batch
- conclude                : finish, with your investigation

How to work:
1. Start by looking wider than the engine did. Establish what actually exists.
2. If you find a candidate outside the thresholds, say precisely how far outside \
and whether anything explains the gap. A fee, a refund, or a rounding difference \
is an explanation; "it is the closest one" is not.
3. Two or more equally good candidates means the data does not identify a winner. \
Conclude that, and escalate.
4. Conclude as soon as you know. Do not spend turns confirming what you already \
established. You have at most {max_turns} turns; after that you must conclude.

When you conclude:
- set sufficient_evidence to false if the data cannot identify a single answer. \
That is a CORRECT and valued outcome. Never manufacture a match to look useful - \
a wrong auto_resolve on money costs far more than an honest escalation.
- cite only record IDs you actually saw in a tool result.
- write the hypothesis for a finance operations person: plain English, specific \
amounts and dates, two or three sentences. Use ASCII only, and write amounts as \
"Rs 1,234.56"."""


STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "One sentence: what you are trying to establish with this step.",
        },
        "action": {"type": "string", "enum": list(TOOL_NAMES) + ["conclude"]},
        "record_id": {
            "type": ["string", "null"],
            "description": "For get_credit / get_settlement / credits_near_settlement.",
        },
        "min_amount": {"type": ["string", "null"], "description": "search_credits, e.g. '1200.00'"},
        "max_amount": {"type": ["string", "null"], "description": "search_credits"},
        "start_date": {"type": ["string", "null"], "description": "search_credits, YYYY-MM-DD"},
        "end_date": {"type": ["string", "null"], "description": "search_credits, YYYY-MM-DD"},
        "utr_fragment": {"type": ["string", "null"], "description": "find_utr, 4+ digits"},
        "conclusion": {
            "description": "Required when action is 'conclude', otherwise null.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "hypothesis": {"type": "string"},
                        "sufficient_evidence": {"type": "boolean"},
                        "confidence": {"type": "string", "enum": CONFIDENCE_LABELS},
                        "recommended_action": {"type": "string", "enum": RECOMMENDED_ACTIONS},
                        "evidence_cited": {"type": "array", "items": {"type": "string"}},
                        "rupee_impact": {"type": "string"},
                    },
                    "required": ["hypothesis", "sufficient_evidence", "confidence",
                                 "recommended_action", "evidence_cited", "rupee_impact"],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ],
        },
    },
    "required": ["thought", "action", "record_id", "min_amount", "max_amount",
                 "start_date", "end_date", "utr_fragment", "conclusion"],
    "additionalProperties": False,
}



def parse_first_object(text: str) -> dict[str, Any]:
    """Parse the first complete JSON object in `text`.

    Belt and braces against a provider returning more than one object glued
    together. The extraction in `agent/llm.py` handles the known cause; this
    keeps an unknown one from taking down a whole investigation.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text)
    return obj


@dataclass
class InvestigationStep:
    turn: int
    thought: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"turn": self.turn, "thought": self.thought, "action": self.action,
                "params": {k: v for k, v in self.params.items() if v is not None},
                "observation": self.observation}


def system_prompt(cfg: MatchConfig) -> str:
    return INVESTIGATION_SYSTEM.format(
        tolerance=cfg.amount_tolerance_paise,
        window=cfg.date_window_days,
        max_turns=MAX_TURNS,
    )


def needs_deep_investigation(result) -> bool:
    """Which exceptions are worth a multi-turn investigation.

    The ones where the engine had something in front of it and declined, or where
    the absence itself is the interesting finding. A duplicate row or a credit
    that resembles nothing needs a sentence, not an investigation - and spending
    four model turns on it would be exactly the kind of undisciplined AI use this
    project argues against.
    """
    if result.status != "unresolved":
        return False
    if result.near_misses:
        return True
    return result.exception_reason in {
        "ambiguous_candidates", "contested_candidate",
        "ambiguous_split", "contested_split_legs", "no_candidate_found",
    }


def investigate_case(provider, bundle: dict[str, Any], inv: BatchInvestigator,
                     cfg: MatchConfig, *, max_turns: int = MAX_TURNS
                     ) -> tuple[dict[str, Any] | None, list[InvestigationStep], int]:
    """Run one bounded investigation.

    Returns ``(conclusion, steps, calls_made)``. A conclusion of None means the
    agent never converged - the caller escalates, which is the safe default.
    """
    steps: list[InvestigationStep] = []
    calls = 0
    transcript: list[str] = [
        "CASE UNDER INVESTIGATION:",
        json.dumps(bundle, indent=2, sort_keys=True),
    ]

    for turn in range(1, max_turns + 1):
        remaining = max_turns - turn
        instruction = (
            f"\nTurn {turn} of {max_turns}."
            + (" This is your LAST turn - you must set action to 'conclude'."
               if remaining == 0 else
               f" {remaining} turn(s) remain after this one.")
            + "\nChoose your next action."
        )
        try:
            response = provider.complete_json(
                system=system_prompt(cfg),
                user="\n".join(transcript) + instruction,
                schema=STEP_SCHEMA,
                max_tokens=MAX_TOKENS,
                schema_name="investigation_step",
            )
            calls += 1
            step_data = parse_first_object(response.text)
        except Exception as exc:                      # noqa: BLE001
            steps.append(InvestigationStep(turn, f"step failed: {exc}", "error"))
            return None, steps, calls

        action = step_data.get("action", "conclude")
        thought = step_data.get("thought", "")
        params = {k: step_data.get(k) for k in
                  ("record_id", "min_amount", "max_amount", "start_date",
                   "end_date", "utr_fragment")}

        if action == "conclude":
            conclusion = step_data.get("conclusion")
            steps.append(InvestigationStep(turn, thought, "conclude"))
            if isinstance(conclusion, dict):
                conclusion["case_id"] = bundle["case_id"]
                return conclusion, steps, calls
            return None, steps, calls

        observation = run_tool(inv, action, params)
        steps.append(InvestigationStep(turn, thought, action, params, observation))
        transcript.append(f"\n--- turn {turn} ---")
        transcript.append(f"You said: {thought}")
        transcript.append(f"You ran: {action} {json.dumps({k: v for k, v in params.items() if v})}")
        transcript.append(f"Result:\n{json.dumps(observation, indent=2)}")

    return None, steps, calls
