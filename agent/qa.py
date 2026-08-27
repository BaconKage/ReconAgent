"""Question answering over the audit trail.

"Why did this reconcile?" is answered by *retrieving* what happened, not by
re-deriving it. The only input is the audit trail written during the run, so the
system can report decisions that were actually made and nothing else. If a
transaction is not in the trail, the honest answer is that it is not in the
trail - and that is what comes back.

There are two rendering paths and they read the same retrieved record:

* **Deterministic** - formats the stored decision directly. No API key needed,
  and it is impossible for it to state anything the trail does not contain.
* **Model-phrased** - hands the same retrieved record to Claude purely to write
  it up more fluently, under instructions to use nothing else.

The deterministic path is the default and the fallback. The model is a
presentation layer over retrieval, which is the whole point: it cannot
hallucinate a reconciliation outcome, because it is not the thing that decides
outcomes and it is not given the freedom to look beyond one record.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audit.trail import AuditTrail, merge_by_record
from core.normalize import format_inr, rupees_to_paise

ID_PATTERN = re.compile(r"\b(pay_[a-z0-9]+|order_[a-z0-9]+|BNK_\d+)\b", re.IGNORECASE)

MODEL = "claude-opus-5"

QA_SYSTEM = """You answer questions about a single reconciliation decision that has \
already been made and recorded in an audit trail.

You will be given exactly one audit record. Answer ONLY from that record.

- Do not speculate about data not in the record.
- Do not re-decide whether the transaction should have matched. The decision is \
already made; your job is to report and explain it.
- If the record does not contain what was asked, say so plainly.
- Quote the specific amounts, dates and record IDs that appear in the record.
- Write for a finance operations person: two to four sentences, plain English, no \
jargon about matching tiers or algorithms.
"""

STATUS_PHRASING = {
    "matched": "reconciled cleanly",
    "matched_split": "reconciled as a split settlement",
    "duplicate": "flagged as a duplicate",
    "unresolved": "did NOT reconcile",
}

REASON_PHRASING = {
    "identifier_match_amount_discrepancy":
        "the UTR matched a bank credit, but the amount credited did not agree with the "
        "settlement net",
    "ambiguous_candidates":
        "more than one bank credit satisfied both the amount tolerance and the date "
        "window, so binding either one would have been a guess",
    "contested_candidate":
        "its only candidate credit was also the only candidate for another settlement, "
        "and one credit cannot settle both",
    "ambiguous_split":
        "several different combinations of credits added up to the settlement net, so no "
        "single combination was identifiable",
    "contested_split_legs":
        "the credits it would need are also claimed by another settlement's split",
    "no_candidate_found":
        "no bank credit satisfied both the amount tolerance and the date window",
    "no_settlement_counterpart":
        "this bank credit was never claimed by any settlement in the batch",
    "duplicate_settlement_row":
        "the same settlement appeared more than once, which is typical of a webhook retry",
}


@dataclass
class Answer:
    question: str
    record_id: str | None
    found: bool
    answer: str
    source: str                       # "trail" | "trail+model" | "not_found"
    record: dict[str, Any] | None = field(default=None, repr=False)


class ReconciliationQA:
    def __init__(self, trail_path: Path | str | None = None):
        self.trail_path = Path(trail_path) if trail_path else AuditTrail.latest()
        entries = AuditTrail.read(self.trail_path) if self.trail_path else []
        self.records = merge_by_record(entries)
        self._index = self._build_index()

    def _build_index(self) -> dict[str, str]:
        """Map every ID mentioned anywhere in a group back to that group's record.

        A user asks about the ID they have in front of them, which is often a bank
        row or an order rather than the transaction the group is keyed by.
        """
        idx: dict[str, str] = {}
        for rid, rec in self.records.items():
            idx.setdefault(rid.lower(), rid)
            for ids in (rec.get("linked_ids") or {}).values():
                for i in ids or []:
                    idx.setdefault(str(i).lower(), rid)
        return idx

    # -- retrieval -----------------------------------------------------

    def resolve(self, query: str) -> str | None:
        for token in ID_PATTERN.findall(query or ""):
            hit = self._index.get(token.lower())
            if hit:
                return hit
        bare = (query or "").strip().lower()
        return self._index.get(bare)

    def lookup(self, record_id: str) -> dict[str, Any] | None:
        rid = self._index.get(str(record_id).lower())
        return self.records.get(rid) if rid else None

    # -- answering -----------------------------------------------------

    def ask(self, question: str, *, use_llm: bool = False) -> Answer:
        rid = self.resolve(question)
        if not rid:
            known = len(self.records)
            return Answer(
                question=question, record_id=None, found=False,
                answer=(
                    "I could not find a transaction ID in that question that appears in "
                    f"this run's audit trail ({known} records). Ask about a specific "
                    "settlement (pay_...), order (order_...) or bank row (BNK_...) from "
                    "this batch."
                ),
                source="not_found",
            )

        record = self.records[rid]
        deterministic = self.explain(record)
        if not use_llm:
            return Answer(question, rid, True, deterministic, "trail", record)

        phrased = self._phrase_with_model(question, record)
        if phrased is None:
            return Answer(question, rid, True, deterministic, "trail", record)
        return Answer(question, rid, True, phrased, "trail+model", record)

    def explain(self, record: dict[str, Any]) -> str:
        """Render a stored decision as prose, using only what is in the record."""
        rid = record.get("record_id", "(unknown)")
        status = record.get("status", "unknown")
        lines = [f"{rid} {STATUS_PHRASING.get(status, status)}."]

        linked = record.get("linked_ids") or {}
        banks = linked.get("bank") or []
        ledger = linked.get("ledger") or []

        if status in ("matched", "matched_split", "duplicate"):
            how = record.get("match_type") or "an engine rule"
            conf = record.get("confidence")
            bit = f" It was linked to {', '.join(banks)}" if banks else ""
            lines.append(f"Matched by {how}{bit}"
                         + (f", confidence {conf:.2f}." if isinstance(conf, float) else "."))
        else:
            reason = record.get("exception_reason")
            lines.append("Reason: " + REASON_PHRASING.get(
                reason, reason or "the engine did not record a reason") + ".")
            if banks:
                lines.append(f"Bank credit(s) held against it: {', '.join(banks)}.")

        near = record.get("near_misses") or []
        if near:
            bits = []
            for nm in near[:3]:
                d = nm.get("amount_delta_paise")
                g = nm.get("date_delta_days")
                piece = nm.get("record_id", "?")
                detail = []
                if isinstance(d, int):
                    detail.append("exact amount" if d == 0 else
                                  f"{format_inr(abs(d))} {'over' if d > 0 else 'under'}")
                if isinstance(g, int):
                    detail.append(f"T+{g}")
                bits.append(f"{piece} ({', '.join(detail)})" if detail else piece)
            lines.append("Closest candidates the engine considered and declined: "
                         + "; ".join(bits) + ".")

        if ledger:
            lines.append(f"Linked ledger order: {', '.join(ledger)}.")

        reasoning = record.get("reasoning")
        if reasoning and reasoning.get("hypothesis"):
            lines.append("")
            lines.append(f"Agent's assessment ({reasoning.get('source', 'unknown')}): "
                         + reasoning["hypothesis"])
            action = reasoning.get("recommended_action")
            conf = reasoning.get("confidence")
            if action:
                suffix = f" (confidence: {conf})" if conf else ""
                lines.append(f"Recommended action: {action}{suffix}.")
            if reasoning.get("sufficient_evidence") is False:
                lines.append("The agent judged the available evidence insufficient to "
                             "decide this case, and deferred to human review.")
        return "\n".join(lines)

    # -- optional model phrasing ---------------------------------------

    def _phrase_with_model(self, question: str, record: dict[str, Any]) -> str | None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        try:
            import anthropic
        except ImportError:
            return None
        import json
        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=QA_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Audit record:\n{json.dumps(record, indent=2, default=str)}"
                    ),
                }],
            )
            return next((b.text for b in response.content if b.type == "text"), None)
        except Exception:                             # noqa: BLE001
            # Falling back to the deterministic rendering is strictly safer than
            # surfacing an error: the answer is still correct, just plainer.
            return None
