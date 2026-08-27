"""Question answering over the audit trail.

"Why did this reconcile?" is answered by *retrieving* what happened, not by
re-deriving it. The only input is the audit trail written during the run, so the
system can report decisions that were actually made and nothing else. If a
transaction is not in the trail, the honest answer is that it is not in the
trail - and that is what comes back.

There are two rendering paths and they read the same retrieved record:

* **Deterministic** - formats the stored decision directly. No API key needed,
  and it is impossible for it to state anything the trail does not contain.
* **Model-phrased** - hands the same retrieved record to a model purely to write
  it up more fluently, under instructions to use nothing else. Which vendor
  answers is decided by `agent.llm` from the environment, and is irrelevant to
  the content of the answer.

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

from agent.llm import LLMUnavailable, get_provider
from audit.trail import AuditTrail, merge_by_record
from core.normalize import format_inr, rupees_to_paise

ID_PATTERN = re.compile(r"\b(pay_[a-z0-9]+|order_[a-z0-9]+|BNK_\d+)\b", re.IGNORECASE)

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
    "weak_amount_date_evidence":
        "one bank credit did fall inside both the amount tolerance and the date window, "
        "but the match was loose and the surrounding dates held many similar amounts, so "
        "a coincidence was more likely than a genuine payout",
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
    def __init__(self, trail_path: Path | str | None = None, *, search_all: bool = True):
        """Load a run's audit trail.

        `search_all` widens the lookup to every trail in the directory when an ID
        is absent from the most recent one. Runs are per-dataset, so asking about
        a dev transaction right after a holdout run would otherwise report "not
        found" for a transaction that was reconciled perfectly well - technically
        true of that trail, and useless to the person asking.
        """
        self.trail_path = Path(trail_path) if trail_path else AuditTrail.latest()
        entries = AuditTrail.read(self.trail_path) if self.trail_path else []
        self.records = merge_by_record(entries)
        self._index = self._build_index()

        self._other_runs: dict[str, Path] = {}
        if search_all and self.trail_path:
            for other in sorted(self.trail_path.parent.glob("*.jsonl")):
                if other == self.trail_path:
                    continue
                for rid, rec in merge_by_record(AuditTrail.read(other)).items():
                    self._other_runs.setdefault(rid.lower(), other)
                    for ids in (rec.get("linked_ids") or {}).values():
                        for i in ids or []:
                            self._other_runs.setdefault(str(i).lower(), other)

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

    def _find_in_other_runs(self, query: str):
        """Locate an ID in a previous run's trail when it is absent from this one."""
        tokens = ID_PATTERN.findall(query or "") or [(query or "").strip()]
        for token in tokens:
            path = self._other_runs.get(str(token).lower())
            if not path:
                continue
            records = merge_by_record(AuditTrail.read(path))
            index = {r.lower(): r for r in records}
            for rid, rec in records.items():
                for ids in (rec.get("linked_ids") or {}).values():
                    for i in ids or []:
                        index.setdefault(str(i).lower(), rid)
            hit = index.get(str(token).lower())
            if hit:
                return path, hit, records
        return None

    def lookup(self, record_id: str) -> dict[str, Any] | None:
        rid = self._index.get(str(record_id).lower())
        return self.records.get(rid) if rid else None

    # -- answering -----------------------------------------------------

    def ask(self, question: str, *, use_llm: bool = False) -> Answer:
        rid = self.resolve(question)
        if not rid:
            elsewhere = self._find_in_other_runs(question)
            if elsewhere is not None:
                other_path, other_rid, other_records = elsewhere
                record = other_records[other_rid]
                return Answer(
                    question, other_rid, True,
                    f"[from an earlier run: {other_path.name}]\n\n" + self.explain(record),
                    "trail", record,
                )
            run = self.trail_path.name if self.trail_path else "(no trail found)"
            return Answer(
                question=question, record_id=None, found=False,
                answer=(
                    "I could not find a transaction ID from that question in any audit "
                    f"trail. The most recent run is {run} ({len(self.records)} records). "
                    "Ask about a specific settlement (pay_...), order (order_...) or bank "
                    "row (BNK_...), or run `python run_demo.py` to produce a fresh trail."
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
        import json
        try:
            provider = get_provider()
        except LLMUnavailable:
            return None
        try:
            response = provider.complete_text(
                system=QA_SYSTEM,
                user=(f"Question: {question}\n\n"
                      f"Audit record:\n{json.dumps(record, indent=2, default=str)}"),
                max_tokens=1024,
            )
            return response.text or None
        except Exception:                             # noqa: BLE001
            # Falling back to the deterministic rendering is strictly safer than
            # surfacing an error: the answer is still correct, just plainer.
            return None
