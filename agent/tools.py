"""Read-only query surface over a reconciliation batch.

These are the tools the investigating agent may call. Every one of them is a
*question*; none of them is an instruction. There is deliberately no tool to
create a link, change a status, adjust a confidence, or mark anything resolved -
so no sequence of agent actions, however confused or adversarial, can alter a
reconciliation outcome. The agent can look at anything and change nothing.

That constraint is what makes it safe to give the agent real autonomy over *how*
it investigates. It also makes the audit trail meaningful: every query it ran and
every answer it saw is recorded, so a reviewer can follow the same path.

What the tools expose is exactly what a human reviewer would have on screen - the
three source files. They do not expose ground truth, case labels, or the engine's
own scoring.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from core.loader import LoadedBatch
from core.matcher import ReconciliationReport
from core.normalize import paise_to_rupees, parse_date, rupees_to_paise

MAX_RESULTS = 12


class BatchInvestigator:
    """Answers questions about a batch. Holds no mutable reconciliation state."""

    def __init__(self, batch: LoadedBatch, report: ReconciliationReport):
        self._settlements = {s.transaction_id: s for s in batch.settlements}
        self._banks = {b.bank_row_id: b for b in batch.banks}
        self._ledger = dict(batch.ledger)
        # Which credits the engine already bound elsewhere. The agent needs this
        # to reason about availability, and it is an observation, not a lever.
        self._claimed: set[str] = set()
        for r in report.results:
            if r.status in ("matched", "matched_split", "duplicate"):
                self._claimed.update(r.linked_ids.get("bank", []))

    # -- serialisers ---------------------------------------------------

    def _credit(self, b) -> dict[str, Any]:
        return {
            "bank_row_id": b.bank_row_id,
            "credit": paise_to_rupees(b.credit_paise),
            "value_date": b.value_date.isoformat(),
            "narration": b.description,
            "utr_reference": b.utr_reference or "(none)",
            "utr_recovered": b.parsed_utr or "(none)",
            "already_matched_to_another_settlement": b.bank_row_id in self._claimed,
        }

    def _settlement(self, s) -> dict[str, Any]:
        return {
            "transaction_id": s.transaction_id,
            "order_id": s.order_id,
            "gross": paise_to_rupees(s.gross_paise),
            "fee": paise_to_rupees(s.fee_paise),
            "gst_on_fee": paise_to_rupees(s.tax_on_fee_paise),
            "net_expected_in_bank": paise_to_rupees(s.net_paise),
            "settlement_date": s.settlement_date.isoformat(),
            "utr": s.utr_number or "(none)",
        }

    def _order(self, o) -> dict[str, Any]:
        return {
            "order_id": o.order_id,
            "customer": o.customer,
            "order_value_gross": paise_to_rupees(o.expected_paise),
            "order_date": o.order_date.isoformat(),
            "status": o.status,
        }

    # -- tools ---------------------------------------------------------

    def search_credits(self, min_amount: str | None = None, max_amount: str | None = None,
                       start_date: str | None = None, end_date: str | None = None,
                       include_matched: bool = False) -> dict[str, Any]:
        """Find bank credits inside an amount and/or date range.

        This is the tool that makes an investigation real rather than a
        restatement: the agent can widen its own search past the engine's
        thresholds and find out whether anything plausible exists at all.
        """
        lo = rupees_to_paise(min_amount) if min_amount else None
        hi = rupees_to_paise(max_amount) if max_amount else None
        d0, d1 = parse_date(start_date), parse_date(end_date)

        hits = []
        for b in self._banks.values():
            if lo is not None and b.credit_paise < lo:
                continue
            if hi is not None and b.credit_paise > hi:
                continue
            if d0 and b.value_date < d0:
                continue
            if d1 and b.value_date > d1:
                continue
            if not include_matched and b.bank_row_id in self._claimed:
                continue
            hits.append(b)

        hits.sort(key=lambda b: (b.value_date, b.credit_paise))
        return {
            "tool": "search_credits",
            "criteria": {"min_amount": min_amount, "max_amount": max_amount,
                         "start_date": start_date, "end_date": end_date,
                         "include_matched": include_matched},
            "total_found": len(hits),
            "showing": min(len(hits), MAX_RESULTS),
            "credits": [self._credit(b) for b in hits[:MAX_RESULTS]],
        }

    def get_credit(self, bank_row_id: str) -> dict[str, Any]:
        b = self._banks.get((bank_row_id or "").strip())
        if not b:
            return {"tool": "get_credit", "found": False,
                    "note": f"no bank row {bank_row_id!r} exists in this batch"}
        return {"tool": "get_credit", "found": True, "credit": self._credit(b)}

    def get_settlement(self, transaction_id: str) -> dict[str, Any]:
        s = self._settlements.get((transaction_id or "").strip())
        if not s:
            return {"tool": "get_settlement", "found": False,
                    "note": f"no settlement {transaction_id!r} exists in this batch"}
        out: dict[str, Any] = {"tool": "get_settlement", "found": True,
                               "settlement": self._settlement(s)}
        if s.order_id and s.order_id in self._ledger:
            out["ledger_order"] = self._order(self._ledger[s.order_id])
        return out

    def find_utr(self, fragment: str) -> dict[str, Any]:
        """Search settlements and credits for a UTR containing this fragment.

        Bank exports truncate and mangle reference numbers, so a partial match is
        often the only thread available.
        """
        frag = "".join(ch for ch in (fragment or "") if ch.isdigit())
        if len(frag) < 4:
            return {"tool": "find_utr", "note": "give at least 4 digits to search on",
                    "settlements": [], "credits": []}
        return {
            "tool": "find_utr",
            "fragment": frag,
            "settlements": [self._settlement(s) for s in self._settlements.values()
                            if s.utr_number and frag in s.utr_number][:MAX_RESULTS],
            "credits": [self._credit(b) for b in self._banks.values()
                        if b.parsed_utr and frag in b.parsed_utr][:MAX_RESULTS],
        }

    def credits_near_settlement(self, transaction_id: str, amount_slack: str = "500.00",
                                days_before: int = 1, days_after: int = 10) -> dict[str, Any]:
        """Everything remotely plausible for one settlement, in a single call.

        A convenience over search_credits so the common investigation - "is there
        anything at all near this payout?" - costs one turn instead of three.
        """
        s = self._settlements.get((transaction_id or "").strip())
        if not s:
            return {"tool": "credits_near_settlement", "found": False,
                    "note": f"no settlement {transaction_id!r} in this batch"}
        slack = rupees_to_paise(amount_slack) or 50_000
        result = self.search_credits(
            min_amount=paise_to_rupees(max(0, s.net_paise - slack)),
            max_amount=paise_to_rupees(s.net_paise + slack),
            start_date=(s.settlement_date - timedelta(days=days_before)).isoformat(),
            end_date=(s.settlement_date + timedelta(days=days_after)).isoformat(),
        )
        for c in result["credits"]:
            credit_paise = rupees_to_paise(c["credit"]) or 0
            c["differs_from_net_by"] = paise_to_rupees(credit_paise - s.net_paise)
            c["days_after_settlement"] = (
                parse_date(c["value_date"]) - s.settlement_date).days
        result["tool"] = "credits_near_settlement"
        result["found"] = True          # same key on both paths, so callers can branch
        result["settlement_net"] = paise_to_rupees(s.net_paise)
        result["settlement_date"] = s.settlement_date.isoformat()
        return result

    def batch_summary(self) -> dict[str, Any]:
        unclaimed = [b for b in self._banks.values() if b.bank_row_id not in self._claimed]
        dates = [b.value_date for b in self._banks.values()]
        return {
            "tool": "batch_summary",
            "settlements": len(self._settlements),
            "bank_credits": len(self._banks),
            "ledger_orders": len(self._ledger),
            "credits_not_yet_matched": len(unclaimed),
            "bank_date_range": [min(dates).isoformat(), max(dates).isoformat()] if dates else [],
        }


#: Name -> callable, resolved against a BatchInvestigator instance.
TOOL_NAMES = ("search_credits", "get_credit", "get_settlement", "find_utr",
              "credits_near_settlement", "batch_summary")


def run_tool(inv: BatchInvestigator, action: str, params: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one agent-chosen action. Unknown actions are reported, not raised.

    An agent that asks for something impossible should be told so and given
    another turn, not crash the run.
    """
    if action not in TOOL_NAMES:
        return {"error": f"unknown tool {action!r}", "available": list(TOOL_NAMES)}
    try:
        if action == "search_credits":
            return inv.search_credits(
                min_amount=params.get("min_amount"), max_amount=params.get("max_amount"),
                start_date=params.get("start_date"), end_date=params.get("end_date"))
        if action == "get_credit":
            return inv.get_credit(params.get("record_id") or "")
        if action == "get_settlement":
            return inv.get_settlement(params.get("record_id") or "")
        if action == "find_utr":
            return inv.find_utr(params.get("utr_fragment") or "")
        if action == "credits_near_settlement":
            return inv.credits_near_settlement(params.get("record_id") or "")
        return inv.batch_summary()
    except Exception as exc:                          # noqa: BLE001
        return {"error": f"{action} failed: {exc}"}
