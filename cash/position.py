"""Forward cash position derived from a reconciliation run.

Reconciliation answers "do the books agree". A finance controller also has to
answer "where is the money" - what has actually landed, what is still expected,
what is late, and what arrived that nobody can account for. This module turns the
reconciliation result into that view.

Nothing here re-decides a match. It reads the engine's output and attributes
value to one of five buckets, which are chosen so that they sum to something a
treasurer would recognise:

    confirmed        credits reconciled to a settlement - money in the bank, explained
    held_for_review  credits that arrived but do not agree with their settlement
    unattributed     credits in the account with no settlement behind them
    in_flight        settled, no credit yet, still inside the expected window
    overdue          settled, no credit yet, past the window - the actual risk

The first three are money that is *in the account today*. The last two are money
that is *not*, split by whether its absence is yet a problem. Keeping "late" and
"merely pending" apart is the entire point: a settlement at T+1 with no credit is
normal, and the same settlement at T+9 is an incident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from core.config import DEFAULT_CONFIG, MatchConfig
from core.loader import LoadedBatch
from core.matcher import ReconciliationReport
from core.normalize import format_inr

MATCHED = {"matched", "matched_split", "duplicate"}


@dataclass
class PendingItem:
    transaction_id: str
    order_id: str | None
    customer: str
    net_paise: int
    settlement_date: date
    days_outstanding: int
    expected_by: date
    reason: str

    @property
    def days_late(self) -> int:
        return max(0, self.days_outstanding - (self.expected_by - self.settlement_date).days)


@dataclass
class CashPosition:
    as_of: date
    window_days: int

    confirmed_paise: int = 0
    held_for_review_paise: int = 0
    unattributed_paise: int = 0
    in_flight_paise: int = 0
    overdue_paise: int = 0

    #: Signed gap between what was settled and what was credited, on rows where
    #: both exist but disagree. Negative means the bank credited less.
    discrepancy_paise: int = 0

    in_flight: list[PendingItem] = field(default_factory=list)
    overdue: list[PendingItem] = field(default_factory=list)
    #: Expected arrival schedule for in-flight money, keyed by date.
    schedule: dict[date, int] = field(default_factory=dict)

    @property
    def in_account_paise(self) -> int:
        """Everything actually sitting in the bank account for this batch."""
        return self.confirmed_paise + self.held_for_review_paise + self.unattributed_paise

    @property
    def receivable_paise(self) -> int:
        return self.in_flight_paise + self.overdue_paise

    @property
    def explained_fraction(self) -> float:
        """Share of money in the account that is fully reconciled."""
        return self.confirmed_paise / self.in_account_paise if self.in_account_paise else 1.0

    @property
    def at_risk_paise(self) -> int:
        """Money a controller should be chasing today.

        Overdue receivables plus credits that arrived but do not reconcile. Both
        need a human; neither can be signed off.
        """
        return self.overdue_paise + self.held_for_review_paise + self.unattributed_paise


def compute_position(report: ReconciliationReport, batch: LoadedBatch, *,
                     as_of: date | None = None,
                     cfg: MatchConfig = DEFAULT_CONFIG) -> CashPosition:
    """Attribute every rupee in the batch to a cash bucket.

    `as_of` defaults to the latest date appearing anywhere in the batch, which is
    the natural "today" for a settlement cycle being closed.
    """
    if as_of is None:
        # The books close on the last settlement date, not the last bank date.
        # Using the latest credit would push every settlement past its window
        # and report the whole book as overdue - an artefact of where the batch
        # happens to end, not a fact about the money.
        dates = [s.settlement_date for s in batch.settlements]
        as_of = max(dates) if dates else date.today()

    settlements = {s.transaction_id: s for s in batch.settlements}
    banks = {b.bank_row_id: b for b in batch.banks}

    pos = CashPosition(as_of=as_of, window_days=cfg.date_window_days)

    for r in report.results:
        linked_banks = [banks[b] for b in r.linked_ids.get("bank", []) if b in banks]
        credited = sum(b.credit_paise for b in linked_banks)
        s = settlements.get(r.record_id)

        if r.status in MATCHED:
            pos.confirmed_paise += credited
            continue

        # --- unresolved -------------------------------------------------
        if s is None:
            # No settlement behind this group at all: an orphan credit. Cash is
            # present, provenance unknown. Kept apart from held_for_review
            # because the remedy differs - one needs a discrepancy explained,
            # the other needs the payment identified at all.
            pos.unattributed_paise += credited
            continue

        if linked_banks:
            # Money arrived but does not agree with what was settled. It is in
            # the account, and it is not signed off.
            pos.held_for_review_paise += credited
            pos.discrepancy_paise += credited - s.net_paise
            continue

        # A settlement with no credit at all. Late, or merely pending?
        order = batch.ledger.get(s.order_id) if s.order_id else None
        outstanding = (as_of - s.settlement_date).days
        expected_by = s.settlement_date + timedelta(days=cfg.date_window_days)
        item = PendingItem(
            transaction_id=s.transaction_id,
            order_id=s.order_id,
            customer=order.customer if order else "(unknown)",
            net_paise=s.net_paise,
            settlement_date=s.settlement_date,
            days_outstanding=outstanding,
            expected_by=expected_by,
            reason=r.exception_reason or "unresolved",
        )
        if as_of <= expected_by:
            pos.in_flight_paise += s.net_paise
            pos.in_flight.append(item)
            pos.schedule[expected_by] = pos.schedule.get(expected_by, 0) + s.net_paise
        else:
            pos.overdue_paise += s.net_paise
            pos.overdue.append(item)

    pos.in_flight.sort(key=lambda i: i.expected_by)
    pos.overdue.sort(key=lambda i: -i.days_outstanding)
    return pos


def format_position(pos: CashPosition) -> str:
    L = ["=" * 78,
         f"CASH POSITION as of {pos.as_of.isoformat()}",
         "=" * 78, ""]
    L.append("  IN THE ACCOUNT")
    L.append(f"    reconciled and explained  {format_inr(pos.confirmed_paise):>16}")
    L.append(f"    arrived but unreconciled  {format_inr(pos.held_for_review_paise):>16}")
    L.append(f"    unattributed inflows      {format_inr(pos.unattributed_paise):>16}")
    L.append(f"    {'-' * 42}")
    L.append(f"    total in account          {format_inr(pos.in_account_paise):>16}   "
             f"({pos.explained_fraction:.0%} explained)")
    L.append("")
    L.append("  EXPECTED, NOT YET RECEIVED")
    L.append(f"    in flight (within T+{pos.window_days})     "
             f"{format_inr(pos.in_flight_paise):>16}   {len(pos.in_flight)} settlements")
    L.append(f"    overdue                   {format_inr(pos.overdue_paise):>16}   "
             f"{len(pos.overdue)} settlements")
    if pos.schedule:
        L.append("")
        L.append("    expected arrival schedule:")
        for d in sorted(pos.schedule):
            L.append(f"      {d.isoformat()}   {format_inr(pos.schedule[d]):>14}")
    if pos.overdue:
        L.append("")
        L.append("    oldest overdue items:")
        for item in pos.overdue[:5]:
            L.append(f"      {item.transaction_id}  {format_inr(item.net_paise):>13}  "
                     f"settled {item.settlement_date}  {item.days_outstanding}d outstanding"
                     f"  ({item.customer})")
    L.append("")
    L.append(f"  NEEDS A HUMAN TODAY         {format_inr(pos.at_risk_paise):>16}")
    L.append("    overdue receivables, unreconciled credits and unattributed inflows")
    if pos.unattributed_paise and pos.overdue_paise:
        L.append("")
        L.append(f"    Note: {format_inr(pos.unattributed_paise)} of unattributed credits sits")
        L.append(f"    alongside {format_inr(pos.overdue_paise)} of overdue receivables.")
        L.append("    These are plausibly the same money - which is exactly the pairing")
        L.append("    the engine declined to assert on the evidence available. Confirming")
        L.append("    it is a human's call, and now it is a short, specific one.")
    if pos.discrepancy_paise:
        L.append(f"    net settlement-vs-credit gap on reviewed rows: "
                 f"{format_inr(pos.discrepancy_paise)}")
    L.append("")
    L.append("=" * 78)
    return "\n".join(L)
