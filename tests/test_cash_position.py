"""Tests for the forward cash position.

The distinction this layer exists to draw is between money that has not arrived
*yet* and money that has not arrived *and should have*. Getting that wrong in
either direction is expensive: call a T+1 settlement overdue and you send someone
chasing a bank for money that is in transit; call a two-week-old settlement
in-flight and a genuine loss sits unnoticed on the books.

Every rupee in the batch must land in exactly one bucket, so the totals are also
asserted to reconcile.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from cash.position import compute_position
from core.config import MatchConfig
from core.loader import LoadedBatch, load_batch
from core.matcher import reconcile
from core.models import BankRecord, LedgerRecord, SettlementRecord
from pathlib import Path

D0 = date(2026, 7, 10)
DEV = Path(__file__).resolve().parents[1] / "data" / "dev"


def settlement(tid, net, day, utr=None, order_id=None):
    return SettlementRecord(tid, order_id, net, 0, 0, net, day, utr)


def bank(bid, credit, day, utr=None):
    return BankRecord(bid, day, "NEFT-RAZORPAY-SETTLEMENT", credit,
                      ("UTR" + utr) if utr else None, utr, "test" if utr else "absent")


def batch_of(settlements=(), banks=(), ledgers=()):
    b = LoadedBatch()
    b.settlements = list(settlements)
    b.banks = list(banks)
    b.ledger = {l.order_id: l for l in ledgers}
    return b


# --------------------------------------------------------------------------
# The in-transit / overdue distinction
# --------------------------------------------------------------------------

def test_settlement_inside_the_window_is_in_flight_not_overdue():
    """Settled today, credit not due until T+2. Nothing is wrong here."""
    b = batch_of([settlement("pay_1", 100_000, D0)])
    pos = compute_position(reconcile(b), b, as_of=D0)
    assert pos.in_flight_paise == 100_000
    assert pos.overdue_paise == 0
    assert len(pos.in_flight) == 1


def test_settlement_exactly_at_the_window_edge_is_still_in_flight():
    b = batch_of([settlement("pay_1", 100_000, D0)])
    pos = compute_position(reconcile(b), b, as_of=D0 + timedelta(days=2))
    assert pos.in_flight_paise == 100_000
    assert pos.overdue_paise == 0


def test_settlement_past_the_window_is_overdue():
    b = batch_of([settlement("pay_1", 100_000, D0)])
    pos = compute_position(reconcile(b), b, as_of=D0 + timedelta(days=3))
    assert pos.overdue_paise == 100_000
    assert pos.in_flight_paise == 0
    assert pos.overdue[0].days_outstanding == 3


def test_window_comes_from_config():
    b = batch_of([settlement("pay_1", 100_000, D0)])
    as_of = D0 + timedelta(days=4)
    assert compute_position(reconcile(b), b, as_of=as_of).overdue_paise == 100_000
    wide = MatchConfig(date_window_days=5)
    pos = compute_position(reconcile(b, wide), b, as_of=as_of, cfg=wide)
    assert pos.in_flight_paise == 100_000


def test_as_of_defaults_to_the_last_settlement_date_not_the_last_credit():
    """Otherwise a late straggler credit would age the whole book artificially."""
    b = batch_of(
        [settlement("pay_1", 100_000, D0, utr="111111111111")],
        [bank("BNK_1", 100_000, D0 + timedelta(days=30), utr="111111111111")],
    )
    assert compute_position(reconcile(b), b).as_of == D0


# --------------------------------------------------------------------------
# Bucket attribution
# --------------------------------------------------------------------------

def test_matched_money_is_confirmed():
    b = batch_of([settlement("pay_1", 100_000, D0, utr="111111111111")],
                 [bank("BNK_1", 100_000, D0, utr="111111111111")])
    pos = compute_position(reconcile(b), b, as_of=D0)
    assert pos.confirmed_paise == 100_000
    assert pos.explained_fraction == 1.0
    assert pos.at_risk_paise == 0


def test_orphan_credit_is_unattributed_not_held_for_review():
    """Different problem, different remedy: identify the payment, not explain a gap."""
    b = batch_of([], [bank("BNK_9", 55_000, D0)])
    pos = compute_position(reconcile(b), b, as_of=D0)
    assert pos.unattributed_paise == 55_000
    assert pos.held_for_review_paise == 0


def test_amount_discrepancy_is_held_for_review_with_the_gap_recorded():
    """A partial refund: the credit arrived, it is just short."""
    b = batch_of([settlement("pay_1", 100_000, D0, utr="111111111111")],
                 [bank("BNK_1", 70_000, D0, utr="111111111111")])
    pos = compute_position(reconcile(b), b, as_of=D0)
    assert pos.held_for_review_paise == 70_000, "the money that did arrive is in the account"
    assert pos.discrepancy_paise == -30_000, "and the shortfall is quantified"
    assert pos.confirmed_paise == 0


def test_money_in_the_account_excludes_what_has_not_arrived():
    b = batch_of(
        [settlement("pay_1", 100_000, D0, utr="111111111111"),
         settlement("pay_2", 50_000, D0)],
        [bank("BNK_1", 100_000, D0, utr="111111111111"), bank("BNK_9", 7_000, D0)],
    )
    pos = compute_position(reconcile(b), b, as_of=D0)
    assert pos.in_account_paise == 107_000
    assert pos.receivable_paise == 50_000


# --------------------------------------------------------------------------
# Totals must reconcile on the real batch
# --------------------------------------------------------------------------

def test_every_credit_is_attributed_exactly_once():
    """The three in-account buckets must sum to the actual bank total."""
    b = load_batch(DEV)
    pos = compute_position(reconcile(b), b)
    assert pos.in_account_paise == sum(x.credit_paise for x in b.banks)


def test_at_risk_is_the_sum_of_what_needs_a_human():
    b = load_batch(DEV)
    pos = compute_position(reconcile(b), b)
    assert pos.at_risk_paise == (pos.overdue_paise + pos.held_for_review_paise
                                 + pos.unattributed_paise)
    assert pos.at_risk_paise > 0


def test_real_batch_has_both_in_flight_and_overdue():
    """If everything collapses into one bucket the distinction is untested."""
    b = load_batch(DEV)
    pos = compute_position(reconcile(b), b)
    assert pos.in_flight_paise > 0 and pos.overdue_paise > 0
    assert pos.schedule, "in-flight money should carry an expected arrival schedule"


def test_schedule_dates_are_all_in_the_future_relative_to_as_of():
    b = load_batch(DEV)
    pos = compute_position(reconcile(b), b)
    for d in pos.schedule:
        assert d >= pos.as_of


def test_position_does_not_mutate_the_reconciliation_report():
    b = load_batch(DEV)
    rep = reconcile(b)
    before = [(r.record_id, r.status, r.confidence) for r in rep.results]
    compute_position(rep, b)
    assert [(r.record_id, r.status, r.confidence) for r in rep.results] == before
