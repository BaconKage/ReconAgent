"""Unit tests for the deterministic matching engine.

These build records by hand rather than reading the generated dataset. The
generated data proves the engine works on a realistic batch; these prove it
behaves exactly as specified at the boundaries, where "roughly right" is the
difference between a match and a false positive.

No LLM is involved in anything tested here, which is the point: the engine's
correctness is fully verifiable without a network call or an API key.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.config import MatchConfig
from core.loader import LoadedBatch
from core.matcher import reconcile
from core.models import BankRecord, LedgerRecord, SettlementRecord
from core.normalize import recover_utr

D0 = date(2026, 7, 10)
CFG = MatchConfig()


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def settlement(tid="pay_1", net=100_000, day=D0, utr=None, order_id=None, gross=None):
    gross = gross if gross is not None else net
    return SettlementRecord(
        transaction_id=tid, order_id=order_id, gross_paise=gross, fee_paise=0,
        tax_on_fee_paise=gross - net, net_paise=net, settlement_date=day,
        utr_number=utr,
    )


def bank(bid="BNK_1", credit=100_000, day=D0, utr=None, description="NEFT-RAZORPAY-SETTLEMENT"):
    parsed, prov = (utr, "test_fixture") if utr else (None, "absent")
    return BankRecord(
        bank_row_id=bid, value_date=day, description=description, credit_paise=credit,
        utr_reference=("UTR" + utr) if utr else None, parsed_utr=parsed, utr_provenance=prov,
    )


def ledger(order_id="order_1", expected=100_000, day=D0, status="paid"):
    return LedgerRecord(order_id=order_id, customer="Test Customer",
                        expected_paise=expected, order_date=day, status=status)


def batch(settlements=(), banks=(), ledgers=()):
    b = LoadedBatch()
    b.settlements = list(settlements)
    b.banks = list(banks)
    b.ledger = {l.order_id: l for l in ledgers}
    return b


def result_for(report, record_id):
    return next(r for r in report.results if r.record_id == record_id)


# --------------------------------------------------------------------------
# Tier 1 - exact UTR
# --------------------------------------------------------------------------

def test_exact_utr_match():
    rep = reconcile(batch([settlement(utr="123456789012")],
                          [bank(utr="123456789012")]))
    r = result_for(rep, "pay_1")
    assert r.status == "matched"
    assert r.match_type == "exact_utr"
    assert r.confidence == 1.0
    assert r.linked_ids["bank"] == ["BNK_1"]


def test_utr_match_with_amount_discrepancy_is_an_exception_not_a_match():
    """A partial refund: the identifier is right, the money is not.

    Binding these as 'matched' would silently under-report a shortfall, which is
    the exact failure a finance controller cannot tolerate.
    """
    rep = reconcile(batch([settlement(net=100_000, utr="123456789012")],
                          [bank(credit=70_000, utr="123456789012")]))
    r = result_for(rep, "pay_1")
    assert r.status == "unresolved"
    assert r.exception_reason == "identifier_match_amount_discrepancy"
    # The credit is still linked - the UTR proves they belong together.
    assert r.linked_ids["bank"] == ["BNK_1"]
    assert r.near_misses[0].amount_delta_paise == -30_000


def test_utr_matched_credit_is_not_offered_to_other_settlements():
    """A credit claimed by its UTR owner must not be re-matched by amount."""
    rep = reconcile(batch(
        [settlement("pay_1", net=100_000, utr="111111111111"),
         settlement("pay_2", net=70_000, utr="222222222222")],
        [bank("BNK_1", credit=70_000, utr="111111111111")],
    ))
    r2 = result_for(rep, "pay_2")
    assert r2.status == "unresolved"
    assert "BNK_1" not in r2.linked_ids["bank"]


def test_several_credits_sharing_one_utr_sum_to_net():
    rep = reconcile(batch([settlement(net=100_000, utr="123456789012")],
                          [bank("BNK_1", 60_000, utr="123456789012"),
                           bank("BNK_2", 40_000, utr="123456789012")]))
    r = result_for(rep, "pay_1")
    assert r.status == "matched_split"
    assert set(r.linked_ids["bank"]) == {"BNK_1", "BNK_2"}


# --------------------------------------------------------------------------
# Tier 2 - repaired / truncated UTR
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "UTR123456789012", "utr123456789012", "UTR-123456789012",
    "UTR 123456789012", "123456789012",
])
def test_utr_normalisation_variants_all_recover(raw):
    digits, _ = recover_utr(raw, None)
    assert digits == "123456789012"


def test_truncated_utr_resolves_by_unique_prefix():
    rep = reconcile(batch([settlement(net=100_000, utr="123456789012")],
                          [bank(credit=100_000, utr="12345678901")]))
    r = result_for(rep, "pay_1")
    assert r.status == "matched"
    assert r.match_type == "repaired_utr"
    assert r.confidence == 0.95


def test_truncated_utr_matching_two_settlements_is_refused():
    """A truncated identifier that fits two candidates identifies neither."""
    rep = reconcile(batch(
        [settlement("pay_1", net=100_000, utr="123456789010"),
         settlement("pay_2", net=200_000, utr="123456789011")],
        [bank("BNK_1", credit=100_000, utr="12345678901")],
    ))
    r = result_for(rep, "pay_1")
    assert r.match_type != "repaired_utr"


def test_truncated_utr_with_wrong_amount_is_not_claimed():
    rep = reconcile(batch([settlement(net=100_000, utr="123456789012")],
                          [bank(credit=50_000, utr="12345678901")]))
    r = result_for(rep, "pay_1")
    assert r.status == "unresolved"
    assert "BNK_1" not in r.linked_ids["bank"]


# --------------------------------------------------------------------------
# Tier 3 - amount and date boundaries
# --------------------------------------------------------------------------

@pytest.mark.parametrize("delta,should_match", [
    (0, True), (50, True), (-50, True), (51, False), (-51, False),
])
def test_amount_tolerance_boundary(delta, should_match):
    rep = reconcile(batch([settlement(net=100_000)],
                          [bank(credit=100_000 + delta)]))
    r = result_for(rep, "pay_1")
    assert (r.status == "matched") is should_match, f"delta {delta}p"


@pytest.mark.parametrize("lag,should_match", [
    (0, True), (1, True), (2, True), (3, False), (-1, False),
])
def test_date_window_boundary(lag, should_match):
    """Money cannot reach the bank before it is settled, so negative lag is out."""
    rep = reconcile(batch([settlement(net=100_000, day=D0)],
                          [bank(credit=100_000, day=D0 + timedelta(days=lag))]))
    r = result_for(rep, "pay_1")
    assert (r.status == "matched") is should_match, f"lag T{lag:+d}"


def test_confidence_degrades_with_looser_fit():
    tight = reconcile(batch([settlement(net=100_000)], [bank(credit=100_000)]))
    loose = reconcile(batch([settlement(net=100_000, day=D0)],
                            [bank(credit=100_045, day=D0 + timedelta(days=2))]))
    assert result_for(tight, "pay_1").confidence > result_for(loose, "pay_1").confidence


# --------------------------------------------------------------------------
# The ambiguity guard - the false-positive defence
# --------------------------------------------------------------------------

def test_two_qualifying_candidates_are_both_refused():
    """The core anti-false-positive rule.

    Both credits satisfy amount and date. Picking the closer one would be right
    half the time and confident every time. The engine claims neither.
    """
    rep = reconcile(batch([settlement(net=100_000)],
                          [bank("BNK_1", credit=100_000),
                           bank("BNK_2", credit=100_002)]))
    r = result_for(rep, "pay_1")
    assert r.status == "unresolved"
    assert r.exception_reason == "ambiguous_candidates"
    assert r.linked_ids["bank"] == []
    assert {n.record_id for n in r.near_misses} == {"BNK_1", "BNK_2"}


def test_engine_does_not_break_ties_by_closest_amount():
    """Explicit: an exact-zero-delta candidate does NOT win over a near one."""
    rep = reconcile(batch([settlement(net=100_000)],
                          [bank("BNK_exact", credit=100_000),
                           bank("BNK_near", credit=100_049)]))
    assert result_for(rep, "pay_1").status == "unresolved"


def test_constraint_propagation_resolves_a_forced_pairing():
    """A settlement with only one option frees its neighbour.

    pay_A can only be BNK_1. Once that is bound, pay_B is left with a single
    candidate and becomes resolvable. Naive one-shot matching would refuse both.
    """
    rep = reconcile(batch(
        [settlement("pay_A", net=100_000), settlement("pay_B", net=100_030)],
        [bank("BNK_1", credit=100_000), bank("BNK_2", credit=100_060)],
    ))
    a, b = result_for(rep, "pay_A"), result_for(rep, "pay_B")
    assert a.status == "matched" and a.linked_ids["bank"] == ["BNK_1"]
    assert b.status == "matched" and b.linked_ids["bank"] == ["BNK_2"]


def test_two_settlements_contending_for_one_credit_are_both_refused():
    rep = reconcile(batch(
        [settlement("pay_A", net=100_000), settlement("pay_B", net=100_000)],
        [bank("BNK_1", credit=100_000)],
    ))
    for tid in ("pay_A", "pay_B"):
        r = result_for(rep, tid)
        assert r.status == "unresolved"
        assert r.exception_reason == "contested_candidate"
        assert r.linked_ids["bank"] == []


# --------------------------------------------------------------------------
# Tier 4 - splits
# --------------------------------------------------------------------------

def test_split_of_two_legs_is_reconstructed():
    rep = reconcile(batch(
        [settlement(net=100_000, day=D0)],
        [bank("BNK_1", 60_000, day=D0 + timedelta(days=1)),
         bank("BNK_2", 40_000, day=D0 + timedelta(days=2))],
    ))
    r = result_for(rep, "pay_1")
    assert r.status == "matched_split"
    assert set(r.linked_ids["bank"]) == {"BNK_1", "BNK_2"}


def test_split_legs_outside_the_window_are_not_used():
    rep = reconcile(batch(
        [settlement(net=100_000, day=D0)],
        [bank("BNK_1", 60_000, day=D0 + timedelta(days=1)),
         bank("BNK_2", 40_000, day=D0 + timedelta(days=9))],
    ))
    assert result_for(rep, "pay_1").status == "unresolved"


def test_ambiguous_split_with_a_decoy_subset_is_refused():
    """Two different subsets both reach the target, so neither is identified."""
    rep = reconcile(batch(
        [settlement(net=100_000, day=D0)],
        [bank("BNK_1", 60_000, day=D0 + timedelta(days=1)),
         bank("BNK_2", 40_000, day=D0 + timedelta(days=1)),
         bank("BNK_3", 30_000, day=D0 + timedelta(days=1)),
         bank("BNK_4", 70_000, day=D0 + timedelta(days=1))],
    ))
    r = result_for(rep, "pay_1")
    assert r.status == "unresolved"
    assert r.exception_reason == "ambiguous_split"
    assert r.linked_ids["bank"] == []


def test_split_respects_the_leg_cap():
    """Five equal legs exceed max_split_legs=4 and must not be assembled."""
    legs = [bank(f"BNK_{i}", 20_000, day=D0 + timedelta(days=1)) for i in range(5)]
    rep = reconcile(batch([settlement(net=100_000, day=D0)], legs))
    assert result_for(rep, "pay_1").status == "unresolved"


def test_contested_split_legs_refuse_both_settlements():
    cfg = MatchConfig(amount_tolerance_paise=50)
    rep = reconcile(batch(
        [settlement("pay_A", net=100_000, day=D0),
         settlement("pay_B", net=100_000, day=D0)],
        [bank("BNK_1", 60_000, day=D0 + timedelta(days=1)),
         bank("BNK_2", 40_000, day=D0 + timedelta(days=1))],
    ), cfg)
    statuses = {result_for(rep, t).status for t in ("pay_A", "pay_B")}
    assert statuses == {"unresolved"}


# --------------------------------------------------------------------------
# Duplicates and orphans
# --------------------------------------------------------------------------

def test_duplicate_settlement_rows_are_flagged_once():
    s = settlement(net=100_000, utr="123456789012")
    rep = reconcile(batch([s, s], [bank(credit=100_000, utr="123456789012")]))
    r = result_for(rep, "pay_1")
    assert r.status == "duplicate"
    assert r.exception_reason == "duplicate_settlement_row"
    assert r.linked_ids["bank"] == ["BNK_1"]


def test_duplicate_with_fresh_id_but_identical_fingerprint_is_caught():
    rep = reconcile(batch(
        [settlement("pay_1", net=100_000, utr="123456789012"),
         settlement("pay_2", net=100_000, utr="123456789012")],
        [bank(credit=100_000, utr="123456789012")],
    ))
    assert {result_for(rep, t).status for t in ("pay_1", "pay_2")} == {"duplicate"}


def test_orphan_bank_row_becomes_an_exception():
    rep = reconcile(batch([], [bank("BNK_9", credit=55_000)]))
    r = result_for(rep, "BNK_9")
    assert r.status == "unresolved"
    assert r.exception_reason == "no_settlement_counterpart"


def test_settlement_with_no_counterpart_records_near_misses():
    """Outside both thresholds, but close enough to be worth explaining."""
    rep = reconcile(batch(
        [settlement(net=100_000, day=D0)],
        [bank("BNK_1", credit=100_095, day=D0 + timedelta(days=5))],
    ))
    r = result_for(rep, "pay_1")
    assert r.status == "unresolved"
    assert r.exception_reason == "no_candidate_found"
    assert r.near_misses and r.near_misses[0].record_id == "BNK_1"
    assert "amount off by" in r.near_misses[0].reason


def test_settlement_with_nothing_nearby_records_no_near_miss():
    rep = reconcile(batch([settlement(net=100_000)], [bank(credit=1_000_000)]))
    assert result_for(rep, "pay_1").near_misses == []


# --------------------------------------------------------------------------
# Contracts the rest of the system relies on
# --------------------------------------------------------------------------

def test_ledger_is_linked_by_exact_order_id():
    rep = reconcile(batch([settlement(utr="123456789012", order_id="order_1")],
                          [bank(utr="123456789012")], [ledger("order_1")]))
    assert result_for(rep, "pay_1").linked_ids["ledger"] == ["order_1"]


def test_every_bank_row_appears_in_exactly_one_group():
    """No credit may be double-counted or silently dropped."""
    b = batch(
        [settlement("pay_1", net=100_000, utr="111111111111"),
         settlement("pay_2", net=50_000)],
        [bank("BNK_1", 100_000, utr="111111111111"),
         bank("BNK_2", 50_000), bank("BNK_3", 999_999)],
    )
    rep = reconcile(b)
    seen = [bid for r in rep.results for bid in r.linked_ids["bank"]]
    assert sorted(seen) == ["BNK_1", "BNK_2", "BNK_3"]
    assert len(seen) == len(set(seen))


def test_engine_is_deterministic():
    b = lambda: batch(
        [settlement("pay_1", net=100_000, utr="111111111111"),
         settlement("pay_2", net=100_000), settlement("pay_3", net=100_001)],
        [bank("BNK_1", 100_000, utr="111111111111"),
         bank("BNK_2", 100_000), bank("BNK_3", 100_001)],
    )
    a = reconcile(b())
    c = reconcile(b())
    assert [(r.record_id, r.status, r.confidence, sorted(r.group_key)) for r in a.results] \
        == [(r.record_id, r.status, r.confidence, sorted(r.group_key)) for r in c.results]


def test_thresholds_are_honoured_from_config_not_hardcoded():
    """Widening tolerance must change the outcome, or the config is decorative."""
    data = lambda: batch([settlement(net=100_000)], [bank(credit=100_200)])
    assert reconcile(data()).results[0].status == "unresolved"
    wide = reconcile(data(), MatchConfig(amount_tolerance_paise=500))
    assert result_for(wide, "pay_1").status == "matched"
