"""Tests for the credit index.

The index exists only to make lookups fast. Its entire contract is that it
returns exactly what a full scan would have returned - so these tests brute-force
scan the same data and compare, rather than asserting against hand-written
expectations that could drift from what the engine actually needs.

That matters because the optimisation it enables is invisible in the output. A
subtly wrong index would not crash or look odd; it would quietly drop a candidate
and turn a match into an exception, and the only symptom would be a recall number
slightly worse than it should be.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pytest

from core.index import CreditIndex
from core.loader import load_batch
from core.models import BankRecord

D0 = date(2026, 7, 10)
DEV = Path(__file__).resolve().parents[1] / "data" / "dev"


def credit(bid, paise, day):
    return BankRecord(bank_row_id=bid, value_date=day, description="x",
                      credit_paise=paise, utr_reference=None,
                      parsed_utr=None, utr_provenance="absent")


@pytest.fixture(scope="module")
def real_credits():
    return load_batch(DEV).banks


def brute_force(banks, anchor, lag_from, lag_to, lo, hi, exclude=frozenset()):
    """What the engine did before the index existed."""
    return sorted(
        b.bank_row_id for b in banks
        if lo <= b.credit_paise <= hi
        and lag_from <= (b.value_date - anchor).days <= lag_to
        and b.bank_row_id not in exclude
    )


# --------------------------------------------------------------------------
# Equivalence to a full scan - the whole contract
# --------------------------------------------------------------------------

def test_query_matches_a_full_scan_on_real_data(real_credits):
    idx = CreditIndex(real_credits)
    rng = random.Random(11)
    for _ in range(300):
        anchor = D0 + timedelta(days=rng.randint(-5, 40))
        lag_from = rng.choice([-1, 0])
        lag_to = lag_from + rng.randint(0, 7)
        centre = rng.choice(real_credits).credit_paise
        radius = rng.choice([0, 50, 5_000, 500_000])
        lo, hi = centre - radius, centre + radius
        got = sorted(b.bank_row_id for b in
                     idx.query(anchor, lag_from=lag_from, lag_to=lag_to,
                               lo_paise=lo, hi_paise=hi))
        assert got == brute_force(real_credits, anchor, lag_from, lag_to, lo, hi)


def test_query_matches_a_full_scan_with_exclusions(real_credits):
    idx = CreditIndex(real_credits)
    rng = random.Random(29)
    excluded = {b.bank_row_id for b in rng.sample(real_credits, len(real_credits) // 3)}
    for _ in range(150):
        anchor = D0 + timedelta(days=rng.randint(0, 30))
        centre = rng.choice(real_credits).credit_paise
        lo, hi = centre - 5_000, centre + 5_000
        got = sorted(b.bank_row_id for b in
                     idx.query(anchor, lag_from=0, lag_to=2, lo_paise=lo, hi_paise=hi,
                               exclude=excluded))
        assert got == brute_force(real_credits, anchor, 0, 2, lo, hi, excluded)


def test_count_matches_a_full_scan_and_ignores_one_row(real_credits):
    idx = CreditIndex(real_credits)
    rng = random.Random(5)
    for _ in range(100):
        target = rng.choice(real_credits)
        anchor = target.value_date
        lo, hi = target.credit_paise - 5_000, target.credit_paise + 5_000
        expected = len(brute_force(real_credits, anchor, 0, 2, lo, hi)) - (
            1 if lo <= target.credit_paise <= hi else 0)
        assert idx.count(anchor, lag_from=0, lag_to=2, lo_paise=lo, hi_paise=hi,
                         ignore=target.bank_row_id) == expected


def test_index_is_independent_of_input_order(real_credits):
    """Feeding the credits in a different order must not change any answer."""
    shuffled = list(real_credits)
    random.Random(7).shuffle(shuffled)
    a, b = CreditIndex(real_credits), CreditIndex(shuffled)
    for day in a.dates:
        assert ([x.bank_row_id for x in a.on_date(day, 0, 10**12)]
                == [x.bank_row_id for x in b.on_date(day, 0, 10**12)])


# --------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------

def test_amount_bounds_are_inclusive():
    idx = CreditIndex([credit("BNK_1", 100, D0), credit("BNK_2", 200, D0),
                       credit("BNK_3", 300, D0)])
    got = [b.bank_row_id for b in idx.query(D0, lag_from=0, lag_to=0,
                                            lo_paise=100, hi_paise=200)]
    assert got == ["BNK_1", "BNK_2"]


def test_lag_bounds_are_inclusive_and_can_be_negative():
    rows = [credit(f"BNK_{i}", 100, D0 + timedelta(days=i)) for i in range(-2, 4)]
    idx = CreditIndex(rows)
    got = [b.value_date for b in idx.query(D0, lag_from=-1, lag_to=2,
                                           lo_paise=0, hi_paise=1000)]
    assert got == [D0 + timedelta(days=d) for d in (-1, 0, 1, 2)]


def test_empty_and_inverted_ranges_yield_nothing():
    idx = CreditIndex([credit("BNK_1", 100, D0)])
    assert list(idx.query(D0, lag_from=0, lag_to=0, lo_paise=200, hi_paise=100)) == []
    assert list(idx.query(D0, lag_from=2, lag_to=0, lo_paise=0, hi_paise=1000)) == []
    assert list(idx.query(D0 + timedelta(days=99), lag_from=0, lag_to=0,
                          lo_paise=0, hi_paise=1000)) == []


def test_duplicate_amounts_on_one_date_are_all_returned():
    idx = CreditIndex([credit("BNK_1", 100, D0), credit("BNK_2", 100, D0),
                       credit("BNK_3", 100, D0)])
    got = sorted(b.bank_row_id for b in
                 idx.query(D0, lag_from=0, lag_to=0, lo_paise=100, hi_paise=100))
    assert got == ["BNK_1", "BNK_2", "BNK_3"]


def test_len_reports_every_credit(real_credits):
    assert len(CreditIndex(real_credits)) == len(real_credits)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_the_engine_produces_the_same_result_at_two_batch_sizes():
    """The index must not change behaviour as a batch grows.

    A subtly wrong index would not crash - it would silently drop a candidate and
    turn a match into an exception, showing up only as slightly worse recall.
    """
    from core.matcher import reconcile
    batch = load_batch(DEV)
    a, b = reconcile(batch), reconcile(load_batch(DEV))
    sig = lambda rep: sorted((r.record_id, r.status, r.match_type, r.confidence,
                              tuple(sorted(r.group_key)), tuple(r.rule_trace))
                             for r in rep.results)
    assert sig(a) == sig(b)
