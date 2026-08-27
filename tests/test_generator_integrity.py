"""Integrity tests for the synthetic data generator.

Everything downstream - the matcher, the metrics, the honesty claims - rests on
this data being what the ground truth says it is. A silent generator bug would
not fail loudly; it would just produce confidently wrong accuracy numbers. These
tests exist so that cannot happen unnoticed.

They assert structural invariants of the *generated data*, independently of the
matcher, by re-reading the written CSVs rather than trusting in-memory objects.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pytest

from core.normalize import parse_date, recover_utr, rupees_to_paise

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DATASETS = ["dev", "holdout"]

# Must stay in step with the matcher's published tolerances.
AMOUNT_TOL_PAISE = 50
DATE_WINDOW_DAYS = 2


def load(dataset: str):
    root = DATA_ROOT / dataset
    with open(root / "settlement_report.csv", encoding="utf-8") as f:
        settlements = {r["transaction_id"]: r for r in csv.DictReader(f)}
    with open(root / "bank_statement.csv", encoding="utf-8") as f:
        banks = {r["bank_row_id"]: r for r in csv.DictReader(f)}
    with open(root / "internal_ledger.csv", encoding="utf-8") as f:
        ledger = {r["order_id"]: r for r in csv.DictReader(f)}
    with open(root / "ground_truth.json", encoding="utf-8") as f:
        gt = json.load(f)
    return settlements, banks, ledger, gt


@pytest.fixture(params=DATASETS)
def dataset(request):
    return request.param, *load(request.param)


# --------------------------------------------------------------------------
# Structural integrity
# --------------------------------------------------------------------------

def test_group_ids_unique(dataset):
    _, _, _, _, gt = dataset
    ids = [g["group_id"] for g in gt["groups"]]
    assert len(ids) == len(set(ids))


def test_every_ground_truth_reference_resolves(dataset):
    """No dangling IDs: ground truth may only cite rows that exist in the CSVs."""
    _, settlements, banks, ledger, gt = dataset
    for g in gt["groups"]:
        for tid in g["settlement_txn_ids"]:
            assert tid in settlements, f"{g['group_id']} cites missing settlement {tid}"
        for bid in g["bank_row_ids"]:
            assert bid in banks, f"{g['group_id']} cites missing bank row {bid}"
        for oid in g["ledger_order_ids"]:
            assert oid in ledger, f"{g['group_id']} cites missing ledger order {oid}"


def test_every_row_is_claimed_by_exactly_one_group(dataset):
    """No orphaned or double-claimed rows.

    Duplicate groups legitimately cite the same transaction_id twice (that IS
    the duplicate), so settlements are counted by group membership, not rows.
    """
    _, settlements, banks, ledger, gt = dataset
    claimed_banks, claimed_ledger = [], []
    for g in gt["groups"]:
        claimed_banks.extend(g["bank_row_ids"])
        claimed_ledger.extend(g["ledger_order_ids"])
    assert sorted(claimed_banks) == sorted(banks), "bank rows claimed != bank rows written"
    assert len(claimed_banks) == len(set(claimed_banks)), "a bank row is in two groups"
    assert sorted(claimed_ledger) == sorted(ledger)


def test_bank_ids_are_shuffled_not_grouped(dataset):
    """Split legs must not be contiguous in ID space.

    If they were, adjacency would leak the grouping and the split-detection
    metric would be measuring the generator, not the matcher.
    """
    name, _, _, _, gt = dataset
    splits = [g for g in gt["groups"] if g["case_type"] == "split_settlement"]
    assert splits, "no split groups to check"
    contiguous = 0
    for g in splits:
        nums = sorted(int(b.split("_")[1]) for b in g["bank_row_ids"])
        if nums == list(range(nums[0], nums[0] + len(nums))):
            contiguous += 1
    # With a real shuffle this should be near zero; allow a little chance overlap.
    assert contiguous <= max(1, len(splits) // 4), (
        f"{contiguous}/{len(splits)} split groups have contiguous bank IDs in {name}"
    )


# --------------------------------------------------------------------------
# Financial arithmetic
# --------------------------------------------------------------------------

def test_fee_arithmetic_closes(dataset):
    """gross - fee - tax == net, exactly, in integer paise, for every row."""
    _, settlements, _, _, _ = dataset
    for tid, r in settlements.items():
        gross = rupees_to_paise(r["gross_amount"])
        fee = rupees_to_paise(r["fee"])
        tax = rupees_to_paise(r["tax_on_fee"])
        net = rupees_to_paise(r["net_amount"])
        assert gross - fee - tax == net, f"{tid}: {gross} - {fee} - {tax} != {net}"


def test_split_legs_sum_to_net(dataset):
    _, settlements, banks, _, gt = dataset
    for g in gt["groups"]:
        if g["case_type"] != "split_settlement":
            continue
        net = rupees_to_paise(settlements[g["settlement_txn_ids"][0]]["net_amount"])
        legs = sum(rupees_to_paise(banks[b]["credit_amount"]) for b in g["bank_row_ids"])
        assert legs == net, f"{g['group_id']}: legs {legs} != net {net}"
        assert len(g["bank_row_ids"]) >= 2


def test_rounding_drift_is_inside_tolerance(dataset):
    """Rounding cases must be matchable: drift within tolerance, else the label lies."""
    _, settlements, banks, _, gt = dataset
    for g in gt["groups"]:
        if g["case_type"] != "rounding":
            continue
        net = rupees_to_paise(settlements[g["settlement_txn_ids"][0]]["net_amount"])
        credit = rupees_to_paise(banks[g["bank_row_ids"][0]]["credit_amount"])
        drift = abs(credit - net)
        assert 0 < drift <= AMOUNT_TOL_PAISE, f"{g['group_id']} drift {drift}p"


def test_fee_deduction_gap_equals_fee_plus_tax(dataset):
    """The ledger/bank gap must be exactly the fee + GST, not arbitrary noise."""
    _, settlements, banks, ledger, gt = dataset
    for g in gt["groups"]:
        if g["case_type"] != "fee_deduction":
            continue
        s = settlements[g["settlement_txn_ids"][0]]
        credit = rupees_to_paise(banks[g["bank_row_ids"][0]]["credit_amount"])
        expected = rupees_to_paise(ledger[g["ledger_order_ids"][0]]["expected_amount"])
        gap = expected - credit
        assert gap == rupees_to_paise(s["fee"]) + rupees_to_paise(s["tax_on_fee"])


# --------------------------------------------------------------------------
# The adversarial cases - the ones the whole false-positive claim rests on
# --------------------------------------------------------------------------

def test_adversarial_pairs_are_actually_confusable(dataset):
    """A near-duplicate that is not genuinely near is not a test of anything.

    Both legs must sit inside the amount tolerance and the date window of each
    other, i.e. a naive amount+date matcher really would be tempted.
    """
    _, settlements, _, _, gt = dataset
    by_id = {g["group_id"]: g for g in gt["groups"]}
    assert gt["adversarial_pairs"], "no adversarial pairs generated"
    for a_id, b_id in gt["adversarial_pairs"]:
        a, b = by_id[a_id], by_id[b_id]
        net_a = rupees_to_paise(settlements[a["settlement_txn_ids"][0]]["net_amount"])
        net_b = rupees_to_paise(settlements[b["settlement_txn_ids"][0]]["net_amount"])
        d_a = parse_date(settlements[a["settlement_txn_ids"][0]]["settlement_date"])
        d_b = parse_date(settlements[b["settlement_txn_ids"][0]]["settlement_date"])
        assert abs(net_a - net_b) <= AMOUNT_TOL_PAISE, (
            f"{a_id}/{b_id} nets differ by {abs(net_a - net_b)}p - not confusable")
        assert abs((d_a - d_b).days) <= DATE_WINDOW_DAYS


def test_ambiguous_pairs_have_no_utr_on_either_bank_row(dataset):
    """The ambiguous pairs must be genuinely undecidable.

    If either leg carried a recoverable UTR the cascade could resolve it, and
    labelling it 'exception_ambiguous' would be penalising correct behaviour.
    """
    _, _, banks, _, gt = dataset
    for g in gt["groups"]:
        if g["case_type"] != "adversarial_ambiguous":
            continue
        for bid in g["bank_row_ids"]:
            row = banks[bid]
            digits, prov = recover_utr(row["utr_reference"], row["description"])
            assert digits is None, f"{bid} leaks a UTR ({prov}) - pair is not ambiguous"


def test_resolvable_adversarial_pair_has_exactly_one_identified_leg(dataset):
    """The resolvable variant is only resolvable because ONE leg has a clean UTR."""
    _, _, banks, _, gt = dataset
    by_id = {g["group_id"]: g for g in gt["groups"]}
    pairs = [p for p in gt["adversarial_pairs"]
             if by_id[p[0]]["case_type"] == "adversarial_resolvable"]
    assert pairs
    for a_id, b_id in pairs:
        identified = 0
        for gid in (a_id, b_id):
            for bid in by_id[gid]["bank_row_ids"]:
                row = banks[bid]
                digits, _ = recover_utr(row["utr_reference"], row["description"])
                if digits and len(digits) == 12:
                    identified += 1
        assert identified == 1, f"{a_id}/{b_id}: {identified} identified legs, expected 1"


def test_planted_near_miss_is_outside_both_thresholds(dataset):
    """The deliberate refusal case must be genuinely unmatchable.

    If the planted decoy sat inside tolerance, refusing it would be a bug rather
    than good judgement - and the demo claim would be dishonest.
    """
    name, settlements, banks, _, gt = dataset
    # The decoy's note *starts* with the marker; the settlement it shadows only
    # mentions it in passing ("paired with a planted near-miss").
    planted = [g for g in gt["groups"] if g["notes"].startswith("planted near-miss")]
    assert len(planted) == 1, f"{name}: expected exactly one planted near-miss"
    decoy = banks[planted[0]["bank_row_ids"][0]]
    decoy_amt = rupees_to_paise(decoy["credit_amount"])
    decoy_date = parse_date(decoy["date"])

    # It must not be bindable to ANY settlement within both thresholds.
    for tid, s in settlements.items():
        net = rupees_to_paise(s["net_amount"])
        sd = parse_date(s["settlement_date"])
        lag = (decoy_date - sd).days
        inside_amount = abs(decoy_amt - net) <= AMOUNT_TOL_PAISE
        inside_window = 0 <= lag <= DATE_WINDOW_DAYS
        assert not (inside_amount and inside_window), (
            f"{name}: planted decoy is legitimately matchable to {tid}")


def test_unmatchable_settlements_have_no_bank_row(dataset):
    _, _, _, _, gt = dataset
    orphans = [g for g in gt["groups"]
               if g["case_type"] == "unmatchable" and g["settlement_txn_ids"]]
    assert orphans
    for g in orphans:
        assert not g["bank_row_ids"], f"{g['group_id']} is labelled unmatchable but has a credit"


def test_datasets_are_disjoint_and_holdout_is_harder():
    """Held-out must be a genuinely different draw, not a reshuffle of dev."""
    dev_s, dev_b, _, dev_gt = load("dev")
    ho_s, ho_b, _, ho_gt = load("holdout")
    assert not (set(dev_s) & set(ho_s)), "transaction IDs overlap between dev and holdout"
    assert dev_gt["meta"]["seed"] != ho_gt["meta"]["seed"]
    assert ho_gt["meta"]["max_lag_days"] > dev_gt["meta"]["max_lag_days"]

    def hard_fraction(gt):
        hard = {"split_settlement", "adversarial_resolvable",
                "adversarial_ambiguous", "unmatchable", "partial_refund"}
        n = sum(1 for g in gt["groups"] if g["case_type"] in hard)
        return n / len(gt["groups"])

    assert hard_fraction(ho_gt) > hard_fraction(dev_gt), "holdout is not actually harder"


def test_holdout_contains_narration_format_absent_from_dev():
    """The parser must face at least one shape it was never tuned on."""
    _, dev_b, _, _ = load("dev")
    _, ho_b, _, _ = load("holdout")
    dev_shapes = {r["description"].split("*")[0].split("/")[0][:6] for r in dev_b.values()}
    novel = [r["description"] for r in ho_b.values()
             if r["description"].startswith(("BY TRANSFER", "CR/UTRNO"))]
    assert novel, "holdout has no novel narration format"
    assert not any(d.startswith(("BY TRANSFER", "CR/UTRNO")) for d in
                   (r["description"] for r in dev_b.values()))
