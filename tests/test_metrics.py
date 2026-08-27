"""Tests for the evaluator itself.

A perfect score is only meaningful if the scorer is capable of reporting an
imperfect one. These tests feed the evaluator deliberately wrong reconciliation
output and assert it catches each failure mode - so that when the real run
reports zero false positives, that is a finding rather than a blind spot.

Every scenario here is a plausible way a matcher goes wrong in production:
binding the wrong counterpart, forcing a match that should have been held,
assembling a split from the wrong legs, and merging two unrelated transactions
that happen to look alike.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import DEFAULT_CONFIG
from core.matcher import ReconciliationReport
from core.models import MatchResult
from evaluation.metrics import evaluate

# --------------------------------------------------------------------------
# A tiny synthetic world: two matchable groups, one that must be held,
# and one adversarial pair.
# --------------------------------------------------------------------------

GROUND_TRUTH = {
    "meta": {"seed": 0, "counts": {}},
    "groups": [
        {"group_id": "G1", "case_type": "clean", "expected_resolution": "matched",
         "ledger_order_ids": ["order_1"], "settlement_txn_ids": ["pay_1"],
         "bank_row_ids": ["BNK_1"], "notes": ""},
        {"group_id": "G2", "case_type": "split_settlement",
         "expected_resolution": "matched_split",
         "ledger_order_ids": ["order_2"], "settlement_txn_ids": ["pay_2"],
         "bank_row_ids": ["BNK_2", "BNK_3"], "notes": ""},
        {"group_id": "G3", "case_type": "unmatchable",
         "expected_resolution": "exception_unmatchable",
         "ledger_order_ids": ["order_3"], "settlement_txn_ids": ["pay_3"],
         "bank_row_ids": [], "notes": ""},
        {"group_id": "G4", "case_type": "adversarial_ambiguous",
         "expected_resolution": "exception_ambiguous",
         "ledger_order_ids": ["order_4"], "settlement_txn_ids": ["pay_4"],
         "bank_row_ids": ["BNK_4"], "notes": ""},
        {"group_id": "G5", "case_type": "adversarial_ambiguous",
         "expected_resolution": "exception_ambiguous",
         "ledger_order_ids": ["order_5"], "settlement_txn_ids": ["pay_5"],
         "bank_row_ids": ["BNK_5"], "notes": ""},
    ],
    "adversarial_pairs": [["G4", "G5"]],
    "unmatchable_groups": ["G3"],
}


@pytest.fixture
def gt_dir(tmp_path):
    with open(tmp_path / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(GROUND_TRUTH, f)
    return tmp_path


def result(record_id, status, *, bank=(), ledger=(), settlement=None, reason=None):
    r = MatchResult(record_id=record_id, status=status, exception_reason=reason)
    r.linked_ids = {
        "settlement": [settlement or record_id] if not record_id.startswith("BNK") else [],
        "bank": list(bank),
        "ledger": list(ledger),
    }
    return r


def report_of(*results):
    return ReconciliationReport(results=list(results), config=DEFAULT_CONFIG,
                                elapsed_seconds=0.001, rows_processed=12)


def perfect():
    return report_of(
        result("pay_1", "matched", bank=["BNK_1"], ledger=["order_1"]),
        result("pay_2", "matched_split", bank=["BNK_2", "BNK_3"], ledger=["order_2"]),
        result("pay_3", "unresolved", ledger=["order_3"], reason="no_candidate_found"),
        result("pay_4", "unresolved", ledger=["order_4"], reason="ambiguous_candidates"),
        result("pay_5", "unresolved", ledger=["order_5"], reason="ambiguous_candidates"),
        result("BNK_4", "unresolved", bank=["BNK_4"], reason="no_settlement_counterpart"),
        result("BNK_5", "unresolved", bank=["BNK_5"], reason="no_settlement_counterpart"),
    )


# --------------------------------------------------------------------------
# Baseline: the evaluator can report a perfect run
# --------------------------------------------------------------------------

def test_perfect_reconciliation_scores_perfectly(gt_dir):
    ev = evaluate(perfect(), gt_dir)
    assert ev.true_positives == 2
    assert ev.false_positives == 0
    assert ev.false_negatives == 0
    assert ev.precision == 1.0 and ev.recall == 1.0
    assert ev.adversarial_conflated == 0


# --------------------------------------------------------------------------
# ... and, more importantly, an imperfect one
# --------------------------------------------------------------------------

def test_wrong_counterpart_is_caught_as_a_false_positive(gt_dir):
    """pay_1 auto-matched, but to BNK_9 instead of BNK_1."""
    rep = report_of(
        result("pay_1", "matched", bank=["BNK_9"], ledger=["order_1"]),
        result("pay_2", "matched_split", bank=["BNK_2", "BNK_3"], ledger=["order_2"]),
        result("pay_3", "unresolved", ledger=["order_3"], reason="no_candidate_found"),
        result("pay_4", "unresolved", ledger=["order_4"], reason="ambiguous_candidates"),
        result("pay_5", "unresolved", ledger=["order_5"], reason="ambiguous_candidates"),
    )
    ev = evaluate(rep, gt_dir)
    assert ev.false_positives == 1, "binding the wrong credit must be a false positive"
    assert ev.false_negatives == 1, "and the correct link was also missed"
    assert ev.true_positives == 1
    assert ev.precision < 1.0
    assert any(d["kind"] == "wrong_counterpart" for d in ev.false_positive_detail)


def test_forcing_a_match_that_should_be_held_is_a_false_positive(gt_dir):
    """pay_3 has no counterpart, but the engine bound one anyway."""
    rep = report_of(
        result("pay_1", "matched", bank=["BNK_1"], ledger=["order_1"]),
        result("pay_2", "matched_split", bank=["BNK_2", "BNK_3"], ledger=["order_2"]),
        result("pay_3", "matched", bank=["BNK_9"], ledger=["order_3"]),
        result("pay_4", "unresolved", ledger=["order_4"], reason="ambiguous_candidates"),
        result("pay_5", "unresolved", ledger=["order_5"], reason="ambiguous_candidates"),
    )
    ev = evaluate(rep, gt_dir)
    assert ev.false_positives == 1
    assert any(d["kind"] == "forced_a_match_that_should_have_been_held"
               for d in ev.false_positive_detail)
    assert ev.exceptions_correctly_held < ev.exceptions_expected


def test_incomplete_split_is_not_partial_credit(gt_dir):
    """Two of three legs is a wrong group, not two-thirds of a right one."""
    rep = report_of(
        result("pay_1", "matched", bank=["BNK_1"], ledger=["order_1"]),
        result("pay_2", "matched_split", bank=["BNK_2"], ledger=["order_2"]),
        result("pay_3", "unresolved", ledger=["order_3"], reason="no_candidate_found"),
        result("pay_4", "unresolved", ledger=["order_4"], reason="ambiguous_candidates"),
        result("pay_5", "unresolved", ledger=["order_5"], reason="ambiguous_candidates"),
    )
    ev = evaluate(rep, gt_dir)
    assert ev.true_positives == 1, "the incomplete split must not count as correct"
    assert ev.false_positives == 1
    assert ev.partial_membership == 1


def test_missing_a_matchable_group_is_a_false_negative_only(gt_dir):
    """Conservative failure: held back something it could have matched."""
    rep = report_of(
        result("pay_1", "unresolved", ledger=["order_1"], reason="no_candidate_found"),
        result("pay_2", "matched_split", bank=["BNK_2", "BNK_3"], ledger=["order_2"]),
        result("pay_3", "unresolved", ledger=["order_3"], reason="no_candidate_found"),
        result("pay_4", "unresolved", ledger=["order_4"], reason="ambiguous_candidates"),
        result("pay_5", "unresolved", ledger=["order_5"], reason="ambiguous_candidates"),
    )
    ev = evaluate(rep, gt_dir)
    assert ev.false_negatives == 1
    assert ev.false_positives == 0, "refusing must never count as a false positive"
    assert ev.recall < 1.0 and ev.precision == 1.0


def test_conflating_an_adversarial_pair_is_detected(gt_dir):
    """The failure the whole refusal design exists to prevent."""
    merged = MatchResult(record_id="pay_4", status="matched")
    merged.linked_ids = {"settlement": ["pay_4", "pay_5"],
                         "bank": ["BNK_4"], "ledger": ["order_4", "order_5"]}
    rep = report_of(
        result("pay_1", "matched", bank=["BNK_1"], ledger=["order_1"]),
        result("pay_2", "matched_split", bank=["BNK_2", "BNK_3"], ledger=["order_2"]),
        result("pay_3", "unresolved", ledger=["order_3"], reason="no_candidate_found"),
        merged,
    )
    ev = evaluate(rep, gt_dir)
    assert ev.adversarial_conflated == 1
    assert ev.adversarial_fp_rate == 1.0
    assert any(d["kind"] == "two_unrelated_transactions_merged"
               for d in ev.false_positive_detail)


def test_miscategorised_exception_is_counted_separately(gt_dir):
    """Held correctly, but for a reason that misdescribes what happened."""
    rep = report_of(
        result("pay_1", "matched", bank=["BNK_1"], ledger=["order_1"]),
        result("pay_2", "matched_split", bank=["BNK_2", "BNK_3"], ledger=["order_2"]),
        result("pay_3", "unresolved", ledger=["order_3"], reason="no_candidate_found"),
        result("pay_4", "unresolved", ledger=["order_4"], reason="no_candidate_found"),
        result("pay_5", "unresolved", ledger=["order_5"], reason="ambiguous_candidates"),
    )
    ev = evaluate(rep, gt_dir)
    assert ev.exceptions_correctly_held == 3
    assert ev.exceptions_correctly_categorised == 2, (
        "pay_4 was correctly held but labelled with the wrong reason")
    assert ev.category_accuracy < 1.0


# --------------------------------------------------------------------------
# Reporting invariants
# --------------------------------------------------------------------------

def test_throughput_numbers_are_reported_separately(gt_dir):
    ev = evaluate(perfect(), gt_dir, reasoning_seconds=4.0)
    assert ev.deterministic_rps > ev.end_to_end_rps
    assert ev.end_to_end_seconds == pytest.approx(0.001 + 4.0)


def test_report_renders_without_a_batch_or_reasoning(gt_dir):
    from evaluation.metrics import format_report
    text = format_report(evaluate(perfect(), gt_dir))
    assert "auto-match rate" in text
    assert "adversarial false-positive rate" in text


def test_counts_partition_every_group(gt_dir):
    """Every ground-truth group lands in exactly one classification bucket."""
    for rep in (perfect(),):
        ev = evaluate(rep, gt_dir)
        total = (ev.true_positives + ev.false_negatives + ev.true_negatives
                 + sum(1 for d in ev.false_positive_detail
                       if d["kind"] == "forced_a_match_that_should_have_been_held"))
        assert total == ev.total_groups
