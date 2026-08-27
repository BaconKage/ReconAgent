"""Evaluation against ground truth.

This module is the only one permitted to read `ground_truth.json`. The engine and
the reasoning layer never see it.

How reconciliation is scored
----------------------------
Reconciliation is two problems at once, and collapsing them into a single
percentage hides the thing that matters. So both are measured.

**The decision.** For every ground-truth group, should it have been auto-matched,
or held for a human? That is a binary classification, and the two error types are
not symmetric:

- A **false positive** means the engine auto-matched something it should have
  held - it bound money to the wrong counterpart, or forced a match that the data
  did not support. In reconciliation this is the expensive error: it produces a
  clean-looking book that is wrong, and nobody goes looking.
- A **false negative** means the engine held something it could have matched.
  This costs an operator a few minutes. It is the cheap error, and a system that
  refuses when uncertain will deliberately incur more of them.

Reporting a single accuracy number would let a system trade the expensive error
for the cheap one and look better for it. These are reported separately.

**The linking.** A predicted group counts as correct only when its member set
*exactly* equals a ground-truth group's. Getting two of three legs of a split
settlement is not two-thirds right; it is a wrong group, and it leaves a credit
stranded. Partial credit is reported alongside as a secondary figure, but the
headline is strict.

**The adversarial rate.** Near-duplicate pairs are counted separately, because
they are the specific case a naive matcher fails silently. A system that conflates
them still posts a high overall match rate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.matcher import ReconciliationReport
from core.models import MatchResult
from core.normalize import format_inr, rupees_to_paise

AUTO_MATCHABLE = {"matched", "matched_split"}

#: Which engine exception reasons are a correct diagnosis of which ground-truth
#: case. Used only for the categorisation figure - never for scoring the decision.
EXPECTED_REASONS = {
    "partial_refund": {"identifier_match_amount_discrepancy"},
    "duplicate": {"duplicate_settlement_row"},
    "adversarial_ambiguous": {"ambiguous_candidates", "contested_candidate"},
    "unmatchable": {"no_candidate_found", "no_settlement_counterpart"},
}


@dataclass
class CaseTypeBreakdown:
    case_type: str
    total: int = 0
    correct: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class EvaluationResult:
    dataset: str = ""
    total_groups: int = 0
    total_rows: int = 0

    # Decision-level classification
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    # Linking
    groups_claimed: int = 0
    exact_membership: int = 0
    partial_membership: int = 0

    # Adversarial
    adversarial_pairs: int = 0
    adversarial_conflated: int = 0

    # Exceptions
    exceptions_expected: int = 0
    exceptions_correctly_held: int = 0
    exceptions_correctly_categorised: int = 0

    # Throughput
    deterministic_seconds: float = 0.0
    end_to_end_seconds: float | None = None
    llm_free_groups: int = 0

    # Money
    paise_auto_matched: int = 0
    paise_in_exceptions: int = 0

    by_case_type: dict[str, CaseTypeBreakdown] = field(default_factory=dict)
    exception_reasons: dict[str, int] = field(default_factory=dict)
    false_positive_detail: list[dict[str, Any]] = field(default_factory=list)

    # -- derived -------------------------------------------------------

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def auto_match_rate(self) -> float:
        """Share of all groups the engine resolved without human involvement."""
        return self.groups_claimed / self.total_groups if self.total_groups else 0.0

    @property
    def adversarial_fp_rate(self) -> float:
        return (self.adversarial_conflated / self.adversarial_pairs
                if self.adversarial_pairs else 0.0)

    @property
    def exception_hold_rate(self) -> float:
        return (self.exceptions_correctly_held / self.exceptions_expected
                if self.exceptions_expected else 0.0)

    @property
    def category_accuracy(self) -> float:
        return (self.exceptions_correctly_categorised / self.exceptions_correctly_held
                if self.exceptions_correctly_held else 0.0)

    @property
    def deterministic_rps(self) -> float:
        return self.total_rows / self.deterministic_seconds if self.deterministic_seconds else 0.0

    @property
    def end_to_end_rps(self) -> float | None:
        if not self.end_to_end_seconds:
            return None
        return self.total_rows / self.end_to_end_seconds


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def load_ground_truth(dataset_dir: Path | str) -> dict[str, Any]:
    with open(Path(dataset_dir) / "ground_truth.json", encoding="utf-8") as f:
        return json.load(f)


def _members(group: dict[str, Any]) -> frozenset[str]:
    return frozenset(group["ledger_order_ids"] + group["settlement_txn_ids"]
                     + group["bank_row_ids"])


def evaluate(report: ReconciliationReport, dataset_dir: Path | str, *,
             batch=None, reasoning_seconds: float | None = None,
             llm_free_groups: int = 0) -> EvaluationResult:
    gt = load_ground_truth(dataset_dir)
    groups = gt["groups"]

    ev = EvaluationResult(
        dataset=Path(dataset_dir).name,
        total_groups=len(groups),
        total_rows=report.rows_processed,
        deterministic_seconds=report.elapsed_seconds,
        llm_free_groups=llm_free_groups,
        exception_reasons=report.by_exception_reason(),
    )
    if reasoning_seconds is not None:
        ev.end_to_end_seconds = report.elapsed_seconds + reasoning_seconds

    by_record: dict[str, MatchResult] = {r.record_id: r for r in report.results}
    owner_of_bank: dict[str, MatchResult] = {}
    for r in report.results:
        for bid in r.linked_ids.get("bank", []):
            owner_of_bank[bid] = r

    net_paise = {}
    if batch is not None:
        net_paise = {s.transaction_id: s.net_paise for s in batch.settlements}

    ev.groups_claimed = sum(1 for r in report.results if r.status in AUTO_MATCHABLE)

    for g in groups:
        case = g["case_type"]
        bucket = ev.by_case_type.setdefault(case, CaseTypeBreakdown(case))
        bucket.total += 1

        expected = _members(g)
        should_auto = g["expected_resolution"] in ("matched", "matched_split")
        if not should_auto:
            ev.exceptions_expected += 1

        owner = _find_owner(g, by_record, owner_of_bank)
        did_auto = owner is not None and owner.status in AUTO_MATCHABLE
        predicted = owner.group_key if owner is not None else frozenset()
        exact = bool(owner is not None and predicted == expected)

        if did_auto:
            if exact:
                ev.exact_membership += 1
            elif predicted & expected:
                ev.partial_membership += 1

        # Money attribution, for the cash position and the cost of errors.
        amount = sum(net_paise.get(t, 0) for t in g["settlement_txn_ids"][:1])
        if did_auto:
            ev.paise_auto_matched += amount
        else:
            ev.paise_in_exceptions += amount

        # --- the decision -------------------------------------------------
        if should_auto and did_auto and exact:
            ev.true_positives += 1
            bucket.correct += 1
        elif should_auto and did_auto and not exact:
            # Auto-matched, but to the wrong counterpart. Counted as BOTH a
            # false positive (a wrong link was asserted) and a false negative
            # (the right link was missed). Scoring it only as a miss would let a
            # mis-linking system look merely conservative.
            ev.false_positives += 1
            ev.false_negatives += 1
            bucket.false_positive += 1
            ev.false_positive_detail.append({
                "group_id": g["group_id"], "case_type": case,
                "kind": "wrong_counterpart",
                "expected": sorted(expected), "predicted": sorted(predicted),
                "note": g["notes"],
            })
        elif should_auto and not did_auto:
            ev.false_negatives += 1
            bucket.false_negative += 1
        elif not should_auto and did_auto:
            ev.false_positives += 1
            bucket.false_positive += 1
            ev.false_positive_detail.append({
                "group_id": g["group_id"], "case_type": case,
                "kind": "forced_a_match_that_should_have_been_held",
                "expected": sorted(expected), "predicted": sorted(predicted),
                "note": g["notes"],
            })
        else:
            ev.true_negatives += 1
            bucket.correct += 1
            ev.exceptions_correctly_held += 1
            if owner is not None and _category_ok(case, owner.exception_reason):
                ev.exceptions_correctly_categorised += 1

    _score_adversarial(ev, gt, report)
    return ev


def _find_owner(group: dict[str, Any], by_record: dict[str, MatchResult],
                owner_of_bank: dict[str, MatchResult]) -> MatchResult | None:
    """Locate the engine result that decided this ground-truth group.

    Keyed on the settlement when there is one. Bank-only groups (orphan credits)
    are located through whichever result claimed the credit - which may be a
    settlement's group, and that is exactly the case worth catching.
    """
    for tid in group["settlement_txn_ids"]:
        if tid in by_record:
            return by_record[tid]
    for bid in group["bank_row_ids"]:
        if bid in by_record:
            return by_record[bid]
        if bid in owner_of_bank:
            return owner_of_bank[bid]
    return None


def _category_ok(case_type: str, reason: str | None) -> bool:
    allowed = EXPECTED_REASONS.get(case_type)
    if allowed is None:
        return reason is not None
    return reason in allowed


def _score_adversarial(ev: EvaluationResult, gt: dict[str, Any],
                       report: ReconciliationReport) -> None:
    """Count how many near-duplicate pairs were conflated into one group.

    Conflation means a single engine group contains records belonging to both
    legs of a pair - i.e. the matcher decided two unrelated transactions were
    the same payment. This is the metric the whole refusal design exists to keep
    at zero, so it is reported whatever it says.
    """
    by_id = {g["group_id"]: g for g in gt["groups"]}
    pairs = gt.get("adversarial_pairs", [])
    ev.adversarial_pairs = len(pairs)

    for a_id, b_id in pairs:
        a, b = _members(by_id[a_id]), _members(by_id[b_id])
        for r in report.results:
            if r.status not in AUTO_MATCHABLE:
                continue
            k = r.group_key
            if (k & a) and (k & b):
                ev.adversarial_conflated += 1
                ev.false_positive_detail.append({
                    "group_id": f"{a_id}+{b_id}", "case_type": "adversarial_conflation",
                    "kind": "two_unrelated_transactions_merged",
                    "expected": [sorted(a), sorted(b)], "predicted": sorted(k),
                    "note": "near-duplicate pair bound into a single group",
                })
                break


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def format_report(ev: EvaluationResult) -> str:
    L: list[str] = []
    add = L.append
    rule = "=" * 78

    add(rule)
    add(f"EVALUATION - {ev.dataset} ({ev.total_rows} rows, {ev.total_groups} groups)")
    add(rule)

    add("")
    add("HEADLINE")
    add(f"  auto-match rate            {ev.auto_match_rate:6.1%}   "
        f"({ev.groups_claimed} of {ev.total_groups} groups resolved with no human)")
    add(f"  precision                  {ev.precision:6.1%}   "
        f"of what it auto-matched, this share was exactly right")
    add(f"  recall                     {ev.recall:6.1%}   "
        f"of what was matchable, this share was found")
    add(f"  F1                         {ev.f1:6.1%}")

    add("")
    add("FALSE POSITIVES - the expensive error")
    add(f"  false positives            {ev.false_positives:6d}   "
        f"money bound to the wrong counterpart, or a match forced")
    add(f"  adversarial pairs tested   {ev.adversarial_pairs:6d}")
    add(f"  ... of those, conflated    {ev.adversarial_conflated:6d}   "
        f"({ev.adversarial_fp_rate:.1%} adversarial false-positive rate)")
    if ev.false_positives == 0 and ev.adversarial_conflated == 0:
        add("  No incorrect link was asserted anywhere in this batch.")
    else:
        add("  Details:")
        for d in ev.false_positive_detail[:6]:
            add(f"    - {d['group_id']} [{d['case_type']}] {d['kind']}")
            add(f"        expected {d['expected']}")
            add(f"        predicted {d['predicted']}")

    add("")
    add("FALSE NEGATIVES - the cheap error, deliberately preferred")
    add(f"  false negatives            {ev.false_negatives:6d}   "
        f"matchable, but held back for a human")
    add("  Every one of these is a case the engine could have guessed at and")
    add("  chose not to. That trade is intentional.")

    add("")
    add("EXCEPTION HANDLING")
    add(f"  should be held             {ev.exceptions_expected:6d}")
    add(f"  correctly held             {ev.exceptions_correctly_held:6d}   "
        f"({ev.exception_hold_rate:.1%})")
    add(f"  correctly categorised      {ev.exceptions_correctly_categorised:6d}   "
        f"({ev.category_accuracy:.1%} of those held)")
    add("")
    add("  Engine's own breakdown of what it could not resolve:")
    for reason, n in sorted(ev.exception_reasons.items(), key=lambda kv: -kv[1]):
        add(f"    {n:>4}  {reason}")

    add("")
    add("BY CASE TYPE")
    add(f"  {'case':<26}{'n':>5}{'correct':>9}{'FP':>5}{'FN':>5}{'acc':>8}")
    for case in sorted(ev.by_case_type, key=lambda c: -ev.by_case_type[c].total):
        b = ev.by_case_type[case]
        add(f"  {case:<26}{b.total:>5}{b.correct:>9}{b.false_positive:>5}"
            f"{b.false_negative:>5}{b.accuracy:>8.0%}")

    add("")
    add("THROUGHPUT")
    add(f"  deterministic engine       {ev.deterministic_seconds * 1000:8.1f} ms   "
        f"{ev.deterministic_rps:,.0f} rows/sec")
    if ev.end_to_end_rps is not None:
        add(f"  end-to-end incl. reasoning {ev.end_to_end_seconds * 1000:8.1f} ms   "
            f"{ev.end_to_end_rps:,.0f} rows/sec")
        add("  Reported separately on purpose: model latency dominates the second")
        add("  number, and blending them would overstate matching throughput.")
    if ev.llm_free_groups:
        add(f"  groups needing no model    {ev.llm_free_groups:8d}   "
            f"({ev.llm_free_groups / ev.total_groups:.0%} of the batch)")

    add("")
    add("VALUE RECONCILED")
    add(f"  auto-matched               {format_inr(ev.paise_auto_matched):>16}")
    add(f"  sitting in exceptions      {format_inr(ev.paise_in_exceptions):>16}")

    add("")
    add(rule)
    return "\n".join(L)
