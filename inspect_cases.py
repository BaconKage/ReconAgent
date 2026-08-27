"""Print one worked example of each messy case type in a dataset.

This reads ground_truth.json, so it is an inspection tool for the builder and
the demo - never part of the reconciliation path.

    python inspect_cases.py --dataset dev [--case split_settlement]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from core.normalize import format_inr, rupees_to_paise

ROOT = Path(__file__).resolve().parent

CASE_ORDER = [
    "clean", "fee_deduction", "timing_lag", "split_settlement", "partial_refund",
    "duplicate", "rounding", "adversarial_resolvable", "adversarial_ambiguous",
    "unmatchable",
]

WHY = {
    "clean": "Baseline. Should match on UTR alone.",
    "fee_deduction": "Ledger holds GROSS, bank credits NET. Defeats ledger-to-bank amount joins.",
    "timing_lag": "Credit lands T+1/T+2 with a truncated UTR. Defeats same-day and exact-ID joins.",
    "split_settlement": "One settlement, several credits. Defeats one-to-one matching.",
    "partial_refund": "UTR joins but the amount is short. Defeats trusting an ID match blindly.",
    "duplicate": "Webhook retry emitted the settlement twice against one real credit.",
    "rounding": "Sub-rupee drift with no UTR. Defeats exact-amount equality.",
    "adversarial_resolvable": "Two unrelated txns, near-identical amount/date. One has a UTR, so cascade order saves it.",
    "adversarial_ambiguous": "Same, but NEITHER has a UTR. Genuinely undecidable - must be refused.",
    "unmatchable": "No counterpart exists. Must land in the exception list, not be forced.",
}


def load(dataset: str):
    root = ROOT / "data" / dataset
    with open(root / "settlement_report.csv", encoding="utf-8") as f:
        settlements = {r["transaction_id"]: r for r in csv.DictReader(f)}
    with open(root / "bank_statement.csv", encoding="utf-8") as f:
        banks = {r["bank_row_id"]: r for r in csv.DictReader(f)}
    with open(root / "internal_ledger.csv", encoding="utf-8") as f:
        ledger = {r["order_id"]: r for r in csv.DictReader(f)}
    with open(root / "ground_truth.json", encoding="utf-8") as f:
        gt = json.load(f)
    return settlements, banks, ledger, gt


def show_group(g, settlements, banks, ledger) -> None:
    print(f"  group {g['group_id']}  ->  expected: {g['expected_resolution']}")
    print(f"  note: {g['notes']}")
    for oid in g["ledger_order_ids"]:
        r = ledger[oid]
        print(f"    LEDGER  {oid}  {r['customer']:<22} expected={format_inr(rupees_to_paise(r['expected_amount'])):>13}"
              f"  {r['order_date']}  [{r['status']}]")
    for tid in g["settlement_txn_ids"]:
        r = settlements[tid]
        print(f"    SETTLE  {tid}  gross={format_inr(rupees_to_paise(r['gross_amount'])):>13}"
              f"  fee={format_inr(rupees_to_paise(r['fee'])):>10}"
              f"  tax={format_inr(rupees_to_paise(r['tax_on_fee'])):>9}"
              f"  net={format_inr(rupees_to_paise(r['net_amount'])):>13}  {r['settlement_date']}  {r['utr_number']}")
    for bid in g["bank_row_ids"]:
        r = banks[bid]
        ref = r["utr_reference"] or "(none)"
        print(f"    BANK    {bid}  credit={format_inr(rupees_to_paise(r['credit_amount'])):>13}"
              f"  {r['date']}  ref={ref:<18}")
        print(f"            narration: {r['description']}")
    if not g["bank_row_ids"]:
        print("    BANK    (no credit in this batch)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--case", default=None, help="show only this case type")
    ap.add_argument("--n", type=int, default=1, help="examples per case type")
    args = ap.parse_args()

    settlements, banks, ledger, gt = load(args.dataset)
    counts = gt["meta"]["counts"]
    print(f"\n=== {args.dataset.upper()} : {counts['total_rows']} rows across 3 sources, "
          f"{counts['groups']} reconciliation groups "
          f"(settlement {counts['settlement_rows']} / bank {counts['bank_rows']} / ledger {counts['ledger_rows']}) ===\n")

    wanted = [args.case] if args.case else CASE_ORDER
    for case in wanted:
        groups = [g for g in gt["groups"] if g["case_type"] == case]
        if not groups:
            continue
        print("-" * 78)
        print(f"{case.upper()}  ({len(groups)} groups)")
        print(f"  why it is here: {WHY.get(case, '')}")
        print("-" * 78)
        # For unmatchable, lead with the planted near-miss - it is the demo case.
        groups.sort(key=lambda g: not g["notes"].startswith("planted near-miss"))
        for g in groups[:args.n]:
            show_group(g, settlements, banks, ledger)


if __name__ == "__main__":
    main()
