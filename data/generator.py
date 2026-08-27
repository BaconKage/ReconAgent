"""Seeded synthetic data generator for three-way reconciliation.

Produces a gateway settlement report, a bank statement and an internal ledger,
plus a ground_truth.json that the matcher and the reasoning layer never read.

Design notes
------------
Every messy case here is constructed deliberately, with a stated reason for why
a naive matcher gets it wrong. Random noise would not test anything: the point
is that each case defeats one specific shortcut.

Row ordering carries no signal. Bank row IDs are assigned *after* a shuffle, so
a matcher cannot exploit adjacency or ID proximity to infer grouping.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import string
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from core.models import CaseType, Resolution
from core.normalize import paise_to_rupees

# --------------------------------------------------------------------------
# Tunables shared with the matcher's expectations (but never imported by it)
# --------------------------------------------------------------------------

FEE_RATE = 0.02          # 2% gateway fee
GST_RATE = 0.18          # 18% GST on the fee
BASE_DATE = date(2026, 7, 1)
WINDOW_DAYS = 28

CUSTOMERS = [
    "Aarav Sharma", "Diya Menon", "Kabir Nair", "Ishaan Reddy", "Ananya Iyer",
    "Vivaan Joshi", "Meera Pillai", "Rohan Desai", "Saanvi Rao", "Arjun Kulkarni",
    "Neha Bhatt", "Aditya Ghosh", "Riya Chawla", "Karthik Subramanian",
    "Priya Malhotra", "Siddharth Bose", "Tanvi Shetty", "Nikhil Verma",
    "Aisha Qureshi", "Devansh Mehta",
]

# Bank narration templates. {u} is the 12-digit UTR; templates without {u}
# force the matcher to fall back to amount/date reasoning.
DESC_WITH_UTR = [
    "NEFT-UTR{u}-RAZORPAY SOFTWARE-SETTLEMENT",
    "IMPS/UTR {u}/RZPY STLMNT",
    "RTGS UTR{u} RAZORPAY",
    "razorpay settlement utr{u}",
    "NEFT-UTR-{u}-RZPY",
]
DESC_NO_UTR = [
    "NEFT-RAZORPAY-SETTLEMENT",
    "IMPS/RZPY STLMNT/COLLECTION",
    "RTGS RAZORPAY SOFTWARE PVT LTD",
    "razorpay payout settlement credit",
]
# Held-out only - a narration shape the dev set never contains, so the parser
# is exercised on a format it was not tuned against.
DESC_HOLDOUT_EXTRA = [
    "BY TRANSFER-NEFT*RZPYSETL*{u}*MUMBAI",
    "CR/UTRNO{u}/RAZORPAY/SETTLE",
]


# --------------------------------------------------------------------------
# Drafts (pre-ID-assignment)
# --------------------------------------------------------------------------

@dataclass
class BankDraft:
    value_date: date
    description: str
    credit_paise: int
    utr_reference: str
    assigned_id: str = ""


@dataclass
class SettlementDraft:
    transaction_id: str
    order_id: str
    gross_paise: int
    fee_paise: int
    tax_paise: int
    net_paise: int
    settlement_date: date
    utr_number: str


@dataclass
class LedgerDraft:
    order_id: str
    customer: str
    expected_paise: int
    order_date: date
    status: str


@dataclass
class GroupDraft:
    group_id: str
    case_type: CaseType
    resolution: Resolution
    notes: str
    ledger: list[LedgerDraft] = field(default_factory=list)
    settlements: list[SettlementDraft] = field(default_factory=list)
    banks: list[BankDraft] = field(default_factory=list)


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

class ReconDataGenerator:
    def __init__(self, seed: int, mix: dict, *, holdout: bool = False,
                 max_lag_days: int = 2):
        self.rng = random.Random(seed)
        self.seed = seed
        self.mix = mix
        self.holdout = holdout
        self.max_lag_days = max_lag_days
        self._used_utrs: set[str] = set()
        self._counter = 0
        self.groups: list[GroupDraft] = []
        self.adversarial_pairs: list[list[str]] = []

    # ---- primitives ------------------------------------------------------

    def _next_gid(self) -> str:
        self._counter += 1
        return "G{:04d}".format(self._counter)

    def _txn_id(self) -> str:
        return "pay_" + "".join(self.rng.choices(string.ascii_lowercase + string.digits, k=10))

    def _order_id(self) -> str:
        return "order_" + "".join(self.rng.choices(string.ascii_lowercase + string.digits, k=10))

    def _utr(self) -> str:
        while True:
            u = "".join(self.rng.choices(string.digits, k=12))
            if u not in self._used_utrs:
                self._used_utrs.add(u)
                return u

    def _gross(self) -> int:
        """A plausible merchant order value, in paise."""
        rupees = self.rng.choice([
            self.rng.randint(100, 2000),
            self.rng.randint(2000, 12000),
            self.rng.randint(12000, 50000),
        ])
        return rupees * 100 + self.rng.choice([0, 50, 25, 75, 99])

    def _fees(self, gross: int) -> tuple:
        fee = round(gross * FEE_RATE)
        tax = round(fee * GST_RATE)
        return fee, tax, gross - fee - tax

    def _date(self) -> date:
        return BASE_DATE + timedelta(days=self.rng.randint(0, WINDOW_DAYS))

    def _desc(self, utr):
        if utr is None:
            return self.rng.choice(DESC_NO_UTR)
        pool = list(DESC_WITH_UTR)
        if self.holdout:
            pool = pool + DESC_HOLDOUT_EXTRA
        return self.rng.choice(pool).format(u=utr)

    def _bank(self, value_date, credit_paise, utr, *, ref_style="canonical",
              desc_has_utr=None) -> BankDraft:
        """Build a bank row.

        ref_style controls how mangled the dedicated reference column is:
        canonical / lower / hyphen / spaced / truncated / empty.
        """
        styles = {
            "canonical": lambda u: "UTR" + u,
            "lower": lambda u: "utr" + u,
            "hyphen": lambda u: "UTR-" + u,
            "spaced": lambda u: "UTR " + u,
            "truncated": lambda u: "UTR" + u[:-1],
            "empty": lambda u: "",
        }
        ref = styles[ref_style](utr) if utr else ""
        if desc_has_utr is None:
            desc_has_utr = utr is not None and ref_style != "empty"
        return BankDraft(
            value_date=value_date,
            description=self._desc(utr if desc_has_utr else None),
            credit_paise=credit_paise,
            utr_reference=ref,
        )

    def _ledger(self, order_id, gross, order_date, status="paid") -> LedgerDraft:
        return LedgerDraft(order_id, self.rng.choice(CUSTOMERS), gross, order_date, status)

    def _settlement(self, order_id, gross, sdate, utr, net_override=None) -> SettlementDraft:
        fee, tax, net = self._fees(gross)
        return SettlementDraft(self._txn_id(), order_id, gross, fee, tax,
                               net if net_override is None else net_override, sdate, utr)

    # ---- case builders ---------------------------------------------------

    def case_clean(self) -> GroupDraft:
        """Baseline. UTR present and canonical on both sides, same date, exact net."""
        oid, gross, d, utr = self._order_id(), self._gross(), self._date(), self._utr()
        s = self._settlement(oid, gross, d, utr)
        return GroupDraft(
            self._next_gid(), CaseType.CLEAN, Resolution.MATCHED,
            "canonical UTR both sides, same-day credit, exact net",
            [self._ledger(oid, gross, d)], [s],
            [self._bank(d, s.net_paise, utr, ref_style="canonical")],
        )

    def case_fee_deduction(self) -> GroupDraft:
        """Bank credit is NET; the ledger holds GROSS. No UTR on the bank row.

        Defeats: any matcher that compares the ledger's expected amount against
        the bank credit. The gap is exactly fee + GST-on-fee.
        """
        oid, gross, d, utr = self._order_id(), self._gross(), self._date(), self._utr()
        s = self._settlement(oid, gross, d, utr)
        return GroupDraft(
            self._next_gid(), CaseType.FEE_DEDUCTION, Resolution.MATCHED,
            "ledger gross {} vs bank net {}; gap = fee {} + tax {}".format(
                paise_to_rupees(gross), paise_to_rupees(s.net_paise),
                paise_to_rupees(s.fee_paise), paise_to_rupees(s.tax_paise)),
            [self._ledger(oid, gross, d)], [s],
            [self._bank(d, s.net_paise, utr, ref_style="empty", desc_has_utr=False)],
        )

    def case_timing_lag(self) -> GroupDraft:
        """Credit lands T+1/T+2 and the UTR reference is truncated by one digit.

        Defeats: same-day joins, and exact-string UTR joins.
        """
        oid, gross, d, utr = self._order_id(), self._gross(), self._date(), self._utr()
        s = self._settlement(oid, gross, d, utr)
        lag = self.rng.randint(1, self.max_lag_days)
        return GroupDraft(
            self._next_gid(), CaseType.TIMING_LAG, Resolution.MATCHED,
            "credit T+{}; UTR reference truncated to 11 digits".format(lag),
            [self._ledger(oid, gross, d)], [s],
            [self._bank(d + timedelta(days=lag), s.net_paise, utr,
                        ref_style="truncated", desc_has_utr=False)],
        )

    def case_split_settlement(self) -> GroupDraft:
        """One settlement paid out as 2-3 bank credits on staggered dates.

        Defeats: one-to-one matching. No individual credit equals the net.
        """
        oid, gross, d, utr = self._order_id(), self._gross(), self._date(), self._utr()
        s = self._settlement(oid, gross, d, utr)
        n = self.rng.choice([2, 2, 3])
        cuts = sorted(self.rng.sample(range(1, s.net_paise), n - 1))
        parts, prev = [], 0
        for c in cuts + [s.net_paise]:
            parts.append(c - prev)
            prev = c
        banks = [
            self._bank(d + timedelta(days=self.rng.randint(1, self.max_lag_days)),
                       p, None, ref_style="empty", desc_has_utr=False)
            for p in parts
        ]
        return GroupDraft(
            self._next_gid(), CaseType.SPLIT_SETTLEMENT, Resolution.MATCHED_SPLIT,
            "net {} settled as {}".format(
                paise_to_rupees(s.net_paise),
                " + ".join(paise_to_rupees(p) for p in parts)),
            [self._ledger(oid, gross, d)], [s], banks,
        )

    def case_partial_refund(self) -> GroupDraft:
        """UTR joins cleanly but the credited amount is short by a refund.

        Defeats: trusting a UTR match without reconciling the amount. This is
        why the engine re-verifies value even after an identifier match.
        """
        oid, gross, d, utr = self._order_id(), self._gross(), self._date(), self._utr()
        s = self._settlement(oid, gross, d, utr)
        refund = round(s.net_paise * self.rng.uniform(0.15, 0.45))
        return GroupDraft(
            self._next_gid(), CaseType.PARTIAL_REFUND, Resolution.EXCEPTION_PARTIAL_REFUND,
            "partial refund of {}; credited {} against net {}".format(
                paise_to_rupees(refund), paise_to_rupees(s.net_paise - refund),
                paise_to_rupees(s.net_paise)),
            [self._ledger(oid, gross, d, status="partially_refunded")], [s],
            [self._bank(d, s.net_paise - refund, utr, ref_style="canonical")],
        )

    def case_duplicate(self) -> GroupDraft:
        """Webhook retry: the same settlement appears twice, one real credit.

        Defeats: naive counting, and any matcher that would happily bind both
        settlement rows to the single bank credit.
        """
        oid, gross, d, utr = self._order_id(), self._gross(), self._date(), self._utr()
        s = self._settlement(oid, gross, d, utr)
        dup = SettlementDraft(s.transaction_id, s.order_id, s.gross_paise, s.fee_paise,
                              s.tax_paise, s.net_paise, s.settlement_date, s.utr_number)
        return GroupDraft(
            self._next_gid(), CaseType.DUPLICATE, Resolution.EXCEPTION_DUPLICATE,
            "transaction_id {} emitted twice (webhook retry); one bank credit".format(
                s.transaction_id),
            [self._ledger(oid, gross, d)], [s, dup],
            [self._bank(d, s.net_paise, utr, ref_style="canonical")],
        )

    def case_rounding(self) -> GroupDraft:
        """Credit differs from net by 1-50 paise, with no UTR to fall back on.

        Defeats: exact-amount equality. Must match inside tolerance.
        """
        oid, gross, d, utr = self._order_id(), self._gross(), self._date(), self._utr()
        s = self._settlement(oid, gross, d, utr)
        drift = self.rng.choice([-1, 1]) * self.rng.randint(1, 50)
        return GroupDraft(
            self._next_gid(), CaseType.ROUNDING, Resolution.MATCHED,
            "credit off net by {} paise; no UTR on the bank row".format(drift),
            [self._ledger(oid, gross, d)], [s],
            [self._bank(d, s.net_paise + drift, utr, ref_style="empty", desc_has_utr=False)],
        )

    def case_adversarial_resolvable(self) -> list:
        """Two unrelated transactions, near-identical amount and date.

        One carries a clean UTR, the other carries none. A cascade that resolves
        identifiers *before* reaching for amount+date consumes the first and then
        has only one candidate left for the second. An amount-first matcher
        conflates them. Both are individually matchable, so both must be matched.
        """
        d = self._date()
        gross_a = self._gross()
        oid_a, oid_b = self._order_id(), self._order_id()
        utr_a, utr_b = self._utr(), self._utr()
        s_a = self._settlement(oid_a, gross_a, d, utr_a)
        # Identical net to its twin. The ONLY thing separating these two
        # transactions is that one carries a recoverable UTR - so this tests
        # cascade order and nothing else.
        s_b = self._settlement(oid_b, gross_a, d, utr_b)
        g_a = GroupDraft(
            self._next_gid(), CaseType.ADVERSARIAL_RESOLVABLE, Resolution.MATCHED,
            "adversarial twin (identified leg): clean UTR lets the cascade claim it first",
            [self._ledger(oid_a, s_a.gross_paise, d)], [s_a],
            [self._bank(d, s_a.net_paise, utr_a, ref_style="canonical")],
        )
        g_b = GroupDraft(
            self._next_gid(), CaseType.ADVERSARIAL_RESOLVABLE, Resolution.MATCHED,
            "adversarial twin (unidentified leg): no UTR, net identical to its twin",
            [self._ledger(oid_b, s_b.gross_paise, d + timedelta(days=1))], [s_b],
            [self._bank(d + timedelta(days=1), s_b.net_paise, utr_b,
                        ref_style="empty", desc_has_utr=False)],
        )
        return [g_a, g_b]

    def case_adversarial_ambiguous(self) -> list:
        """Two unrelated transactions, same date, nets within tolerance, NO UTR
        on either bank row.

        The two settlements carry IDENTICAL nets on the SAME date, and neither
        bank row carries a UTR. This is deliberate: undecidability must not be an
        artefact of the tolerance setting. An earlier version separated the twins
        by a few paise, which made them separable under a tight tolerance and
        turned a correct match into a scored false positive. With the amounts
        equal, no threshold anywhere can distinguish them, so "refuse both" is
        the right answer at every configuration.
        """
        d = self._date()
        gross_a = self._gross()
        oid_a, oid_b = self._order_id(), self._order_id()
        s_a = self._settlement(oid_a, gross_a, d, self._utr())
        s_b = self._settlement(oid_b, gross_a, d, self._utr())
        out = []
        for oid, s, tag in ((oid_a, s_a, "A"), (oid_b, s_b, "B")):
            out.append(GroupDraft(
                self._next_gid(), CaseType.ADVERSARIAL_AMBIGUOUS,
                Resolution.EXCEPTION_AMBIGUOUS,
                "ambiguous twin {}: same date, nets within tolerance, no UTR on either "
                "bank row - not decidable from the data".format(tag),
                [self._ledger(oid, s.gross_paise, d)], [s],
                [self._bank(d, s.net_paise, None, ref_style="empty", desc_has_utr=False)],
            ))
        return out

    def case_unmatchable_settlement(self, *, tempting: bool = False) -> list:
        """A settlement with no corresponding bank credit at all.

        When tempting, an unrelated orphan credit is planted just outside BOTH
        the amount tolerance and the date window - close enough to look like a
        lead, far enough that binding them would be wrong. This is the case the
        reasoning layer must decline to force.
        """
        oid, gross, d, utr = self._order_id(), self._gross(), self._date(), self._utr()
        s = self._settlement(oid, gross, d, utr)
        groups = [GroupDraft(
            self._next_gid(), CaseType.UNMATCHABLE, Resolution.EXCEPTION_UNMATCHABLE,
            "settlement with no bank credit anywhere in the batch"
            + (" (paired with a planted near-miss)" if tempting else ""),
            [self._ledger(oid, gross, d)], [s], [],
        )]
        if tempting:
            decoy_amt = s.net_paise + self.rng.choice([-1, 1]) * self.rng.randint(75, 140)
            decoy_date = d + timedelta(days=self.rng.randint(4, 6))
            groups.append(GroupDraft(
                self._next_gid(), CaseType.UNMATCHABLE, Resolution.EXCEPTION_UNMATCHABLE,
                "planted near-miss: orphan credit just outside both the amount tolerance "
                "and the date window - must NOT be bound to the settlement it resembles",
                [], [],
                [self._bank(decoy_date, decoy_amt, None, ref_style="empty", desc_has_utr=False)],
            ))
        return groups

    def case_unmatchable_bank(self) -> GroupDraft:
        """A bank credit with no settlement behind it (e.g. an unrelated inflow)."""
        return GroupDraft(
            self._next_gid(), CaseType.UNMATCHABLE, Resolution.EXCEPTION_UNMATCHABLE,
            "bank credit with no settlement counterpart in the batch",
            [], [],
            [self._bank(self._date(), self._gross(), None,
                        ref_style="empty", desc_has_utr=False)],
        )

    # ---- assembly --------------------------------------------------------

    def build(self) -> list:
        m = self.mix
        for _ in range(m.get(CaseType.CLEAN, 0)):
            self.groups.append(self.case_clean())
        for _ in range(m.get(CaseType.FEE_DEDUCTION, 0)):
            self.groups.append(self.case_fee_deduction())
        for _ in range(m.get(CaseType.TIMING_LAG, 0)):
            self.groups.append(self.case_timing_lag())
        for _ in range(m.get(CaseType.SPLIT_SETTLEMENT, 0)):
            self.groups.append(self.case_split_settlement())
        for _ in range(m.get(CaseType.PARTIAL_REFUND, 0)):
            self.groups.append(self.case_partial_refund())
        for _ in range(m.get(CaseType.DUPLICATE, 0)):
            self.groups.append(self.case_duplicate())
        for _ in range(m.get(CaseType.ROUNDING, 0)):
            self.groups.append(self.case_rounding())

        for _ in range(m.get(CaseType.ADVERSARIAL_RESOLVABLE, 0)):
            pair = self.case_adversarial_resolvable()
            self.groups.extend(pair)
            self.adversarial_pairs.append([g.group_id for g in pair])
        for _ in range(m.get(CaseType.ADVERSARIAL_AMBIGUOUS, 0)):
            pair = self.case_adversarial_ambiguous()
            self.groups.extend(pair)
            self.adversarial_pairs.append([g.group_id for g in pair])

        n_unmatch = m.get(CaseType.UNMATCHABLE, 0)
        n_bank_orphans = max(1, n_unmatch // 3)
        n_settlement_orphans = n_unmatch - n_bank_orphans
        for i in range(n_settlement_orphans):
            # Exactly one tempting case - the deliberate refusal demo.
            self.groups.extend(self.case_unmatchable_settlement(tempting=(i == 0)))
        for _ in range(n_bank_orphans):
            self.groups.append(self.case_unmatchable_bank())

        self._assign_bank_ids()
        return self.groups

    def _assign_bank_ids(self) -> None:
        """Shuffle every bank row across all groups, THEN assign sequential IDs.

        Without this, BNK_00041 and BNK_00042 being adjacent would itself be a
        hint that they belong to the same split - an artefact of generation
        order, not a property of the data. Removing it keeps evaluation honest.
        """
        all_banks = [b for g in self.groups for b in g.banks]
        self.rng.shuffle(all_banks)
        for i, b in enumerate(all_banks, start=1):
            b.assigned_id = "BNK_{:05d}".format(i)


# --------------------------------------------------------------------------
# Mixes
# --------------------------------------------------------------------------

DEV_MIX = {
    CaseType.CLEAN: 10, CaseType.FEE_DEDUCTION: 15, CaseType.TIMING_LAG: 10,
    CaseType.SPLIT_SETTLEMENT: 8, CaseType.PARTIAL_REFUND: 8, CaseType.DUPLICATE: 5,
    CaseType.ROUNDING: 5, CaseType.ADVERSARIAL_RESOLVABLE: 3,
    CaseType.ADVERSARIAL_AMBIGUOUS: 2, CaseType.UNMATCHABLE: 8,
}

# Held-out deliberately re-weights toward the hard cases, widens the settlement
# lag, and adds narration formats the dev set never contained. Metrics are
# expected to be LOWER here; that is the point of keeping it sealed.
HOLDOUT_MIX = {
    CaseType.CLEAN: 6, CaseType.FEE_DEDUCTION: 12, CaseType.TIMING_LAG: 12,
    CaseType.SPLIT_SETTLEMENT: 10, CaseType.PARTIAL_REFUND: 9, CaseType.DUPLICATE: 6,
    CaseType.ROUNDING: 7, CaseType.ADVERSARIAL_RESOLVABLE: 5,
    CaseType.ADVERSARIAL_AMBIGUOUS: 4, CaseType.UNMATCHABLE: 10,
}


def write_dataset(out_dir: Path, gen: ReconDataGenerator) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = gen.build()
    rng = gen.rng

    settlements = [(g, s) for g in groups for s in g.settlements]
    ledgers = [(g, l) for g in groups for l in g.ledger]
    banks = [(g, b) for g in groups for b in g.banks]
    rng.shuffle(settlements)
    rng.shuffle(ledgers)
    banks.sort(key=lambda gb: gb[1].assigned_id)

    with open(out_dir / "settlement_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["transaction_id", "order_id", "gross_amount", "fee", "tax_on_fee",
                    "net_amount", "settlement_date", "utr_number"])
        for _, s in settlements:
            w.writerow([s.transaction_id, s.order_id, paise_to_rupees(s.gross_paise),
                        paise_to_rupees(s.fee_paise), paise_to_rupees(s.tax_paise),
                        paise_to_rupees(s.net_paise), s.settlement_date.isoformat(),
                        "UTR" + s.utr_number])

    with open(out_dir / "bank_statement.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bank_row_id", "date", "description", "credit_amount", "utr_reference"])
        for _, b in banks:
            w.writerow([b.assigned_id, b.value_date.isoformat(), b.description,
                        paise_to_rupees(b.credit_paise), b.utr_reference])

    with open(out_dir / "internal_ledger.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "customer", "expected_amount", "order_date", "status"])
        for _, l in ledgers:
            w.writerow([l.order_id, l.customer, paise_to_rupees(l.expected_paise),
                        l.order_date.isoformat(), l.status])

    gt = {
        "meta": {
            "seed": gen.seed,
            "holdout": gen.holdout,
            "max_lag_days": gen.max_lag_days,
            "mix": {k.value: v for k, v in gen.mix.items()},
            "counts": {
                "settlement_rows": len(settlements),
                "bank_rows": len(banks),
                "ledger_rows": len(ledgers),
                "total_rows": len(settlements) + len(banks) + len(ledgers),
                "groups": len(groups),
            },
        },
        "groups": [
            {
                "group_id": g.group_id,
                "case_type": g.case_type.value,
                "expected_resolution": g.resolution.value,
                "ledger_order_ids": [l.order_id for l in g.ledger],
                "settlement_txn_ids": [s.transaction_id for s in g.settlements],
                "bank_row_ids": [b.assigned_id for b in g.banks],
                "notes": g.notes,
            }
            for g in groups
        ],
        "adversarial_pairs": gen.adversarial_pairs,
        "unmatchable_groups": [g.group_id for g in groups
                               if g.resolution == Resolution.EXCEPTION_UNMATCHABLE],
    }
    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic reconciliation datasets.")
    ap.add_argument("--which", choices=["dev", "holdout", "both"], default="both")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent

    if args.which in ("dev", "both"):
        gt = write_dataset(root / "dev", ReconDataGenerator(42, DEV_MIX, max_lag_days=2))
        print("dev     ", gt["meta"]["counts"])
    if args.which in ("holdout", "both"):
        gt = write_dataset(root / "holdout",
                           ReconDataGenerator(1337, HOLDOUT_MIX, holdout=True, max_lag_days=4))
        print("holdout ", gt["meta"]["counts"])


if __name__ == "__main__":
    main()
