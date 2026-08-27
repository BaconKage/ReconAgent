"""Load the three CSV sources into typed records.

This is the only place raw strings become domain objects. Money is converted to
integer paise here and stays that way; UTRs are recovered here and the
provenance of that recovery is retained for the audit trail.

The loader is deliberately tolerant: real bank exports contain rows that do not
parse, and a reconciliation run that dies on one malformed line is useless. Bad
rows are collected and reported rather than raised.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from core.models import BankRecord, LedgerRecord, SettlementRecord
from core.normalize import parse_date, recover_utr, rupees_to_paise, utr_digits


@dataclass
class LoadedBatch:
    settlements: list[SettlementRecord] = field(default_factory=list)
    banks: list[BankRecord] = field(default_factory=list)
    ledger: dict[str, LedgerRecord] = field(default_factory=dict)
    #: (source, raw_row, reason) for anything that would not parse.
    rejected: list[tuple[str, dict, str]] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return len(self.settlements) + len(self.banks) + len(self.ledger)

    def summary(self) -> str:
        s = (f"{self.total_rows} rows "
             f"(settlement {len(self.settlements)} / bank {len(self.banks)} / "
             f"ledger {len(self.ledger)})")
        if self.rejected:
            s += f", {len(self.rejected)} rejected"
        return s


def load_batch(directory: str | Path) -> LoadedBatch:
    """Read settlement_report.csv, bank_statement.csv and internal_ledger.csv."""
    root = Path(directory)
    batch = LoadedBatch()

    with open(root / "settlement_report.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gross = rupees_to_paise(row.get("gross_amount"))
            net = rupees_to_paise(row.get("net_amount"))
            sdate = parse_date(row.get("settlement_date"))
            if gross is None or net is None or sdate is None:
                batch.rejected.append(("settlement", row, "unparseable amount or date"))
                continue
            batch.settlements.append(SettlementRecord(
                transaction_id=row["transaction_id"],
                order_id=row.get("order_id") or None,
                gross_paise=gross,
                fee_paise=rupees_to_paise(row.get("fee")) or 0,
                tax_on_fee_paise=rupees_to_paise(row.get("tax_on_fee")) or 0,
                net_paise=net,
                settlement_date=sdate,
                utr_number=utr_digits(row.get("utr_number")),
            ))

    with open(root / "bank_statement.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            credit = rupees_to_paise(row.get("credit_amount"))
            vdate = parse_date(row.get("date"))
            if credit is None or vdate is None:
                batch.rejected.append(("bank", row, "unparseable amount or date"))
                continue
            digits, provenance = recover_utr(row.get("utr_reference"),
                                             row.get("description"))
            batch.banks.append(BankRecord(
                bank_row_id=row["bank_row_id"],
                value_date=vdate,
                description=row.get("description", ""),
                credit_paise=credit,
                utr_reference=row.get("utr_reference") or None,
                parsed_utr=digits,
                utr_provenance=provenance,
            ))

    with open(root / "internal_ledger.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expected = rupees_to_paise(row.get("expected_amount"))
            odate = parse_date(row.get("order_date"))
            if expected is None or odate is None:
                batch.rejected.append(("ledger", row, "unparseable amount or date"))
                continue
            batch.ledger[row["order_id"]] = LedgerRecord(
                order_id=row["order_id"],
                customer=row.get("customer", ""),
                expected_paise=expected,
                order_date=odate,
                status=row.get("status", "unknown"),
            )

    return batch
