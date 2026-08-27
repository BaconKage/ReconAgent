"""Domain model for reconciliation.

Money is represented as **integer paise** everywhere inside the system. The CSVs
carry rupee decimal strings because that is what real bank and gateway exports
look like, but the moment a value crosses into the engine it becomes an int.
There is no float arithmetic on money anywhere in `core/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from enum import Enum
from typing import Any, Literal


# --------------------------------------------------------------------------
# Case taxonomy — shared by the generator (as ground truth) and the evaluator.
# The matcher never sees these.
# --------------------------------------------------------------------------

class CaseType(str, Enum):
    CLEAN = "clean"
    FEE_DEDUCTION = "fee_deduction"
    TIMING_LAG = "timing_lag"
    SPLIT_SETTLEMENT = "split_settlement"
    PARTIAL_REFUND = "partial_refund"
    DUPLICATE = "duplicate"
    ROUNDING = "rounding"
    ADVERSARIAL_RESOLVABLE = "adversarial_resolvable"
    ADVERSARIAL_AMBIGUOUS = "adversarial_ambiguous"
    UNMATCHABLE = "unmatchable"


class Resolution(str, Enum):
    """What *should* happen to a group. Ground-truth label."""
    MATCHED = "matched"
    MATCHED_SPLIT = "matched_split"
    EXCEPTION_PARTIAL_REFUND = "exception_partial_refund"
    EXCEPTION_DUPLICATE = "exception_duplicate"
    EXCEPTION_AMBIGUOUS = "exception_ambiguous"
    EXCEPTION_UNMATCHABLE = "exception_unmatchable"

    @property
    def is_auto_matchable(self) -> bool:
        return self in (Resolution.MATCHED, Resolution.MATCHED_SPLIT)


# --------------------------------------------------------------------------
# Source records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SettlementRecord:
    transaction_id: str
    order_id: str | None
    gross_paise: int
    fee_paise: int
    tax_on_fee_paise: int
    net_paise: int
    settlement_date: date
    utr_number: str | None

    @property
    def record_id(self) -> str:
        return self.transaction_id


@dataclass(frozen=True)
class BankRecord:
    bank_row_id: str
    value_date: date
    description: str
    credit_paise: int
    utr_reference: str | None
    #: UTR digits recovered by normalize.py from the reference column or the
    #: narration text. May be the wrong length when the source truncated it.
    parsed_utr: str | None = None
    #: How it was recovered - lands in the audit trail so a reviewer can see
    #: whether a match rested on a clean field or a regex dig through narration.
    utr_provenance: str = "absent"

    @property
    def record_id(self) -> str:
        return self.bank_row_id


@dataclass(frozen=True)
class LedgerRecord:
    order_id: str
    customer: str
    expected_paise: int          # gross — will NOT equal the bank credit
    order_date: date
    status: str                  # paid | refunded | partially_refunded | pending

    @property
    def record_id(self) -> str:
        return self.order_id


# --------------------------------------------------------------------------
# Matcher output
# --------------------------------------------------------------------------

MatchStatus = Literal["matched", "matched_split", "duplicate", "unresolved"]


@dataclass
class NearMiss:
    """A candidate the matcher considered and declined.

    This is the raw material the reasoning layer works from. Without it the LLM
    would be guessing; with it, every hypothesis is grounded in a record that
    actually exists in the batch.
    """
    source: str                  # "bank" | "settlement" | "ledger"
    record_id: str
    reason: str                  # why it was declined, in engine terms
    amount_delta_paise: int | None = None
    date_delta_days: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchResult:
    record_id: str
    status: MatchStatus
    match_type: str | None = None            # which tier fired
    confidence: float = 0.0
    linked_ids: dict[str, list[str]] = field(
        default_factory=lambda: {"settlement": [], "bank": [], "ledger": []}
    )
    near_misses: list[NearMiss] = field(default_factory=list)
    rule_trace: list[str] = field(default_factory=list)
    #: Machine-readable category when status == "unresolved". This is the
    #: engine's structural finding, not an explanation - explaining it is the
    #: reasoning layer's job, and it works from this plus the near misses.
    exception_reason: str | None = None

    def note(self, msg: str) -> None:
        """Append to the rule trace. Every tier logs, whether it fires or not."""
        self.rule_trace.append(msg)

    @property
    def group_key(self) -> frozenset[str]:
        """Canonical identity of the reconciliation group this record landed in.

        Evaluation compares these as sets, because reconciliation is a *linking*
        problem: getting 2 of 3 legs of a split right is not 2/3 correct, it is
        a wrong group.
        """
        members: list[str] = []
        for ids in self.linked_ids.values():
            members.extend(ids)
        return frozenset(members)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["near_misses"] = [nm.to_dict() for nm in self.near_misses]
        return d
