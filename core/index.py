"""Amount-and-date index over bank credits.

The first implementation of the engine answered "which credits could belong to
this settlement?" by scanning every credit in the batch, once per settlement, in
three separate tiers. On 252 rows that is invisible. Scaling the benchmark showed
what it really was: `_amount_ok` called 21.5 million times on a 25,000-row batch,
matching taking 17 seconds, and throughput falling from 132,000 rows/sec to 2,700
as the batch grew. Quadratic behaviour hiding behind a small test set - the same
shape of mistake as the underpowered precision metric, in a different dimension.

This index answers the same question by construction. Credits are bucketed by
value date, and each bucket is sorted by amount, so a query becomes a couple of
binary searches per day in the window instead of a full scan.

**It changes speed and nothing else.** Every query returns exactly the set the
scan returned - the callers still apply their own predicates, and this only
avoids visiting rows that could not possibly qualify. The equivalence is asserted
directly in `tests/test_index.py`, which brute-force scans the same batch and
compares, and again end-to-end by comparing full engine output before and after.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, timedelta
from typing import Iterable, Iterator

from core.models import BankRecord


class CreditIndex:
    """Bank credits bucketed by value date, each bucket sorted by amount."""

    def __init__(self, banks: Iterable[BankRecord]):
        buckets: dict[date, list[BankRecord]] = defaultdict(list)
        for b in banks:
            buckets[b.value_date].append(b)

        self._records: dict[date, list[BankRecord]] = {}
        self._amounts: dict[date, list[int]] = {}
        for day, rows in buckets.items():
            # Sorted by amount, then id - the id keeps the order total, so the
            # index yields the same sequence for any input ordering.
            rows.sort(key=lambda b: (b.credit_paise, b.bank_row_id))
            self._records[day] = rows
            self._amounts[day] = [b.credit_paise for b in rows]

        self._total = sum(len(v) for v in self._records.values())

    def __len__(self) -> int:
        return self._total

    @property
    def dates(self) -> list[date]:
        return sorted(self._records)

    # ------------------------------------------------------------------

    def on_date(self, day: date, lo_paise: int, hi_paise: int) -> Iterator[BankRecord]:
        """Credits on one date with an amount in [lo, hi]."""
        amounts = self._amounts.get(day)
        if not amounts or lo_paise > hi_paise:
            return
        rows = self._records[day]
        for i in range(bisect_left(amounts, lo_paise), bisect_right(amounts, hi_paise)):
            yield rows[i]

    def query(self, anchor: date, *, lag_from: int, lag_to: int,
              lo_paise: int, hi_paise: int,
              exclude: set[str] | frozenset[str] = frozenset()) -> Iterator[BankRecord]:
        """Credits within a lag range of `anchor` and an amount range.

        `exclude` is the set of already-claimed bank row IDs. Filtering here
        rather than rebuilding an "unclaimed" list per settlement is most of the
        win: the old code allocated a fresh list of every unclaimed credit 6,229
        times in a single 25,000-row run.
        """
        if lag_from > lag_to or lo_paise > hi_paise:
            return
        for lag in range(lag_from, lag_to + 1):
            for row in self.on_date(anchor + timedelta(days=lag), lo_paise, hi_paise):
                if row.bank_row_id not in exclude:
                    yield row

    def count(self, anchor: date, *, lag_from: int, lag_to: int,
              lo_paise: int, hi_paise: int, ignore: str | None = None) -> int:
        """How many credits fall in the box. Used by the coincidence guard.

        Counts claimed credits too - the guard is measuring how crowded the
        neighbourhood is, and a credit being spoken for does not make the
        neighbourhood less crowded.
        """
        n = 0
        for lag in range(lag_from, lag_to + 1):
            for row in self.on_date(anchor + timedelta(days=lag), lo_paise, hi_paise):
                if row.bank_row_id != ignore:
                    n += 1
        return n
