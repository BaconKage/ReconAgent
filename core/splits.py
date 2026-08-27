"""Bounded subset-sum for split-settlement detection.

A single settlement is often paid out as several bank credits. Detecting that
means finding a subset of candidate credits whose sum equals the settlement net.

Two properties matter more than raw speed:

1. **It must find ALL solutions, not the first one.** If two different subsets
   both hit the target, the data does not identify which is real, and binding
   either would be a coin flip dressed up as a match. The search therefore stops
   as soon as it has found enough solutions to prove ambiguity.

2. **It must be bounded.** Subset-sum is NP-complete in general. The bounds here
   (leg count, minimum leg size, date window applied by the caller) keep the
   search tiny on realistic batches and, more importantly, keep the
   false-positive rate down: given enough candidates, *something* always sums to
   the target by coincidence.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitCandidate:
    """One way of decomposing a settlement into bank credits."""
    bank_row_ids: tuple[str, ...]
    total_paise: int
    delta_paise: int          # signed: total - target

    @property
    def leg_count(self) -> int:
        return len(self.bank_row_ids)


def find_split_candidates(
    legs: list[tuple[str, int]],
    target_paise: int,
    *,
    tolerance_paise: int,
    max_legs: int,
    min_leg_paise: int = 0,
    max_solutions: int = 4,
) -> list[SplitCandidate]:
    """Find subsets of `legs` summing to `target_paise` within tolerance.

    Args:
        legs: (bank_row_id, credit_paise) pairs, already filtered by the caller
            to a plausible pool (e.g. inside the date window).
        target_paise: the settlement net to reconstruct.
        tolerance_paise: acceptable absolute deviation from target.
        max_legs: maximum subset size to consider.
        min_leg_paise: ignore credits smaller than this.
        max_solutions: stop once this many distinct subsets are found. Two is
            already enough to declare ambiguity; a few more make the audit trail
            more informative when a human reviews it.

    Returns:
        Distinct subsets of size >= 2, best (smallest deviation, then fewest
        legs) first. A single-leg "split" is not a split - that is an ordinary
        one-to-one match and belongs to an earlier tier.
    """
    pool = [(rid, amt) for rid, amt in legs
            if amt >= min_leg_paise and amt <= target_paise + tolerance_paise]
    # Descending by amount, then by id. The id tiebreak matters: without it the
    # result depends on the caller's iteration order, so an indexing change that
    # merely reorders the input could silently change which subset is found.
    pool.sort(key=lambda x: (-x[1], x[0]))

    # suffix_sums[i] = sum of every amount from i onward. Lets us abandon a
    # branch the moment even taking everything left cannot reach the target.
    n = len(pool)
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + pool[i][1]

    # Amounts negated so the descending pool reads as ascending, which lets
    # bisect find the first leg small enough to fit. Without it the search walks
    # past every too-large leg one at a time; on a dense batch that scan was 71%
    # of total matching time. Skipping legs the loop would have rejected anyway,
    # so the result is unchanged - only the work to reach it.
    neg_amounts = [-amt for _, amt in pool]

    found: list[SplitCandidate] = []
    seen: set[tuple[str, ...]] = set()

    def record(chosen: list[int], total: int) -> None:
        ids = tuple(sorted(pool[i][0] for i in chosen))
        if ids in seen:
            return
        seen.add(ids)
        found.append(SplitCandidate(ids, total, total - target_paise))

    def dfs(start: int, chosen: list[int], total: int) -> None:
        if len(found) >= max_solutions:
            return
        if len(chosen) >= 2 and abs(total - target_paise) <= tolerance_paise:
            record(chosen, total)
            # Do not return: a superset could also land inside tolerance, and
            # discovering that is exactly how ambiguity gets detected.
        if len(chosen) >= max_legs:
            return
        # Jump straight past the legs that would overshoot rather than testing
        # each one. Everything before this index is too large to fit.
        capacity = target_paise + tolerance_paise - total
        first = max(start, bisect_left(neg_amounts, -capacity))
        for i in range(first, n):
            amt = pool[i][1]
            new_total = total + amt
            # Even consuming the entire remaining tail cannot reach the target.
            if total + suffix_sums[i] < target_paise - tolerance_paise:
                return
            chosen.append(i)
            dfs(i + 1, chosen, new_total)
            chosen.pop()
            if len(found) >= max_solutions:
                return

    if target_paise > 0 and pool:
        dfs(0, [], 0)

    found.sort(key=lambda c: (abs(c.delta_paise), c.leg_count, c.bank_row_ids))
    return found
