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
    # Descending order makes the remaining-sum prune bite early.
    pool.sort(key=lambda x: -x[1])

    # suffix_sums[i] = sum of every amount from i onward. Lets us abandon a
    # branch the moment even taking everything left cannot reach the target.
    n = len(pool)
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + pool[i][1]

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
        for i in range(start, n):
            amt = pool[i][1]
            new_total = total + amt
            # Overshoot: everything after i is <= amt, so no later pick helps.
            if new_total - target_paise > tolerance_paise:
                continue
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

    found.sort(key=lambda c: (abs(c.delta_paise), c.leg_count))
    return found
