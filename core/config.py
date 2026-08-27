"""Matching thresholds.

These live in one place, and deliberately so: the honesty of the held-out
evaluation depends on being able to point at exactly what was tuned and when it
was frozen. Changing a number here changes the reported metrics, so any change
should be visible in the diff rather than buried in a function body.

All monetary values are integer paise. There is no float money anywhere in core.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchConfig:
    #: Two amounts are "the same" within this many paise. Covers currency
    #: rounding and paisa-level drift between gateway and bank ledgers.
    amount_tolerance_paise: int = 50

    #: A credit may land on the settlement date or up to this many days after.
    #: Never before - money cannot arrive at the bank before it is settled, and
    #: allowing negative lag would roughly double the candidate pool for nothing.
    date_window_days: int = 2

    #: Upper bound on legs in a split settlement. Beyond this the subset-sum
    #: search space grows fast and the false-positive risk grows faster: with
    #: enough legs some subset will always hit the target by chance.
    max_split_legs: int = 4

    #: A split leg must be at least this fraction of the total, which rules out
    #: absurd decompositions like "one paisa plus the rest".
    min_split_leg_fraction: float = 0.02

    #: Below this, a match is real but worth a second look. These are the rows
    #: the reasoning layer sanity-checks; everything above never reaches an LLM.
    low_confidence_threshold: float = 0.80

    #: A candidate outside the hard thresholds but inside these is recorded as a
    #: near miss - not matched, but handed to the reasoning layer as evidence.
    #: This is what lets the agent say "there IS something close, and here is
    #: precisely why I still will not bind it".
    near_miss_amount_paise: int = 50_00      # Rs 50
    near_miss_window_days: int = 7

    # -- coincidence guard ---------------------------------------------
    # An amount-and-date match carries no identifier, so it is only as good as
    # the odds against a coincidence. Scaling the benchmark to 25,000 records
    # showed that every false positive the engine produces comes from this tier:
    # a settlement that should have been a split, or should have had no
    # counterpart at all, bound to one unrelated credit that happened to land
    # inside both thresholds. The existing ambiguity guard fires on *two or more*
    # candidates and has no answer to exactly one.
    #
    # The guard estimates the expected number of unrelated credits at least as
    # close as the one matched:
    #
    #     score = neighbours_within_radius x (delta / radius) x ((lag + 1) / window)
    #
    # A credit that matches the net exactly, on the settlement date, in a sparse
    # neighbourhood scores ~0 and is accepted. One sitting mid-tolerance, a day
    # or two late, in a crowded neighbourhood scores high - which is precisely
    # what a coincidence looks like, since coincidental deltas are spread evenly
    # across the tolerance band while real ones cluster at zero.

    #: Reference radius for counting neighbours. Rs 50 - two orders of magnitude
    #: wider than the match tolerance, so it measures local density rather than
    #: re-testing the match.
    coincidence_radius_paise: int = 50_00

    #: Above this expected-coincidence score, an amount+date match is held for
    #: review instead of auto-matched. Set from the measured trade-off curve at
    #: 25k records: it catches about half the false positives for roughly an
    #: equal number of demoted true matches. That trade is deliberate - binding
    #: money to the wrong counterpart costs far more than asking a human to look.
    #: Set to 0 to disable the guard entirely.
    coincidence_threshold: float = 0.05


DEFAULT_CONFIG = MatchConfig()
