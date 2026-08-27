"""The deterministic reconciliation engine.

This module decides what matches what. It contains no LLM calls, no network
access and no randomness: given the same CSVs and the same MatchConfig it
produces byte-identical output. Correctness lives here; the reasoning layer
downstream only explains what this module already decided.

The cascade
-----------
Tiers run in descending order of evidence quality, and each tier consumes the
rows it claims before the next tier runs. **Cascade order is itself a
correctness mechanism**, not just an optimisation: resolving strong identifiers
first removes exactly the rows that would otherwise create false amount-based
ties later.

    0. Duplicate detection      - webhook retries, before anything is matched
    1. Exact UTR                - authoritative identifier, both sides clean
    2. Repaired / truncated UTR - prefix resolution, refused when not unique
    3. Amount + date window     - constraint propagation, never nearest-guess
    4. Split settlement         - bounded subset-sum, refused when not unique
    5. Everything left          - unresolved, with near misses attached

The refusal principle
---------------------
When two candidates both satisfy the thresholds, this engine claims *neither*.
It never breaks a tie by picking the closer amount. That rule costs recall on
genuinely ambiguous rows, and it is the single reason the false-positive rate
stays where it does: on a near-duplicate pair, "pick the closest" is a coin flip
that scores 50% and reports 100% confidence.
"""

from __future__ import annotations

import time
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from core.config import DEFAULT_CONFIG, MatchConfig
from core.index import CreditIndex
from core.loader import LoadedBatch
from core.models import BankRecord, MatchResult, NearMiss, SettlementRecord
from core.normalize import paise_to_rupees
from core.splits import find_split_candidates

UTR_LEN = 12


@dataclass
class ReconciliationReport:
    results: list[MatchResult] = field(default_factory=list)
    config: MatchConfig = DEFAULT_CONFIG
    elapsed_seconds: float = 0.0
    rows_processed: int = 0

    def by_status(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in self.results:
            out[r.status] += 1
        return dict(out)

    def by_exception_reason(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in self.results:
            if r.status == "unresolved" and r.exception_reason:
                out[r.exception_reason] += 1
        return dict(out)

    @property
    def unresolved(self) -> list[MatchResult]:
        return [r for r in self.results if r.status == "unresolved"]

    @property
    def low_confidence(self) -> list[MatchResult]:
        return [r for r in self.results
                if r.status != "unresolved"
                and r.confidence < self.config.low_confidence_threshold]

    @property
    def throughput(self) -> float:
        return self.rows_processed / self.elapsed_seconds if self.elapsed_seconds else 0.0


class ReconciliationEngine:
    def __init__(self, config: MatchConfig = DEFAULT_CONFIG):
        self.cfg = config

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, batch: LoadedBatch) -> ReconciliationReport:
        started = time.perf_counter()

        self.batch = batch
        self.claimed_banks: set[str] = set()
        self.results: dict[str, MatchResult] = {}
        self.banks_by_id = {b.bank_row_id: b for b in batch.banks}
        # One index serves every tier: amount-and-date lookups become a pair of
        # binary searches instead of a scan over the whole batch per settlement.
        self.index = CreditIndex(batch.banks)

        settlements, duplicate_ids = self._detect_duplicates(batch.settlements)

        self._tier1_exact_utr(settlements)
        self._tier2_repaired_utr(settlements)
        self._tier3_amount_and_date(settlements)
        self._tier4_split_settlement(settlements)
        self._tier5_leftovers(settlements)
        self._orphan_bank_rows()

        for tid in duplicate_ids:
            if tid in self.results:
                r = self.results[tid]
                r.status = "duplicate"
                r.exception_reason = "duplicate_settlement_row"
                r.confidence = min(r.confidence, 0.99)

        report = ReconciliationReport(
            results=list(self.results.values()),
            config=self.cfg,
            elapsed_seconds=time.perf_counter() - started,
            rows_processed=batch.total_rows,
        )
        return report

    # ------------------------------------------------------------------
    # Result plumbing
    # ------------------------------------------------------------------

    def _result_for(self, s: SettlementRecord) -> MatchResult:
        """Fetch or create the result for a settlement, pre-linked to its ledger row."""
        if s.transaction_id in self.results:
            return self.results[s.transaction_id]
        r = MatchResult(record_id=s.transaction_id, status="unresolved")
        r.linked_ids["settlement"] = [s.transaction_id]
        if s.order_id and s.order_id in self.batch.ledger:
            r.linked_ids["ledger"] = [s.order_id]
            r.note(f"ledger: order {s.order_id} linked by exact order_id")
        elif s.order_id:
            r.note(f"ledger: order_id {s.order_id} not present in internal ledger")
        else:
            r.note("ledger: settlement carries no order_id")
        self.results[s.transaction_id] = r
        return r

    def _claim(self, r: MatchResult, banks: list[BankRecord]) -> None:
        for b in banks:
            self.claimed_banks.add(b.bank_row_id)
            r.linked_ids["bank"].append(b.bank_row_id)

    def _unclaimed(self) -> list[BankRecord]:
        return [b for b in self.batch.banks if b.bank_row_id not in self.claimed_banks]

    def _is_settled(self, s: SettlementRecord) -> bool:
        r = self.results.get(s.transaction_id)
        return r is not None and r.status != "unresolved"

    def _is_decided(self, s: SettlementRecord) -> bool:
        """Settled, or deliberately parked as an exception with a reason."""
        r = self.results.get(s.transaction_id)
        return r is not None and (r.status != "unresolved" or r.exception_reason is not None)

    def _lag(self, s: SettlementRecord, b: BankRecord) -> int:
        return (b.value_date - s.settlement_date).days

    def _amount_ok(self, s: SettlementRecord, b: BankRecord) -> bool:
        return abs(b.credit_paise - s.net_paise) <= self.cfg.amount_tolerance_paise

    def _window_ok(self, s: SettlementRecord, b: BankRecord) -> bool:
        return 0 <= self._lag(s, b) <= self.cfg.date_window_days

    # ------------------------------------------------------------------
    # Tier 0 - duplicates
    # ------------------------------------------------------------------

    def _detect_duplicates(
        self, settlements: list[SettlementRecord]
    ) -> tuple[list[SettlementRecord], set[str]]:
        """Collapse webhook-retry duplicates before any matching happens.

        Running this first matters: two identical settlement rows competing for
        one bank credit would otherwise look like genuine contention and poison
        the ambiguity guard in tier 3.
        """
        by_txn: dict[str, list[SettlementRecord]] = defaultdict(list)
        for s in settlements:
            by_txn[s.transaction_id].append(s)

        # Same money, same UTR, same day, different transaction_id - a retry that
        # was assigned a fresh id.
        by_fingerprint: dict[tuple, list[SettlementRecord]] = defaultdict(list)
        for tid, rows in by_txn.items():
            s = rows[0]
            by_fingerprint[(s.utr_number, s.net_paise, s.settlement_date)].append(s)

        duplicate_ids: set[str] = set()
        canonical: list[SettlementRecord] = []
        for tid, rows in by_txn.items():
            if len(rows) > 1:
                duplicate_ids.add(tid)
            canonical.append(rows[0])

        for key, rows in by_fingerprint.items():
            if key[0] and len(rows) > 1:
                for s in rows:
                    duplicate_ids.add(s.transaction_id)

        # Keep a stable order so runs are reproducible.
        canonical.sort(key=lambda s: (s.settlement_date, s.transaction_id))
        return canonical, duplicate_ids

    # ------------------------------------------------------------------
    # Tier 1 - exact UTR
    # ------------------------------------------------------------------

    def _tier1_exact_utr(self, settlements: list[SettlementRecord]) -> None:
        by_utr: dict[str, list[BankRecord]] = defaultdict(list)
        for b in self.batch.banks:
            if b.parsed_utr and len(b.parsed_utr) == UTR_LEN:
                by_utr[b.parsed_utr].append(b)

        for s in settlements:
            r = self._result_for(s)
            if not s.utr_number:
                r.note("tier1 exact-UTR: skipped, settlement has no UTR")
                continue
            hits = [b for b in by_utr.get(s.utr_number, [])
                    if b.bank_row_id not in self.claimed_banks]
            if not hits:
                r.note("tier1 exact-UTR: no bank row carries this UTR")
                continue

            if len(hits) == 1:
                b = hits[0]
                delta = b.credit_paise - s.net_paise
                if abs(delta) <= self.cfg.amount_tolerance_paise:
                    self._claim(r, [b])
                    r.status = "matched"
                    r.match_type = "exact_utr"
                    r.confidence = 1.0
                    r.note(f"tier1 exact-UTR: matched {b.bank_row_id} "
                           f"(via {b.utr_provenance}), amount delta {delta}p")
                else:
                    # The identifier is authoritative, so these rows DO belong
                    # together - but the money does not reconcile. Claiming the
                    # bank row here is deliberate: it stops a later amount-based
                    # tier from binding this credit to some unrelated settlement.
                    self._claim(r, [b])
                    r.status = "unresolved"
                    r.exception_reason = "identifier_match_amount_discrepancy"
                    r.confidence = 0.0
                    r.match_type = "exact_utr_amount_mismatch"
                    r.near_misses.append(NearMiss(
                        source="bank", record_id=b.bank_row_id,
                        reason="UTR matches but credited amount differs from settlement net",
                        amount_delta_paise=delta, date_delta_days=self._lag(s, b)))
                    r.note(f"tier1 exact-UTR: {b.bank_row_id} matches on UTR but amount "
                           f"differs by {delta}p (tolerance "
                           f"{self.cfg.amount_tolerance_paise}p) - held as an exception")
                continue

            # Several credits carry the same UTR: legitimate if they sum to net.
            total = sum(b.credit_paise for b in hits)
            if abs(total - s.net_paise) <= self.cfg.amount_tolerance_paise:
                self._claim(r, hits)
                r.status = "matched_split"
                r.match_type = "exact_utr_split"
                r.confidence = 0.97
                r.note(f"tier1 exact-UTR: {len(hits)} credits share this UTR and sum to net")
            else:
                r.note(f"tier1 exact-UTR: {len(hits)} credits share this UTR but do not "
                       f"sum to net - deferring")

    # ------------------------------------------------------------------
    # Tier 2 - repaired / truncated UTR
    # ------------------------------------------------------------------

    def _tier2_repaired_utr(self, settlements: list[SettlementRecord]) -> None:
        """Resolve UTRs the bank mangled - typically truncated by a digit.

        A short UTR cannot be repaired from the token alone, so it is resolved by
        prefix against the settlement UTRs actually present in this batch. If the
        prefix fits more than one settlement, it is refused: a truncated
        identifier that matches two candidates identifies neither.
        """
        pending = [s for s in settlements if not self._is_decided(s) and s.utr_number]
        if not pending:
            return
        utr_to_settlements: dict[str, list[SettlementRecord]] = defaultdict(list)
        for s in pending:
            utr_to_settlements[s.utr_number].append(s)

        # Sorted once, so a prefix lookup is a binary search rather than a scan
        # over every pending UTR. Scanning made this tier quadratic: on a
        # 50,000-row batch it was 1,980 malformed references x 17,000 settlement
        # UTRs. The sorted order also means the two candidates examined below are
        # always the same two, whatever order the credits arrive in.
        sorted_utrs = sorted(utr_to_settlements)

        for b in self._unclaimed():
            digits = b.parsed_utr
            if not digits or len(digits) == UTR_LEN:
                continue
            # Only ever need to know whether the prefix is unique, so stop at two.
            start = bisect_left(sorted_utrs, digits)
            matches = []
            for i in range(start, min(start + 2, len(sorted_utrs))):
                if sorted_utrs[i].startswith(digits):
                    matches.append(sorted_utrs[i])
                else:
                    break
            if len(matches) != 1:
                continue
            cands = [s for s in utr_to_settlements[matches[0]] if not self._is_decided(s)]
            if len(cands) != 1:
                continue
            s = cands[0]
            r = self._result_for(s)
            delta = b.credit_paise - s.net_paise
            if abs(delta) > self.cfg.amount_tolerance_paise:
                r.note(f"tier2 repaired-UTR: {b.bank_row_id} prefix-resolves to this "
                       f"settlement but amount differs by {delta}p - not claimed")
                continue
            self._claim(r, [b])
            r.status = "matched"
            r.match_type = "repaired_utr"
            r.confidence = 0.95
            r.note(f"tier2 repaired-UTR: '{digits}' ({len(digits)} digits, truncated) "
                   f"uniquely prefixes settlement UTR {s.utr_number}; "
                   f"amount delta {delta}p, lag T+{self._lag(s, b)}")

    # ------------------------------------------------------------------
    # Tier 3 - amount + date, by constraint propagation
    # ------------------------------------------------------------------

    def _tier3_amount_and_date(self, settlements: list[SettlementRecord]) -> None:
        pending = [s for s in settlements if not self._is_decided(s)]
        if not pending:
            return

        tol = self.cfg.amount_tolerance_paise
        candidates: dict[str, set[str]] = {}
        for s in pending:
            candidates[s.transaction_id] = {
                b.bank_row_id for b in self.index.query(
                    s.settlement_date, lag_from=0, lag_to=self.cfg.date_window_days,
                    lo_paise=s.net_paise - tol, hi_paise=s.net_paise + tol,
                    exclude=self.claimed_banks)
            }

        by_tid = {s.transaction_id: s for s in pending}

        # Singleton elimination. A settlement is bound only when it has exactly
        # one surviving candidate AND no other settlement is also down to that
        # same single candidate. Assignments shrink other candidate sets, so this
        # is iterated to a fixed point - which is what lets a forced pairing
        # cascade into resolving its neighbour.
        changed = True
        while changed:
            changed = False
            forced = {tid: next(iter(c - self.claimed_banks))
                      for tid, c in candidates.items()
                      if tid in by_tid and not self._is_decided(by_tid[tid])
                      and len(c - self.claimed_banks) == 1}
            contested: set[str] = set()
            seen: dict[str, str] = {}
            for tid, bid in forced.items():
                if bid in seen:
                    contested.add(bid)
                seen[bid] = tid

            for tid, bid in forced.items():
                if bid in contested:
                    continue
                s = by_tid[tid]
                b = self.banks_by_id[bid]
                r = self._result_for(s)

                score = self._coincidence_score(s, b)
                if score > self.cfg.coincidence_threshold > 0:
                    # Unique, inside both thresholds - and still not good enough.
                    # The credit is left unclaimed so it stays available to a
                    # settlement that can account for it more convincingly.
                    r.status = "unresolved"
                    r.exception_reason = "weak_amount_date_evidence"
                    r.confidence = 0.0
                    r.near_misses.append(NearMiss(
                        source="bank", record_id=bid,
                        reason=("only candidate inside both thresholds, but the match is "
                                "loose enough and the neighbourhood crowded enough that a "
                                "coincidence is likely"),
                        amount_delta_paise=b.credit_paise - s.net_paise,
                        date_delta_days=self._lag(s, b)))
                    r.note(f"tier3 amount+date: {bid} is the only candidate, but the "
                           f"expected-coincidence score is {score:.3f} against a limit of "
                           f"{self.cfg.coincidence_threshold}. Amount and date alone are "
                           f"not enough evidence here - held for review.")
                    continue

                self._claim(r, [b])
                r.status = "matched"
                r.match_type = "amount_date_window"
                r.confidence = self._confidence(s, b)
                r.note(f"tier3 amount+date: uniquely matched {bid} "
                       f"(delta {b.credit_paise - s.net_paise}p, "
                       f"lag T+{self._lag(s, b)}); no competing candidate, "
                       f"coincidence score {score:.3f}")
                changed = True

        # Whatever is left is unclaimable for one of two distinct reasons, and
        # the exception list is more useful when it says which.
        for s in pending:
            if self._is_decided(s):
                continue
            remaining = candidates[s.transaction_id] - self.claimed_banks
            if not remaining:
                continue                      # tiers 4 and 5 still get a turn
            r = self._result_for(s)
            r.status = "unresolved"
            r.confidence = 0.0
            for bid in sorted(remaining):
                b = self.banks_by_id[bid]
                r.near_misses.append(NearMiss(
                    source="bank", record_id=bid,
                    reason="satisfies both amount tolerance and date window",
                    amount_delta_paise=b.credit_paise - s.net_paise,
                    date_delta_days=self._lag(s, b)))

            if len(remaining) == 1:
                # Propagation ran to a fixed point, so a lone surviving candidate
                # that still went unclaimed can only mean another settlement is
                # also down to this same credit. One credit, two claimants.
                bid = next(iter(remaining))
                r.exception_reason = "contested_candidate"
                r.note(f"tier3 amount+date: {bid} is this settlement's only candidate, but "
                       f"another settlement has no other candidate either. One credit "
                       f"cannot settle both - refusing both rather than picking one.")
            else:
                r.exception_reason = "ambiguous_candidates"
                r.note(f"tier3 amount+date: {len(remaining)} credits satisfy both thresholds "
                       f"({', '.join(sorted(remaining))}). Refusing to guess - picking the "
                       f"closest would be a coin flip reported as a match.")

    def _coincidence_score(self, s: SettlementRecord, b: BankRecord) -> float:
        """Expected number of unrelated credits at least as close as this one.

        An amount-and-date match carries no identifier, so its worth depends
        entirely on how unlikely it is to have happened by chance. Three factors
        multiply:

        * **neighbourhood density** - how many other credits sit near this net
          inside the window. A crowded neighbourhood makes any single hit cheap.
        * **how close the match is** - coincidental deltas are spread evenly
          across the tolerance band, real ones cluster at zero, so a delta near
          the tolerance limit is itself evidence of coincidence.
        * **the lag** - same-day credits are far more often genuine.

        Zero means "nothing else was anywhere near this, and it matched exactly"
        - accept. High means "in a field this crowded, a hit this loose was
        always going to happen" - hold it for a human.
        """
        radius = self.cfg.coincidence_radius_paise
        if radius <= 0:
            return 0.0
        neighbours = self.index.count(
            s.settlement_date, lag_from=0, lag_to=self.cfg.date_window_days,
            lo_paise=s.net_paise - radius, hi_paise=s.net_paise + radius,
            ignore=b.bank_row_id)
        delta = max(abs(b.credit_paise - s.net_paise), 1)
        lag = self._lag(s, b)
        window = max(1, self.cfg.date_window_days + 1)
        return neighbours * (delta / radius) * ((lag + 1) / window)

    def _confidence(self, s: SettlementRecord, b: BankRecord) -> float:
        """Confidence for an amount+date match, scaled by how tight the fit is."""
        amt = abs(b.credit_paise - s.net_paise) / max(1, self.cfg.amount_tolerance_paise)
        lag = self._lag(s, b) / max(1, self.cfg.date_window_days)
        return round(0.90 - 0.10 * amt - 0.05 * lag, 4)

    # ------------------------------------------------------------------
    # Tier 4 - split settlements
    # ------------------------------------------------------------------

    def _tier4_split_settlement(self, settlements: list[SettlementRecord]) -> None:
        pending = [s for s in settlements if not self._is_decided(s)]
        if not pending:
            return

        proposals: dict[str, tuple[SettlementRecord, tuple[str, ...], int]] = {}
        for s in pending:
            # The minimum-leg filter is applied inside find_split_candidates, not
            # here. Pre-filtering would be marginally faster and would change the
            # pool size, which decides whether the "no subset sums to net" line
            # is written to the rule trace at all - a difference a reviewer would
            # see. Speed is not worth quietly editing the audit trail.
            min_leg = int(s.net_paise * self.cfg.min_split_leg_fraction)
            pool = [(b.bank_row_id, b.credit_paise) for b in self.index.query(
                s.settlement_date, lag_from=0, lag_to=self.cfg.date_window_days,
                lo_paise=0, hi_paise=s.net_paise - 1,
                exclude=self.claimed_banks)]
            if len(pool) < 2:
                continue
            found = find_split_candidates(
                pool, s.net_paise,
                tolerance_paise=self.cfg.amount_tolerance_paise,
                max_legs=self.cfg.max_split_legs,
                min_leg_paise=min_leg,
                max_solutions=3,
            )
            r = self._result_for(s)
            if not found:
                r.note("tier4 split: no subset of credits in the window sums to net")
                continue
            if len(found) > 1:
                r.status = "unresolved"
                r.exception_reason = "ambiguous_split"
                r.confidence = 0.0
                for c in found[:3]:
                    r.near_misses.append(NearMiss(
                        source="bank", record_id="+".join(c.bank_row_ids),
                        reason=f"{c.leg_count} credits also sum to net",
                        amount_delta_paise=c.delta_paise))
                r.note(f"tier4 split: {len(found)} different subsets sum to net within "
                       f"tolerance - not identifiable, refusing to bind any of them")
                continue
            proposals[s.transaction_id] = (s, found[0].bank_row_ids, found[0].delta_paise)

        # Two settlements proposing overlapping legs cannot both be right, and
        # nothing in the data says which is. Both are refused.
        leg_owner: dict[str, list[str]] = defaultdict(list)
        for tid, (_, legs, _) in proposals.items():
            for leg in legs:
                leg_owner[leg].append(tid)
        contested = {tid for owners in leg_owner.values() if len(owners) > 1
                     for tid in owners}

        for tid, (s, legs, delta) in proposals.items():
            r = self._result_for(s)
            if tid in contested:
                r.status = "unresolved"
                r.exception_reason = "contested_split_legs"
                r.confidence = 0.0
                r.note(f"tier4 split: legs {', '.join(legs)} are also claimed by another "
                       f"settlement's split - refusing both")
                continue
            banks = [self.banks_by_id[b] for b in legs]
            self._claim(r, banks)
            r.status = "matched_split"
            r.match_type = "split_subset_sum"
            lag = max(self._lag(s, b) for b in banks)
            r.confidence = round(0.85 - 0.05 * (len(legs) - 2) - 0.05 * (lag / max(1, self.cfg.date_window_days)), 4)
            r.note(f"tier4 split: net reconstructed from {len(legs)} credits "
                   f"({', '.join(legs)}), delta {delta}p, latest leg T+{lag}; "
                   f"unique subset within tolerance")

    # ------------------------------------------------------------------
    # Tier 5 - leftovers, with near misses for the reasoning layer
    # ------------------------------------------------------------------

    def _tier5_leftovers(self, settlements: list[SettlementRecord]) -> None:
        for s in settlements:
            if self._is_decided(s):
                continue
            r = self._result_for(s)
            r.status = "unresolved"
            r.exception_reason = "no_candidate_found"
            r.confidence = 0.0
            near = self._near_misses_for(s)
            r.near_misses.extend(near)
            if near:
                closest = min(near, key=lambda n: abs(n.amount_delta_paise or 0))
                r.note(f"tier5: no credit satisfies both thresholds. Closest is "
                       f"{closest.record_id} at {closest.amount_delta_paise}p / "
                       f"T+{closest.date_delta_days} - outside the accepted window.")
            else:
                r.note("tier5: no credit anywhere in the batch resembles this settlement")

    def _near_misses_for(self, s: SettlementRecord) -> list[NearMiss]:
        """Credits that are close but outside the hard thresholds.

        These are evidence, not matches. Handing them to the reasoning layer is
        what lets it say precisely why it will not bind them, instead of just
        reporting an absence.
        """
        near = self.cfg.near_miss_amount_paise
        out: list[NearMiss] = []
        for b in self.index.query(
                s.settlement_date, lag_from=-1, lag_to=self.cfg.near_miss_window_days,
                lo_paise=s.net_paise - near, hi_paise=s.net_paise + near,
                exclude=self.claimed_banks):
            delta = b.credit_paise - s.net_paise
            lag = self._lag(s, b)
            why = []
            if abs(delta) > self.cfg.amount_tolerance_paise:
                why.append(f"amount off by {delta}p (tolerance "
                           f"{self.cfg.amount_tolerance_paise}p)")
            if not (0 <= lag <= self.cfg.date_window_days):
                why.append(f"lag T+{lag} (window 0..{self.cfg.date_window_days})")
            out.append(NearMiss(source="bank", record_id=b.bank_row_id,
                                reason="; ".join(why) or "inside thresholds but unclaimed",
                                amount_delta_paise=delta, date_delta_days=lag))
        out.sort(key=lambda n: (abs(n.amount_delta_paise or 0),
                                abs(n.date_delta_days or 0), n.record_id))
        return out[:5]

    # ------------------------------------------------------------------
    # Orphan bank rows
    # ------------------------------------------------------------------

    def _orphan_bank_rows(self) -> None:
        for b in self._unclaimed():
            r = MatchResult(record_id=b.bank_row_id, status="unresolved")
            r.linked_ids["bank"] = [b.bank_row_id]
            r.exception_reason = "no_settlement_counterpart"
            r.confidence = 0.0
            r.note(f"orphan credit: {b.bank_row_id} on {b.value_date} for "
                   f"Rs {paise_to_rupees(b.credit_paise)} was never claimed by any "
                   f"settlement (UTR provenance: {b.utr_provenance})")
            self.results[b.bank_row_id] = r


def reconcile(batch: LoadedBatch, config: MatchConfig = DEFAULT_CONFIG) -> ReconciliationReport:
    """Convenience wrapper: run the engine over a loaded batch."""
    return ReconciliationEngine(config).run(batch)
