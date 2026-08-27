"""The reasoning layer: explains exceptions, never decides matches.

This module is the only place an LLM is used, and it runs strictly downstream of
the deterministic engine. It receives the cases the engine could not resolve
(plus any it resolved with low confidence) and produces, for each one, a
plain-English hypothesis, a confidence label and a recommended action.

Three properties are enforced here rather than merely intended:

* **It cannot change a match.** Nothing in this module writes to a MatchResult's
  status, confidence or linked IDs. `tests/test_reasoning_layer.py` asserts the
  reconciliation output is byte-identical with and without it.
* **It is not called on work that does not need it.** Cleanly matched records
  never reach the model. The run reports what fraction was handled without one.
* **It degrades instead of failing.** No API key, no SDK, or an API error yields
  a placeholder investigation and a run that still completes with full metrics.
* **It is not tied to one vendor.** The model is reached through `agent.llm`,
  which selects Anthropic or OpenAI from the environment. Swapping providers
  changes the wording of explanations and nothing else - not one match, not
  one metric.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from agent.cache import TraceCache, evidence_key
from agent.investigator import (investigate_case, needs_deep_investigation,
                                parse_first_object,
                                system_prompt as investigation_system_prompt)
from agent.llm import LLMUnavailable, get_provider
from agent.tools import BatchInvestigator
from agent.prompts import INVESTIGATION_SCHEMA, build_user_message, system_prompt
from core.config import DEFAULT_CONFIG, MatchConfig
from core.loader import LoadedBatch
from core.matcher import ReconciliationReport
from core.models import MatchResult
from core.normalize import paise_to_rupees

BATCH_SIZE = 6
MAX_TOKENS = 16000
#: Concurrent deep investigations. Independent work, so this is pure latency
#: reduction - kept modest to stay well inside provider rate limits.
DEEP_WORKERS = 6


@dataclass
class ReasoningStats:
    total_records: int = 0
    sent_to_llm: int = 0
    served_from_cache: int = 0
    live_calls: int = 0
    api_errors: int = 0
    unavailable: bool = False
    provider: str = ""
    elapsed_seconds: float = 0.0
    #: Cases that got a multi-turn tool-using investigation rather than a
    #: one-shot batched explanation.
    deep_investigations: int = 0
    deep_turns: int = 0

    @property
    def never_touched_llm(self) -> int:
        return self.total_records - self.sent_to_llm

    @property
    def llm_free_fraction(self) -> float:
        if not self.total_records:
            return 1.0
        return self.never_touched_llm / self.total_records


@dataclass
class ReasoningOutcome:
    investigations: dict[str, dict[str, Any]] = field(default_factory=dict)
    stats: ReasoningStats = field(default_factory=ReasoningStats)


# --------------------------------------------------------------------------
# Evidence bundles - what the model actually sees
# --------------------------------------------------------------------------

def build_evidence_bundle(result: MatchResult, batch: LoadedBatch,
                          cfg: MatchConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Assemble the real records behind one exception.

    Deliberately excludes anything the engine did not itself observe: no case
    type, no ground-truth label, no hint about what the answer should be. The
    model sees exactly the evidence the engine had.
    """
    settlements = {s.transaction_id: s for s in batch.settlements}
    banks = {b.bank_row_id: b for b in batch.banks}

    bundle: dict[str, Any] = {
        "case_id": result.record_id,
        "engine_finding": result.exception_reason or result.status,
        "engine_confidence": result.confidence,
        "engine_rule_trace": result.rule_trace,
    }

    s = settlements.get(result.record_id)
    if s:
        bundle["settlement"] = {
            "transaction_id": s.transaction_id,
            "gross": paise_to_rupees(s.gross_paise),
            "fee": paise_to_rupees(s.fee_paise),
            "gst_on_fee": paise_to_rupees(s.tax_on_fee_paise),
            "net_expected_in_bank": paise_to_rupees(s.net_paise),
            "settlement_date": s.settlement_date.isoformat(),
            "utr": s.utr_number,
        }

    for oid in result.linked_ids.get("ledger", []):
        led = batch.ledger.get(oid)
        if led:
            bundle["ledger"] = {
                "order_id": led.order_id,
                "customer": led.customer,
                "order_value_gross": paise_to_rupees(led.expected_paise),
                "order_date": led.order_date.isoformat(),
                "status": led.status,
            }

    linked = [banks[b] for b in result.linked_ids.get("bank", []) if b in banks]
    if linked:
        bundle["bank_credits_already_linked"] = [{
            "bank_row_id": b.bank_row_id,
            "credit": paise_to_rupees(b.credit_paise),
            "value_date": b.value_date.isoformat(),
            "narration": b.description,
            "utr_reference": b.utr_reference,
            "utr_recovered_via": b.utr_provenance,
        } for b in linked]

    if result.near_misses:
        out = []
        for nm in result.near_misses:
            row: dict[str, Any] = {
                "record_id": nm.record_id,
                "why_engine_declined_it": nm.reason,
            }
            if nm.amount_delta_paise is not None:
                row["amount_differs_by"] = paise_to_rupees(nm.amount_delta_paise)
            if nm.date_delta_days is not None:
                row["days_after_settlement"] = nm.date_delta_days
            b = banks.get(nm.record_id)
            if b:
                row.update({
                    "credit": paise_to_rupees(b.credit_paise),
                    "value_date": b.value_date.isoformat(),
                    "narration": b.description,
                    "utr_reference": b.utr_reference or "(none)",
                })
            out.append(row)
        bundle["near_miss_candidates"] = out

    if result.record_id in banks and not s:
        b = banks[result.record_id]
        bundle["orphan_bank_credit"] = {
            "bank_row_id": b.bank_row_id,
            "credit": paise_to_rupees(b.credit_paise),
            "value_date": b.value_date.isoformat(),
            "narration": b.description,
            "utr_reference": b.utr_reference or "(none)",
            "utr_recovered_via": b.utr_provenance,
        }

    return bundle


def select_cases(report: ReconciliationReport) -> list[MatchResult]:
    """Only exceptions and low-confidence matches. Clean matches never go out.

    This is the cost and latency argument, and also the correctness one: sending
    a confidently matched row to a model creates a chance of contradicting a
    decision that was already right.
    """
    cfg = report.config
    cases = [r for r in report.results if r.status == "unresolved"]
    cases += [r for r in report.results
              if r.status != "unresolved" and r.confidence < cfg.low_confidence_threshold]
    seen, unique = set(), []
    for c in cases:
        if c.record_id not in seen:
            seen.add(c.record_id)
            unique.append(c)
    unique.sort(key=lambda r: r.record_id)
    return unique


# --------------------------------------------------------------------------
# The reasoner
# --------------------------------------------------------------------------

class ExceptionReasoner:
    def __init__(self, cfg: MatchConfig = DEFAULT_CONFIG, *, cache: TraceCache | None = None,
                 use_llm: bool = True, deep: bool = True):
        self.cfg = cfg
        self.deep = deep
        self.cache = cache if cache is not None else TraceCache()
        self.use_llm = use_llm
        self._client = None
        self._client_error: str | None = None

    # -- client --------------------------------------------------------

    def _get_client(self):
        """Resolve a model provider lazily, and tolerate its absence.

        A missing key or missing SDK is an ordinary condition here, not an
        error: the cached-trace path is a supported way to run this project.
        Which vendor answers is decided by `agent.llm` from the environment.
        """
        if self._client is not None or self._client_error is not None:
            return self._client
        if not self.use_llm:
            self._client_error = "llm disabled for this run"
            return None
        try:
            self._client = get_provider()
        except LLMUnavailable as exc:
            self._client_error = str(exc)
            return None
        return self._client

    # -- main entry ----------------------------------------------------

    def investigate(self, report: ReconciliationReport, batch: LoadedBatch,
                    *, trail=None) -> ReasoningOutcome:
        import time
        started = time.perf_counter()

        cases = select_cases(report)
        stats = ReasoningStats(total_records=len(report.results), sent_to_llm=len(cases))
        outcome = ReasoningOutcome(stats=stats)

        bundles = {c.record_id: build_evidence_bundle(c, batch, self.cfg) for c in cases}
        # The prompt is part of the key: an answer is a function of the evidence
        # and the instructions together, so editing one must invalidate the other.
        instructions = system_prompt(self.cfg) + (
            investigation_system_prompt(self.cfg) if self.deep else '')
        keys = {rid: evidence_key(b, instructions) for rid, b in bundles.items()}

        pending: list[str] = []
        for rid in bundles:
            hit = self.cache.get(keys[rid])
            if hit is not None and not self.is_placeholder(hit):
                inv = dict(hit)
                inv["source"] = "cached_trace"
                outcome.investigations[rid] = inv
                stats.served_from_cache += 1
                # A replayed investigation is still an investigation. Counting it
                # keeps the summary honest on a keyless run, where every trace
                # comes from cache and the live counters are all zero.
                trace = inv.get("investigation_trace") or []
                if trace:
                    stats.deep_investigations += 1
                    stats.deep_turns += len(trace)
            else:
                # A cached placeholder is treated as a miss, so a cache poisoned
                # by an earlier failed run heals itself on the next good one.
                pending.append(rid)

        if pending:
            client = self._get_client()
            if client is None:
                stats.unavailable = True
                for rid in pending:
                    outcome.investigations[rid] = self._placeholder(rid, self._client_error)
            else:
                by_id = {r.record_id: r for r in cases}
                # Effort is matched to difficulty, the same way the matching
                # engine spends its effort. A case where the engine had a
                # candidate and declined earns a multi-turn investigation with
                # tools; a credit that resembles nothing earns a sentence.
                deep = [r for r in pending
                        if self.deep and needs_deep_investigation(by_id[r])]
                shallow = [r for r in pending if r not in set(deep)]

                if deep:
                    # Investigations are independent of each other, so they run
                    # concurrently. Sequentially this is ~20s x 60 cases; the
                    # pool brings a full run down to a few minutes. Results are
                    # merged on this thread, so the cache and stats stay
                    # single-writer.
                    investigator = BatchInvestigator(batch, report)
                    with ThreadPoolExecutor(max_workers=DEEP_WORKERS) as pool:
                        futures = {
                            pool.submit(investigate_case, client, bundles[rid],
                                        investigator, self.cfg): rid
                            for rid in deep
                        }
                        for future in as_completed(futures):
                            rid = futures[future]
                            try:
                                conclusion, steps, calls = future.result()
                            except Exception as exc:          # noqa: BLE001
                                conclusion, steps, calls = None, [], 0
                                stats.api_errors += 1
                            inv = self._finalise_investigation(
                                client, bundles[rid], conclusion, steps, calls, stats)
                            outcome.investigations[rid] = inv
                            if not self.is_placeholder(inv):
                                self.cache.put(keys[rid], {k: v for k, v in inv.items()
                                                           if k != "source"})

                for i in range(0, len(shallow), BATCH_SIZE):
                    chunk = shallow[i:i + BATCH_SIZE]
                    got = self._call(client, [bundles[r] for r in chunk], stats)
                    for rid in chunk:
                        inv = got.get(rid)
                        if inv is None:
                            outcome.investigations[rid] = self._placeholder(
                                rid, "model returned no investigation for this case")
                            continue
                        inv["source"] = "live_llm"
                        inv["model"] = stats.provider
                        outcome.investigations[rid] = inv
                        if not self.is_placeholder(inv):
                            self.cache.put(keys[rid], {k: v for k, v in inv.items()
                                                       if k != "source"})
                self.cache.save()

        if trail is not None:
            for rid, inv in outcome.investigations.items():
                trail.attach_reasoning(rid, {**inv, "evidence_key": keys[rid]})

        stats.elapsed_seconds = time.perf_counter() - started
        return outcome

    def _finalise_investigation(self, provider, bundle, conclusion, steps, calls,
                                stats: ReasoningStats) -> dict[str, Any]:
        """Turn one completed investigation into a stored result.

        Called on the main thread so stats and the cache have a single writer.
        """
        stats.live_calls += calls
        stats.deep_investigations += 1
        stats.deep_turns += len(steps)
        if not stats.provider:
            stats.provider = f"{provider.name}/{provider.model}"

        trace = [st.to_dict() for st in steps]
        if conclusion is None:
            # Never converged. Escalating is the safe default, and the trace
            # still shows a reviewer everything the agent looked at.
            out = self._placeholder(
                bundle["case_id"], "investigation did not converge within its turn limit")
            out["investigation_trace"] = trace
            return out

        conclusion["source"] = "live_llm"
        conclusion["model"] = stats.provider
        conclusion["investigation_trace"] = trace
        conclusion["tools_used"] = [st.action for st in steps if st.action != "conclude"]
        return conclusion

    # -- api -----------------------------------------------------------

    def _call(self, provider, bundles: list[dict[str, Any]],
              stats: ReasoningStats) -> dict[str, dict[str, Any]]:
        """One batched, schema-constrained request through the active provider."""
        try:
            response = provider.complete_json(
                system=system_prompt(self.cfg),
                user=build_user_message(bundles),
                schema=INVESTIGATION_SCHEMA,
                max_tokens=MAX_TOKENS,
                schema_name="reconciliation_investigations",
            )
        except Exception as exc:                      # noqa: BLE001
            # Any API failure degrades this batch to placeholders; the run and
            # every deterministic metric still complete.
            stats.api_errors += 1
            return {b["case_id"]: self._placeholder(b["case_id"], f"API error: {exc}")
                    for b in bundles}

        stats.live_calls += 1
        stats.provider = f"{response.provider}/{response.model}"
        try:
            parsed = parse_first_object(response.text)
        except json.JSONDecodeError:
            stats.api_errors += 1
            return {}
        return {inv["case_id"]: inv for inv in parsed.get("investigations", [])
                if isinstance(inv, dict) and "case_id" in inv}

    @staticmethod
    def is_placeholder(inv: dict[str, Any] | None) -> bool:
        """Whether this is a stand-in rather than a real investigation.

        Placeholders must never be written to, or served from, the trace cache.
        They record the *absence* of an explanation - caching one would let a
        single transient API failure permanently poison that case, and the
        poisoned entry would then be served with `source: cached_trace`, which
        reads exactly like a real result.
        """
        return bool(inv) and inv.get("_unavailable") is True

    @staticmethod
    def _placeholder(record_id: str, why: str | None) -> dict[str, Any]:
        """What an exception looks like when no explanation could be produced.

        It deliberately still escalates. An unexplained exception is not a
        resolved one, and the safe default when reasoning is unavailable is to
        put it in front of a person.
        """
        return {
            "case_id": record_id,
            "hypothesis": ("No agent explanation available for this exception "
                           f"({why or 'reasoning layer unavailable'}). The engine's rule "
                           "trace above records exactly why it did not match."),
            "sufficient_evidence": False,
            "confidence": "low",
            "recommended_action": "escalate_to_human",
            "evidence_cited": [],
            "rupee_impact": "unknown",
            "source": "unavailable",
            # Sentinel. Keeps this out of the persistent cache - see is_placeholder.
            "_unavailable": True,
        }
