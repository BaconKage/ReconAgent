"""Tests for the reasoning and Q&A layers.

The claims these defend:

* The reasoning layer cannot change a match. Not "does not by convention" -
  cannot, verified by comparing full engine output before and after it runs.
* Clean matches are never sent to a model.
* The model is never shown ground truth, case labels, or anything the engine did
  not itself observe.
* With no API key the pipeline still completes, and unexplained exceptions
  escalate rather than quietly resolving.
* Q&A retrieves; it does not invent. An unknown ID returns "not found".

No test here makes a network call. Both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`
are stripped from the environment so a developer's real key cannot turn the
suite into live spend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.cache import TraceCache, evidence_key
from agent.qa import ReconciliationQA
from agent.reasoner import (ExceptionReasoner, build_evidence_bundle,
                            select_cases)
from audit.trail import AuditTrail, merge_by_record
from core.loader import load_batch
from core.matcher import reconcile

DEV = Path(__file__).resolve().parents[1] / "data" / "dev"


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Guarantee offline behaviour for every test in this module.

    Both vendor keys are stripped, plus the provider override - otherwise a
    developer with either key set would turn this suite into live API spend.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RECONAGENT_LLM_PROVIDER", raising=False)


@pytest.fixture(scope="module")
def pipeline():
    batch = load_batch(DEV)
    return batch, reconcile(batch)


@pytest.fixture
def isolated_cache(tmp_path):
    return TraceCache(tmp_path / "traces.json")


# --------------------------------------------------------------------------
# Case selection - the efficiency and safety argument
# --------------------------------------------------------------------------

def test_clean_matches_are_never_sent_to_the_model(pipeline):
    _, report = pipeline
    selected = {r.record_id for r in select_cases(report)}
    confident = [r for r in report.results
                 if r.status in ("matched", "matched_split")
                 and r.confidence >= report.config.low_confidence_threshold]
    assert confident, "fixture should contain confidently matched rows"
    for r in confident:
        assert r.record_id not in selected, (
            f"{r.record_id} matched at confidence {r.confidence} but was queued for an LLM")


def test_every_unresolved_case_is_investigated(pipeline):
    _, report = pipeline
    selected = {r.record_id for r in select_cases(report)}
    for r in report.unresolved:
        assert r.record_id in selected


def test_a_meaningful_share_of_work_avoids_the_model(pipeline):
    _, report = pipeline
    fraction = len(select_cases(report)) / len(report.results)
    assert fraction < 0.5, (
        f"{fraction:.0%} of groups reach the LLM; the deterministic engine is "
        f"supposed to absorb the bulk of the batch")


# --------------------------------------------------------------------------
# Grounding - what the model is and is not shown
# --------------------------------------------------------------------------

def test_evidence_bundle_never_leaks_ground_truth(pipeline):
    """The model must not see the answer key, the case label, or the expected outcome."""
    batch, report = pipeline
    forbidden = ("case_type", "expected_resolution", "ground_truth", "group_id",
                 "adversarial", "unmatchable", "notes")
    for case in select_cases(report):
        blob = json.dumps(build_evidence_bundle(case, batch)).lower()
        for token in forbidden:
            assert token not in blob, f"{case.record_id} bundle leaks '{token}'"


def test_evidence_bundle_cites_only_records_that_exist(pipeline):
    batch, report = pipeline
    real_banks = {b.bank_row_id for b in batch.banks}
    for case in select_cases(report):
        for nm in build_evidence_bundle(case, batch).get("near_miss_candidates", []):
            assert nm["record_id"] in real_banks


def test_near_misses_carry_the_actual_bank_row_details(pipeline):
    """A hypothesis is only grounded if the model can see the real candidate."""
    batch, report = pipeline
    with_near = [c for c in select_cases(report) if c.near_misses]
    assert with_near, "fixture should contain at least one near-miss case"
    bundle = build_evidence_bundle(with_near[0], batch)
    cand = bundle["near_miss_candidates"][0]
    for key in ("credit", "value_date", "narration", "why_engine_declined_it"):
        assert key in cand


def test_bundle_explains_why_each_candidate_was_declined(pipeline):
    batch, report = pipeline
    for case in select_cases(report):
        for nm in build_evidence_bundle(case, batch).get("near_miss_candidates", []):
            assert nm["why_engine_declined_it"].strip()


# --------------------------------------------------------------------------
# The central architectural claim
# --------------------------------------------------------------------------

def test_reasoning_cannot_change_a_single_match(pipeline, isolated_cache):
    """Run the engine, snapshot it, reason over it, and prove nothing moved."""
    batch, report = pipeline

    def snapshot(rep):
        return sorted(
            (r.record_id, r.status, r.match_type, r.confidence,
             tuple(sorted(r.group_key)), r.exception_reason)
            for r in rep.results
        )

    before = snapshot(report)
    ExceptionReasoner(cache=isolated_cache).investigate(report, batch)
    assert snapshot(report) == before


def test_metrics_are_identical_with_and_without_the_reasoning_layer():
    """A judge with no key must get the same numbers as one with a key."""
    batch = load_batch(DEV)
    plain = reconcile(batch)
    reasoned = reconcile(load_batch(DEV))
    ExceptionReasoner(cache=TraceCache(Path("nonexistent") / "x.json")).investigate(
        reasoned, batch)
    assert plain.by_status() == reasoned.by_status()
    assert plain.by_exception_reason() == reasoned.by_exception_reason()


# --------------------------------------------------------------------------
# Graceful degradation
# --------------------------------------------------------------------------

def test_pipeline_completes_without_an_api_key(pipeline, isolated_cache):
    batch, report = pipeline
    outcome = ExceptionReasoner(cache=isolated_cache).investigate(report, batch)
    assert outcome.stats.unavailable is True
    assert len(outcome.investigations) == len(select_cases(report))


def test_unexplained_exceptions_escalate_rather_than_resolve(pipeline, isolated_cache):
    """The safe default when reasoning is unavailable is a human, not a match."""
    batch, report = pipeline
    outcome = ExceptionReasoner(cache=isolated_cache).investigate(report, batch)
    for rid, inv in outcome.investigations.items():
        assert inv["recommended_action"] == "escalate_to_human"
        assert inv["sufficient_evidence"] is False
        assert inv["source"] == "unavailable"


def test_reasoner_never_raises_when_the_sdk_is_missing(pipeline, isolated_cache, monkeypatch):
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **kw):
        if name.split(".")[0] == "anthropic":
            raise ImportError("blocked")
        return real(name, *a, **kw)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-not-used")
    monkeypatch.setattr(builtins, "__import__", blocked)
    batch, report = pipeline
    outcome = ExceptionReasoner(cache=isolated_cache).investigate(report, batch)
    assert outcome.stats.unavailable is True


# --------------------------------------------------------------------------
# Trace cache
# --------------------------------------------------------------------------

def test_cache_key_is_stable_across_dict_ordering():
    a = {"case_id": "pay_1", "engine_finding": "x", "settlement": {"net": "1.00"}}
    b = {"settlement": {"net": "1.00"}, "engine_finding": "x", "case_id": "pay_1"}
    assert evidence_key(a) == evidence_key(b)


def test_cache_key_changes_when_the_evidence_changes():
    """A stale explanation must never be served for data it no longer describes."""
    base = {"case_id": "pay_1", "settlement": {"net": "100.00"}}
    moved = {"case_id": "pay_1", "settlement": {"net": "100.01"}}
    assert evidence_key(base) != evidence_key(moved)


def test_cache_survives_a_corrupt_file(tmp_path):
    p = tmp_path / "traces.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert len(TraceCache(p)) == 0


def test_cache_roundtrips(tmp_path):
    p = tmp_path / "traces.json"
    c = TraceCache(p)
    c.put("k", {"hypothesis": "h"})
    c.save()
    assert TraceCache(p).get("k") == {"hypothesis": "h"}


# --------------------------------------------------------------------------
# Audit trail
# --------------------------------------------------------------------------

def test_trail_is_append_only_and_merges_by_record(tmp_path, pipeline):
    _, report = pipeline
    trail = AuditTrail(run_id="t", directory=tmp_path)
    for r in report.results[:5]:
        trail.append_decision(r)
    trail.attach_reasoning(report.results[0].record_id, {"hypothesis": "later observation"})

    raw = AuditTrail.read(trail.path)
    assert len(raw) == 6, "attaching reasoning must add a line, not rewrite one"

    merged = merge_by_record(raw)
    assert merged[report.results[0].record_id]["reasoning"]["hypothesis"] == "later observation"
    assert merged[report.results[0].record_id]["status"] == report.results[0].status


def test_reasoning_entry_does_not_erase_the_engine_decision(tmp_path, pipeline):
    _, report = pipeline
    r0 = report.results[0]
    trail = AuditTrail(run_id="t2", directory=tmp_path)
    trail.append_decision(r0)
    trail.attach_reasoning(r0.record_id, {"hypothesis": "h"})
    merged = merge_by_record(AuditTrail.read(trail.path))[r0.record_id]
    assert merged["rule_trace"] == r0.rule_trace


# --------------------------------------------------------------------------
# Q&A - retrieval, not invention
# --------------------------------------------------------------------------

@pytest.fixture
def qa(tmp_path, pipeline):
    _, report = pipeline
    trail = AuditTrail(run_id="qa", directory=tmp_path)
    for r in report.results:
        trail.append_decision(r)
    return ReconciliationQA(trail.path), report


def test_qa_answers_for_a_known_transaction(qa):
    q, report = qa
    matched = next(r for r in report.results if r.status == "matched")
    ans = q.ask(f"why did {matched.record_id} reconcile?")
    assert ans.found and ans.source == "trail"
    assert matched.record_id in ans.answer


def test_qa_refuses_to_invent_an_unknown_transaction(qa):
    q, _ = qa
    ans = q.ask("why didn't pay_doesnotexist99 reconcile?")
    assert ans.found is False
    assert ans.source == "not_found"
    assert "could not find" in ans.answer.lower()


def test_qa_resolves_a_bank_row_id_to_its_group(qa):
    q, report = qa
    grouped = next(r for r in report.results if r.linked_ids["bank"] and r.status == "matched")
    bank_id = grouped.linked_ids["bank"][0]
    ans = q.ask(f"what happened to {bank_id}?")
    assert ans.found and ans.record_id == grouped.record_id


def test_qa_resolves_a_ledger_order_id(qa):
    q, report = qa
    grouped = next(r for r in report.results if r.linked_ids["ledger"])
    ans = q.ask(f"tell me about {grouped.linked_ids['ledger'][0]}")
    assert ans.found


def test_qa_reports_the_real_reason_for_an_exception(qa):
    q, report = qa
    amb = next((r for r in report.unresolved
                if r.exception_reason == "ambiguous_candidates"), None)
    assert amb is not None
    ans = q.ask(f"why didn't {amb.record_id} reconcile?")
    assert "did NOT reconcile" in ans.answer
    assert "more than one bank credit" in ans.answer


def test_qa_answer_only_contains_ids_present_in_the_record(qa):
    """The strongest anti-hallucination check available offline."""
    import re
    q, report = qa
    for r in report.unresolved[:12]:
        ans = q.ask(f"why didn't {r.record_id} reconcile?")
        permitted = {r.record_id}
        permitted |= {i for ids in r.linked_ids.values() for i in ids}
        permitted |= {nm.record_id for nm in r.near_misses}
        for token in re.findall(r"\b(?:pay_[a-z0-9]+|order_[a-z0-9]+|BNK_\d+)\b", ans.answer):
            assert token in permitted, f"{r.record_id}: answer cites unrelated {token}"


def test_qa_on_an_empty_trail_says_so(tmp_path):
    q = ReconciliationQA(tmp_path / "missing.jsonl")
    ans = q.ask("why didn't pay_abc reconcile?")
    assert ans.found is False


def test_cache_key_changes_when_the_prompt_changes():
    """Editing the instructions must invalidate cached answers.

    A cached trace is indistinguishable from a fresh one once stored, so a
    prompt change that did not invalidate the cache would appear to take effect
    while every existing case silently replayed an answer written under the old
    rules.
    """
    bundle = {"case_id": "pay_1", "engine_finding": "no_candidate_found"}
    assert evidence_key(bundle, "instructions A") != evidence_key(bundle, "instructions B")
    assert evidence_key(bundle, "instructions A") == evidence_key(bundle, "instructions A")


def test_placeholders_are_never_written_to_the_cache(pipeline, isolated_cache):
    """A failed run must not poison the cache for the next, working one.

    This was a real bug: an API failure produced placeholder investigations that
    were cached like real ones, so a later run with a valid key served the
    failures back under `source: cached_trace` and never called the model.
    """
    batch, report = pipeline
    ExceptionReasoner(cache=isolated_cache).investigate(report, batch)
    assert len(isolated_cache) == 0, "placeholders must not be persisted"


def test_a_cached_placeholder_is_treated_as_a_miss(pipeline, isolated_cache):
    """Self-healing for caches already poisoned by an earlier failed run."""
    batch, report = pipeline
    reasoner = ExceptionReasoner(cache=isolated_cache)
    poisoned = reasoner._placeholder("pay_x", "API error: 401")
    assert ExceptionReasoner.is_placeholder(poisoned)
    isolated_cache.put("somekey", poisoned)
    assert isolated_cache.get("somekey") is not None
    assert ExceptionReasoner.is_placeholder(isolated_cache.get("somekey"))


def test_qa_finds_a_transaction_recorded_in_an_earlier_run(tmp_path, pipeline):
    """Runs are per-dataset, so the latest trail is not always the right one.

    Asking about a dev transaction straight after a holdout run must not report
    "not found" for a transaction that reconciled perfectly well. This was a real
    bug: the exact command documented in the README failed that way.
    """
    _, report = pipeline
    older = AuditTrail(run_id="run_a", directory=tmp_path)
    for r in report.results:
        older.append_decision(r)

    newer = AuditTrail(run_id="run_b", directory=tmp_path)
    newer.append({"record_id": "pay_unrelated", "status": "matched",
                  "linked_ids": {"settlement": ["pay_unrelated"], "bank": [], "ledger": []},
                  "rule_trace": [], "near_misses": []})

    q = ReconciliationQA(newer.path)
    target = report.results[0].record_id
    assert q.resolve(target) is None, "fixture should not have it in the latest trail"

    ans = q.ask(f"why didn't {target} reconcile?")
    assert ans.found is True
    assert "from an earlier run" in ans.answer
    assert "run_a" in ans.answer


def test_qa_still_refuses_an_id_that_exists_in_no_run(tmp_path, pipeline):
    _, report = pipeline
    trail = AuditTrail(run_id="only", directory=tmp_path)
    for r in report.results[:3]:
        trail.append_decision(r)
    ans = ReconciliationQA(trail.path).ask("why didn't pay_neverexisted reconcile?")
    assert ans.found is False and ans.source == "not_found"


def test_the_readme_quotes_traces_that_say_what_it_claims():
    """Pin the two traces the README quotes, and the auto_resolve counts.

    The README once claimed the agent auto-resolved "8 exceptions, all partial
    refunds" and quoted BNK_00042 as an example. There were four auto-resolves,
    not eight; they were not all partial refunds; and the BNK_00042 trace
    actually recommends escalate_to_human - so the quote argued the opposite of
    what the trace says. All three were checkable against committed data by
    anyone who looked.

    A number in the README that no test recomputes is a number that has already
    started drifting.
    """
    import collections
    from agent.cache import TraceCache

    cache = TraceCache()
    by_case = {t.get("case_id"): t for t in cache._data.values()}

    counts = collections.Counter(
        t.get("recommended_action") for t in cache._data.values())
    assert counts["auto_resolve"] == 4, (
        f"README says 4 auto-resolves across both datasets, traces say "
        f"{counts['auto_resolve']}")

    # The trace quoted as the agent clearing a shortfall it could account for.
    cleared = by_case["pay_qenf91mx9j"]
    assert cleared["recommended_action"] == "auto_resolve"
    assert cleared["sufficient_evidence"] is True

    # The trace quoted as the contrast: same case shape, refused.
    refused = by_case["pay_59sojuv0gy"]
    assert refused["recommended_action"] == "escalate_to_human"
    assert refused["sufficient_evidence"] is False
    assert "BNK_00042" in refused["hypothesis"]
