"""Tests for the agent evaluator.

Two jobs here, and the second is the important one.

The first is ordinary: the label rules do what they claim. The second is that the
headline number - unsafe auto-resolves - currently reads zero, and a metric that
has only ever read zero is indistinguishable from a metric that cannot fire. So
several tests below construct a recommendation that *should* be flagged and assert
that it is. Without them, "0 unsafe auto-resolves" is an untested claim dressed up
as a measurement, which is the exact failure DEVLOG entry 9 is about.

The third job, quieter: assert the report cannot read as a pass when it is not.
An always-escalate policy scores full marks on action accuracy, and the report has
to keep saying so.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from evaluation import agent_eval as ae

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def case(expected_resolution, *, record_id="pay_x", engine_status="unresolved",
         reason=None, bank=(), action=None, sufficient=None, cited=(),
         hypothesis="", dataset="dev"):
    """One synthetic case carrying one synthetic arm answer under 'deep'."""
    c = ae.Case(
        record_id=record_id, dataset=dataset, group_id="G0001",
        expected_resolution=expected_resolution, case_type="synthetic",
        engine_status=engine_status, exception_reason=reason,
        routed_deep=True, true_bank_rows=frozenset(bank),
        bundle={"case_id": record_id},
    )
    if action is not None:
        c.arms["deep"] = {
            "case_id": record_id, "recommended_action": action,
            "sufficient_evidence": sufficient, "evidence_cited": list(cited),
            "hypothesis": hypothesis,
        }
    c.arms["always_escalate"] = {
        "case_id": record_id, "recommended_action": "escalate_to_human",
        "sufficient_evidence": False, "evidence_cited": [],
    }
    return c


# --------------------------------------------------------------------------
# The headline must be able to fire
# --------------------------------------------------------------------------

@pytest.mark.parametrize("resolution", sorted(ae.UNSAFE_TO_CLEAR))
def test_clearing_an_unclearable_case_is_unsafe(resolution):
    """auto_resolve where no correct resolution exists must be caught.

    Ambiguous twins cannot be identified at any tolerance, unmatchable rows have
    no counterpart, and a pending settlement is money that has not arrived -
    clearing that one overstates cash in the position report, which is the error
    with a downstream consumer.
    """
    res = ae.score([case(resolution, action="auto_resolve")], "deep")
    assert res.unsafe_clears == 1, f"{resolution} + auto_resolve must be unsafe"
    assert res.unverified_clears == 0
    assert res.action_accuracy == 0.0


def test_clearing_a_partial_refund_is_unverified_not_unsafe():
    """A correct resolution exists; nothing in the sources can confirm it.

    Keeping the two tiers apart is deliberate. Folding this into the headline
    would blunt the metric that means 'money bound to nothing'.
    """
    res = ae.score([case("exception_partial_refund", action="auto_resolve")], "deep")
    assert res.unsafe_clears == 0
    assert res.unverified_clears == 1
    # It still disagrees with ground truth, so it still costs action accuracy.
    assert res.action_accuracy == 0.0


def test_an_unsafe_recommendation_fails_the_build(monkeypatch, capsys):
    """The exit code is the contract CI depends on."""
    monkeypatch.setattr(ae, "load_cases",
                        lambda *a, **k: ([case("exception_ambiguous",
                                               action="auto_resolve")], []))
    monkeypatch.setattr("sys.argv", ["agent_eval"])
    assert ae.main() == 1
    assert "unsafe auto-resolve" in capsys.readouterr().out


def test_a_clean_corpus_exits_zero(monkeypatch):
    monkeypatch.setattr(ae, "load_cases",
                        lambda *a, **k: ([case("exception_ambiguous",
                                               action="escalate_to_human")], []))
    monkeypatch.setattr("sys.argv", ["agent_eval"])
    assert ae.main() == 0


# --------------------------------------------------------------------------
# Anti-vacuity: the report must not read as a pass when it is not
# --------------------------------------------------------------------------

def test_escalating_everything_scores_full_action_accuracy():
    """The trivial policy must be visibly trivial.

    If a fixed rule that never looks at anything scores 100%, then 100% is not
    evidence the agent is doing work, and the report has to carry that.
    """
    cases = [case(r, record_id=f"pay_{i}", action="escalate_to_human")
             for i, r in enumerate(sorted(ae.ESCALATION_REQUIRED))]
    control = ae.score(cases, "always_escalate")
    assert control.action_accuracy == 1.0
    assert control.unsafe_clears == 0
    assert control.recovery_full == 0


def test_the_control_column_is_always_rendered():
    cases = [case("exception_ambiguous", action="escalate_to_human")]
    text = ae.format_report(ae.score(cases, "deep"),
                            ae.score(cases, "always_escalate"), cases, [])
    assert "always-esc" in text
    assert "flaw in the metric" in text


def test_duplicates_are_reported_as_unreachable_not_as_perfect():
    """The engine resolves duplicates at 0.99, so none ever reach the agent.

    A flag_duplicate precision of 100% over an empty set is exactly the kind of
    number this project exists not to print.
    """
    cases = [case("exception_ambiguous", action="escalate_to_human")]
    res = ae.score(cases, "deep")
    assert res.duplicate_cases_available == 0
    assert res.flag_duplicates == 0
    text = ae.format_report(res, ae.score(cases, "always_escalate"), cases, [])
    assert "untested" in text and "not perfect" in text


def test_an_empty_denominator_reports_none_not_zero():
    """A rate of 0% that means 'no cases' is a lie, so it is not produced."""
    res = ae.score([case("matched_split", action="escalate_to_human")], "deep")
    assert res.action_scored == 0
    assert res.action_accuracy is None
    assert ae._pct(None).strip() == "n/a"


# --------------------------------------------------------------------------
# Label rules
# --------------------------------------------------------------------------

def test_an_engine_miss_is_not_scored_for_action():
    """These are the engine's own false negatives, not exceptions.

    Escalating one is not an error - a human resolves it - so crediting the agent
    for escalating something that was matchable would inflate the metric.
    """
    for resolution in ("matched", "matched_split"):
        res = ae.score([case(resolution, action="escalate_to_human")], "deep")
        assert res.action_scored == 0, f"{resolution} must not be scored for action"


def test_recovery_requires_the_complete_leg_set():
    """Two legs of a three-leg group is a wrong group that strands a credit.

    The same rule evaluation/metrics.py holds the engine to.
    """
    legs = ("BNK_1", "BNK_2", "BNK_3")
    partial = ae.score([case("matched_split", bank=legs, action="escalate_to_human",
                             cited=("BNK_1", "BNK_2"))], "deep")
    assert (partial.recovery_full, partial.recovery_partial) == (0, 1)

    full = ae.score([case("matched_split", bank=legs, action="escalate_to_human",
                          cited=legs)], "deep")
    assert (full.recovery_full, full.recovery_partial) == (1, 0)


def test_recovery_counts_ids_named_in_prose_not_only_citations():
    """An agent that names the legs in its explanation has still found them."""
    res = ae.score([case("matched_split", bank=("BNK_1", "BNK_2"),
                         action="escalate_to_human",
                         hypothesis="the legs are BNK_1 and BNK_2")], "deep")
    assert res.recovery_full == 1


def test_sufficient_evidence_is_scored_only_where_ground_truth_determines_it():
    cases = [
        case("exception_ambiguous", record_id="pay_a", action="escalate_to_human",
             sufficient=False),
        # Unmatchable is deliberately unscored: for a settlement with no
        # counterpart, "nothing is here" is arguably a single identified answer,
        # and that ambiguity belongs to the schema rather than to the agent.
        case("exception_unmatchable", record_id="pay_b",
             action="escalate_to_human", sufficient=True),
    ]
    res = ae.score(cases, "deep")
    assert res.se_scored == 1
    assert res.se_accuracy == 1.0


def test_an_unmapped_case_is_reported_and_lands_in_no_denominator():
    cases = [case("exception_ambiguous", action="escalate_to_human")]
    text = ae.format_report(ae.score(cases, "deep"),
                            ae.score(cases, "always_escalate"), cases,
                            ["dev/pay_ghost"])
    assert "pay_ghost" in text
    assert "UNMAPPED" in text


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

def test_the_two_arms_occupy_different_cache_slots():
    """The comparison rests entirely on this being true.

    All 104 committed traces were written in a deep-enabled run and are keyed
    under the deep instructions. If a shallow key collided with one of them,
    generating the one-shot arm would silently destroy the arm it is meant to be
    compared against.
    """
    from agent.cache import TraceCache, evidence_key
    from agent.investigator import system_prompt as deep_sp
    from agent.prompts import system_prompt
    from core.config import DEFAULT_CONFIG

    bundle = {"case_id": "pay_x", "engine_finding": "ambiguous_candidates"}
    shallow = evidence_key(bundle, system_prompt(DEFAULT_CONFIG))
    deep = evidence_key(bundle, system_prompt(DEFAULT_CONFIG)
                        + deep_sp(DEFAULT_CONFIG))
    assert shallow != deep
    assert shallow not in TraceCache()


def test_the_deterministic_arm_needs_no_model_and_no_network(monkeypatch):
    """Arm A is the honest baseline only if it genuinely uses nothing."""
    def no_sockets(*a, **k):
        raise AssertionError("the deterministic arm must not open a socket")
    monkeypatch.setattr(socket, "socket", no_sockets)

    cases, _ = ae.load_cases(("dev",))
    res = ae.score(cases, "deterministic")
    assert res.answered == len(cases)
    assert res.model == ""


def test_a_missing_arm_is_reported_and_does_not_fail_the_run():
    cases = [case("exception_ambiguous", action="escalate_to_human")]
    res = ae.score(cases, "shallow")
    assert res.answered == 0
    assert res.action_accuracy is None
    assert "not gen." in ae.format_comparison(cases)


def test_a_mixed_model_arm_is_flagged_as_confounded():
    """Two providers in one arm would score as an arm difference otherwise."""
    a = case("exception_ambiguous", record_id="pay_a", action="escalate_to_human")
    b = case("exception_ambiguous", record_id="pay_b", action="escalate_to_human")
    a.arms["deep"]["model"] = "openai/gpt-5.6-terra"
    b.arms["deep"]["model"] = "anthropic/claude-opus-5"
    assert ae.score([a, b], "deep").model == "MIXED"


# --------------------------------------------------------------------------
# Against the real committed corpus
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    return ae.load_cases()


def test_every_agent_case_maps_to_a_ground_truth_group(real):
    """The denominator guard. If this ever fails, every rate is over a subset."""
    cases, unmapped = real
    assert not unmapped, f"unmapped cases: {unmapped}"
    assert len(cases) == 104


def test_record_ids_are_unique_within_a_dataset_but_not_across(real):
    """Bank row ids restart per dataset, and that once broke the grounding check.

    BNK_00002 exists in dev and in holdout with different rows behind it. An
    index keyed on record id alone let the holdout bundle overwrite the dev one,
    so those traces were checked against evidence they were never shown - and it
    still reported zero, which is why nobody noticed. verify_grounding now keys
    by the cache key instead. This test pins the collision that made that
    necessary, so the shortcut cannot come back.
    """
    cases, _ = real
    for dataset in ("dev", "holdout"):
        ids = [c.record_id for c in cases if c.dataset == dataset]
        assert len(set(ids)) == len(ids), f"{dataset} has a duplicate record_id"

    shared = {c.record_id for c in cases if c.dataset == "dev"} &              {c.record_id for c in cases if c.dataset == "holdout"}
    assert shared, "expected some bank ids to collide across datasets"


def test_the_committed_corpus_has_no_unsafe_auto_resolves(real):
    """The headline claim, over the real data rather than a fixture.

    Paired with the parametrised test above, which proves the check fires.
    """
    cases, _ = real
    res = ae.score(cases, "deep")
    assert res.unsafe_clears == 0
    assert res.ids_ungrounded == 0


def test_grounding_agrees_with_verify_grounding(real):
    """Two scripts asserting the same invariant must not disagree about it."""
    from agent.cache import TraceCache
    from evaluation.grounding import ungrounded_in
    from verify_grounding import build_case_index

    cases, _ = real
    index = build_case_index(["dev", "holdout"])
    standalone = sum(len(ungrounded_in(index[key], trace))
                     for key, trace in TraceCache()._data.items())
    assert ae.score(cases, "deep").ids_ungrounded == standalone


def test_report_is_ascii(real):
    """A cp1252 console must not be able to crash the report - DEVLOG entry 1."""
    cases, unmapped = real
    text = ae.format_report(ae.score(cases, "deep"),
                            ae.score(cases, "always_escalate"), cases, unmapped)
    text.encode("ascii")
    ae.format_comparison(cases).encode("ascii")


def test_the_evaluator_writes_nothing(real):
    """It reads ground truth and the trace cache. It must not touch either."""
    traces = ROOT / "agent" / "cache" / "traces.json"
    before = traces.stat().st_mtime_ns, traces.stat().st_size
    cases, unmapped = real
    ae.format_report(ae.score(cases, "deep"), ae.score(cases, "always_escalate"),
                     cases, unmapped)
    assert (traces.stat().st_mtime_ns, traces.stat().st_size) == before


def test_json_output_is_machine_readable(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["agent_eval", "--json", "--dataset", "dev"])
    ae.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["unsafe_auto_resolves"] == 0
    assert payload["cases"] == 31
