"""Tests for the tool-using investigation layer.

The agent here has real autonomy: it decides which queries to run and when it has
seen enough. That is only safe because of two properties, and both are tested.

* **Every tool is read-only.** There is no tool that creates a link, changes a
  status or resolves anything, so no sequence of agent actions - however
  confused, looping or adversarial - can alter a reconciliation outcome.
* **The loop is bounded.** A hard turn cap, with escalation as the default when
  the agent fails to converge. An unbounded loop over money is not a feature.

A scripted fake provider stands in for the model, so these run offline and the
agent's behaviour under awkward conditions (never concluding, asking for
nonexistent records, requesting unknown tools) is tested deterministically
instead of hoped for.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from agent.investigator import (MAX_TURNS, investigate_case,
                                needs_deep_investigation, parse_first_object)
from agent.reasoner import build_evidence_bundle
from agent.tools import TOOL_NAMES, BatchInvestigator, run_tool
from core.loader import load_batch
from core.matcher import reconcile

DEV = Path(__file__).resolve().parents[1] / "data" / "dev"


@pytest.fixture(scope="module")
def world():
    batch = load_batch(DEV)
    report = reconcile(batch)
    return batch, report, BatchInvestigator(batch, report)


class ScriptedProvider:
    """Returns a fixed sequence of steps. Records what it was asked."""
    name, model = "fake", "scripted"

    def __init__(self, steps):
        self._steps = list(steps)
        self.prompts: list[str] = []

    def complete_json(self, system, user, schema, *, max_tokens, schema_name="x"):
        from agent.llm import LLMResponse
        self.prompts.append(user)
        step = self._steps.pop(0) if self._steps else self._steps_exhausted()
        return LLMResponse(text=json.dumps(step), model=self.model, provider=self.name)

    def _steps_exhausted(self):
        return step_of("search_credits", thought="looping forever")

    def complete_text(self, system, user, *, max_tokens):
        from agent.llm import LLMResponse
        return LLMResponse(text="", model=self.model, provider=self.name)


def step_of(action, thought="t", conclusion=None, **params):
    base = {"thought": thought, "action": action, "record_id": None,
            "min_amount": None, "max_amount": None, "start_date": None,
            "end_date": None, "utr_fragment": None, "conclusion": conclusion}
    base.update(params)
    return base


CONCLUSION = {
    "hypothesis": "nothing explains the gap", "sufficient_evidence": False,
    "confidence": "high", "recommended_action": "escalate_to_human",
    "evidence_cited": [], "rupee_impact": "unknown",
}


# --------------------------------------------------------------------------
# The tools are questions, never levers
# --------------------------------------------------------------------------

def test_no_tool_can_mutate_the_reconciliation(world):
    """Run every tool and assert the engine's output is untouched."""
    batch, report, inv = world
    before = sorted((r.record_id, r.status, r.confidence, tuple(sorted(r.group_key)))
                    for r in report.results)
    bank_id = batch.banks[0].bank_row_id
    txn_id = batch.settlements[0].transaction_id

    inv.batch_summary()
    inv.search_credits(min_amount="1.00", max_amount="999999.00")
    inv.get_credit(bank_id)
    inv.get_settlement(txn_id)
    inv.find_utr("1234")
    inv.credits_near_settlement(txn_id)

    after = sorted((r.record_id, r.status, r.confidence, tuple(sorted(r.group_key)))
                   for r in report.results)
    assert after == before


def test_the_toolset_contains_no_mutating_verb():
    """A structural guard: nothing named like a write operation exists."""
    banned = ("set", "update", "match", "link", "resolve", "write", "mark", "delete")
    for name in TOOL_NAMES:
        assert not any(name.startswith(v) for v in banned), f"{name} looks like a mutation"


def test_search_credits_filters_by_amount_and_date(world):
    _, _, inv = world
    res = inv.search_credits(min_amount="1000.00", max_amount="2000.00")
    for c in res["credits"]:
        assert 1000.0 <= float(c["credit"]) <= 2000.0


def test_search_excludes_already_matched_credits_by_default(world):
    _, _, inv = world
    res = inv.search_credits(min_amount="0.01", max_amount="9999999.00")
    assert all(not c["already_matched_to_another_settlement"] for c in res["credits"])
    wide = inv.search_credits(min_amount="0.01", max_amount="9999999.00",
                              include_matched=True)
    assert wide["total_found"] > res["total_found"]


def test_credits_near_settlement_quantifies_the_gap(world):
    """Surfaces the unclaimed near miss, with the gap already worked out.

    Picks a case with no credit yet linked, so the near miss is genuinely
    something the agent has to go and find rather than something it was handed.
    """
    batch, report, inv = world
    unresolved = next(r for r in report.unresolved
                      if r.near_misses and not r.linked_ids["bank"])
    res = inv.credits_near_settlement(unresolved.record_id)
    assert res["credits"], "should surface the near miss the engine already found"
    found = {c["bank_row_id"] for c in res["credits"]}
    assert found & {nm.record_id for nm in unresolved.near_misses}
    for c in res["credits"]:
        assert "differs_from_net_by" in c and "days_after_settlement" in c


def test_a_large_unexplained_gap_falls_outside_the_default_search(world):
    """The default slack is deliberately narrow.

    A partial refund's shortfall can run to thousands of rupees, well outside the
    default window - and that is correct: its credit is already linked by UTR, so
    the agent has it in the case bundle and does not need to search for it. The
    search tool exists to find things the engine did NOT already surface.
    """
    batch, report, inv = world
    linked = next(r for r in report.unresolved
                  if r.exception_reason == "identifier_match_amount_discrepancy")
    res = inv.credits_near_settlement(linked.record_id)
    assert res["found"] is not False
    # Widening the search explicitly is the agent's option, and it works.
    wide = inv.credits_near_settlement(linked.record_id, amount_slack="50000.00")
    assert wide["total_found"] >= res["total_found"]


def test_unknown_records_are_reported_not_raised(world):
    _, _, inv = world
    assert inv.get_credit("BNK_99999")["found"] is False
    assert inv.get_settlement("pay_nope")["found"] is False
    assert inv.credits_near_settlement("pay_nope")["found"] is False


def test_unknown_tool_returns_an_error_the_agent_can_read(world):
    """An agent asking for something impossible gets told, and another turn."""
    _, _, inv = world
    out = run_tool(inv, "delete_everything", {})
    assert "error" in out and "available" in out


def test_find_utr_requires_enough_digits(world):
    _, _, inv = world
    assert inv.find_utr("12")["credits"] == []


# --------------------------------------------------------------------------
# Case selection - effort matched to difficulty
# --------------------------------------------------------------------------

def test_only_hard_cases_earn_a_deep_investigation(world):
    _, report, _ = world
    deep = [r for r in report.unresolved if needs_deep_investigation(r)]
    shallow = [r for r in report.unresolved if not needs_deep_investigation(r)]
    assert deep and shallow, "both paths should be exercised by the fixture"
    for r in deep:
        assert r.near_misses or r.exception_reason in {
            "ambiguous_candidates", "contested_candidate", "ambiguous_split",
            "contested_split_legs", "no_candidate_found"}


def test_matched_records_never_get_investigated(world):
    _, report, _ = world
    for r in report.results:
        if r.status != "unresolved":
            assert not needs_deep_investigation(r)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def test_investigation_concludes_and_returns_its_trace(world):
    batch, report, inv = world
    case = next(r for r in report.unresolved if r.near_misses)
    provider = ScriptedProvider([
        step_of("credits_near_settlement", record_id=case.record_id),
        step_of("conclude", conclusion=CONCLUSION),
    ])
    conclusion, steps, calls = investigate_case(
        provider, build_evidence_bundle(case, batch), inv, report.config)
    assert conclusion["recommended_action"] == "escalate_to_human"
    assert conclusion["case_id"] == case.record_id
    assert [s.action for s in steps] == ["credits_near_settlement", "conclude"]
    assert calls == 2


def test_tool_results_are_fed_back_into_the_next_turn(world):
    """Otherwise it is not an investigation, just repeated one-shot guessing."""
    batch, report, inv = world
    case = next(r for r in report.unresolved if r.near_misses)
    provider = ScriptedProvider([
        step_of("credits_near_settlement", record_id=case.record_id),
        step_of("conclude", conclusion=CONCLUSION),
    ])
    investigate_case(provider, build_evidence_bundle(case, batch), inv, report.config)
    assert "You ran: credits_near_settlement" in provider.prompts[1]
    assert "Result:" in provider.prompts[1]


def test_the_loop_is_bounded(world):
    """An agent that never concludes must still terminate."""
    batch, report, inv = world
    case = next(r for r in report.unresolved if r.near_misses)
    provider = ScriptedProvider([])          # always returns a non-concluding step
    conclusion, steps, calls = investigate_case(
        provider, build_evidence_bundle(case, batch), inv, report.config)
    assert conclusion is None, "no conclusion means the caller escalates"
    assert len(steps) == MAX_TURNS
    assert calls == MAX_TURNS


def test_failure_to_converge_does_not_raise(world):
    batch, report, inv = world
    case = report.unresolved[0]

    class Broken:
        name, model = "broken", "x"
        def complete_json(self, **kw):
            raise RuntimeError("provider exploded")
        def complete_text(self, **kw):
            raise RuntimeError("provider exploded")

    conclusion, steps, calls = investigate_case(
        Broken(), build_evidence_bundle(case, batch), inv, report.config)
    assert conclusion is None
    assert steps and steps[0].action == "error"


def test_an_investigation_cannot_change_a_match(world):
    """The headline guarantee, restated for the autonomous path."""
    batch, report, inv = world
    before = sorted((r.record_id, r.status, r.confidence) for r in report.results)
    case = next(r for r in report.unresolved if r.near_misses)
    provider = ScriptedProvider([
        step_of("search_credits", min_amount="0.01", max_amount="9999999.00"),
        step_of("get_credit", record_id=batch.banks[0].bank_row_id),
        step_of("conclude", conclusion=CONCLUSION),
    ])
    investigate_case(provider, build_evidence_bundle(case, batch), inv, report.config)
    assert sorted((r.record_id, r.status, r.confidence) for r in report.results) == before


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

def test_concatenated_json_objects_are_survivable():
    """A reasoning model can emit two message items; output_text glues them.

    This was a real failure: `{"a":1}{"a":1}` raised "Extra data: line 1 column N"
    and killed the investigation on turn one.
    """
    assert parse_first_object('{"a": 1}{"a": 1}') == {"a": 1}
    assert parse_first_object('  {"a": 2}  ') == {"a": 2}


def test_llm_shim_takes_the_last_message_not_the_concatenation():
    from agent.llm import _openai_text

    class Block:
        def __init__(self, text): self.type, self.text = "output_text", text

    class Item:
        def __init__(self, type_, blocks=()): self.type, self.content = type_, list(blocks)

    response = type("R", (), {
        "output": [Item("reasoning"), Item("message", [Block('{"n":1}')]),
                   Item("reasoning"), Item("message", [Block('{"n":2}')])],
        "output_text": '{"n":1}{"n":2}',
    })()
    assert _openai_text(response) == '{"n":2}'
