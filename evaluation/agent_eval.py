"""Score the reasoning layer's recommendations against ground truth.

The engine is measured hard - precision, recall, adversarial conflations, a sealed
holdout. The reasoning layer was not measured at all, and the README said so: "its
recommendations are not checked against ground truth". That admission was honest
and it was also the largest hole in the project, because an advisory layer nobody
has scored is indistinguishable from a decorative one.

No new benchmark is needed. Every exception the agent sees belongs to a
ground-truth group that already records what should happen to it, and every agent
case maps to exactly one such group - dev 31/31, holdout 73/73, nothing unmapped.

    python -m evaluation.agent_eval [--dataset dev holdout] [--compare] [--json]

What can be scored, and what deliberately cannot
------------------------------------------------
**Not every case is an exception.** 32 of the 104 carry an `expected_resolution`
of `matched` or `matched_split`: they are the engine's own false negatives, mostly
the holdout T+4 window gap the README already confesses. Escalating one is not an
error - a human does resolve it - and clearing one is not unsafe, because a
correct answer exists. Scoring them for action would credit the agent for
escalating things that were matchable. They are excluded from action accuracy and
scored on a different axis instead: **did the agent find what the engine missed?**

**Three classes make clearing the item unsafe**, because no correct resolution
exists: the ambiguous twins (identical net, identical date, no UTR - undecidable
at any tolerance, and the generator builds them that way on purpose), the
unmatchable rows (no counterpart anywhere in the batch), and the pending
settlements (the money has not arrived). That last one is in the set because
`cash/position.py` consumes reconciled status: clearing a settlement still in
transit does not leave a visibly wrong link, it leaves a clean-looking book that
overstates cash, and nobody goes looking.

**Partial refunds are a separate tier.** A correct resolution does exist - link it
and book the shortfall - so clearing one is not unsafe. But the ledger records
refund *status* and never refund *amount*, so nothing can corroborate the gap.
An `auto_resolve` there is counted as **unverified**, on its own line, and is
scored as disagreeing with ground truth without being folded into the headline.
Collapsing the two tiers would blunt the metric that matters.

**Duplicates never reach the agent at all.** The engine resolves them at 0.99
confidence, so `select_cases` never picks one: 11 duplicate groups across both
datasets, 0 cases routed. `flag_duplicate` is therefore untested, not perfect, and
is reported as a count over an empty set rather than as a rate.

Every rate here returns None on an empty denominator and prints `n/a`, because a
0% that means "no cases" is a lie. And an always-escalate control is printed
beside the agent in every table: on a corpus that is overwhelmingly escalations, a
fixed policy scores well, and a metric that cannot distinguish the two should not
be quoted alone.

This module reads ground truth and is therefore confined to `evaluation/`, the
only package permitted to. It changes nothing: no match, no status, no trace.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.cache import TraceCache, evidence_key
from agent.investigator import (needs_deep_investigation,
                                system_prompt as investigation_system_prompt)
from agent.prompts import system_prompt
from agent.reasoner import build_evidence_bundle, select_cases
from core.config import DEFAULT_CONFIG, MatchConfig
from core.loader import load_batch
from core.matcher import reconcile
from core.normalize import make_stdout_safe
from evaluation.grounding import ids_in, ungrounded_in

make_stdout_safe()

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("dev", "holdout")

#: Classes where the only defensible recommendation is to send it to a person,
#: and therefore the only classes scored for action accuracy.
ESCALATION_REQUIRED = frozenset({
    "exception_ambiguous", "exception_unmatchable",
    "exception_pending", "exception_partial_refund",
})

#: Clearing one of these binds money to a counterpart that cannot be identified,
#: does not exist, or has not arrived. This is the number that fails a build.
UNSAFE_TO_CLEAR = frozenset({
    "exception_ambiguous", "exception_unmatchable", "exception_pending",
})

#: A correct resolution exists but no source can corroborate the gap. Wrong, but
#: not the same kind of wrong - reported on its own line.
UNVERIFIED_TO_CLEAR = frozenset({"exception_partial_refund"})

#: The engine's own misses. Not scored for action; scored on recovery instead.
ENGINE_MISS = frozenset({"matched", "matched_split"})

#: Where ground truth fixes what `sufficient_evidence` must be. Ambiguous twins
#: and partial refunds are both definitively "the data cannot settle this".
#: Unmatchable and pending are left unscored: the schema asks whether the
#: evidence "identifies a single answer", and for a settlement with no
#: counterpart the single answer arguably IS "nothing is here". That ambiguity is
#: in the schema, not in the agent, so it is not scored against the agent.
DETERMINED_SUFFICIENT_EVIDENCE: dict[str, bool] = {
    "exception_ambiguous": False,
    "exception_partial_refund": False,
}

#: Arm A. What the engine alone can say, with no model in the loop.
DETERMINISTIC_ACTION: dict[str | None, str] = {
    "duplicate_settlement_row": "flag_duplicate",
}
DETERMINISTIC_DEFAULT = "escalate_to_human"

ARMS = ("deterministic", "always_escalate", "shallow", "deep")


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

@dataclass
class Case:
    record_id: str
    dataset: str
    group_id: str
    expected_resolution: str
    case_type: str
    engine_status: str
    exception_reason: str | None
    routed_deep: bool
    true_bank_rows: frozenset[str]
    bundle: dict[str, Any]
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def scored_for_action(self) -> bool:
        return self.expected_resolution in ESCALATION_REQUIRED

    @property
    def is_engine_miss(self) -> bool:
        """A matchable group the engine failed to claim - a recovery opportunity."""
        return (self.expected_resolution in ENGINE_MISS
                and self.engine_status == "unresolved")


def load_cases(datasets=DATASETS, cfg: MatchConfig = DEFAULT_CONFIG,
               cache: TraceCache | None = None) -> tuple[list[Case], list[str]]:
    """Rebuild every agent case and attach each arm's answer.

    Arms are found by recomputing the cache key rather than by guessing from a
    trace's shape, because shape is misleading here: `ExceptionReasoner.investigate`
    computes one `instructions` string for a whole run, so every committed trace -
    including the 42 that took the one-shot path - is keyed under the *deep*
    instructions. A genuinely shallow arm (`deep=False`) has different
    instructions, hence different keys, hence its own slots. Verified: 0 of the
    104 committed keys collide with a shallow key.
    """
    cache = cache or TraceCache()
    shallow_key_of = lambda b: evidence_key(b, system_prompt(cfg))
    deep_key_of = lambda b: evidence_key(
        b, system_prompt(cfg) + investigation_system_prompt(cfg))

    cases: list[Case] = []
    unmapped: list[str] = []

    for name in datasets:
        data_dir = ROOT / "data" / name
        with open(data_dir / "ground_truth.json", encoding="utf-8") as f:
            gt = json.load(f)
        member_to_group: dict[str, dict] = {}
        for g in gt["groups"]:
            for rid in (g["ledger_order_ids"] + g["settlement_txn_ids"]
                        + g["bank_row_ids"]):
                prior = member_to_group.get(rid)
                assert prior is None or prior["group_id"] == g["group_id"], (
                    f"{rid} belongs to two groups; the mapping is not a function")
                member_to_group[rid] = g

        batch = load_batch(data_dir)
        report = reconcile(batch, cfg)
        for result in select_cases(report):
            group = member_to_group.get(result.record_id)
            if group is None:
                # Reported, never dropped: a silent skip would shrink the
                # denominator to whatever happens to pass.
                unmapped.append(f"{name}/{result.record_id}")
                continue

            bundle = build_evidence_bundle(result, batch, cfg)
            case = Case(
                record_id=result.record_id, dataset=name,
                group_id=group["group_id"],
                expected_resolution=group["expected_resolution"],
                case_type=group["case_type"],
                engine_status=result.status,
                exception_reason=result.exception_reason,
                routed_deep=needs_deep_investigation(result),
                true_bank_rows=frozenset(group["bank_row_ids"]),
                bundle=bundle,
            )

            case.arms["deterministic"] = {
                "case_id": result.record_id,
                "recommended_action": DETERMINISTIC_ACTION.get(
                    result.exception_reason, DETERMINISTIC_DEFAULT),
                # The engine has no opinion on evidence sufficiency, and
                # inventing one here would flatter the baseline.
                "sufficient_evidence": None,
                "evidence_cited": sorted({nm.record_id for nm in result.near_misses}),
            }
            # The trivial policy, printed beside everything as a control.
            case.arms["always_escalate"] = {
                "case_id": result.record_id,
                "recommended_action": "escalate_to_human",
                "sufficient_evidence": False,
                "evidence_cited": [],
            }
            shallow = cache.get(shallow_key_of(bundle))
            if shallow:
                case.arms["shallow"] = shallow
            deep = cache.get(deep_key_of(bundle))
            if deep:
                case.arms["deep"] = deep

            cases.append(case)

    return cases, unmapped


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _rate(num: int, den: int) -> float | None:
    """None, not 0.0, on an empty denominator - a 0% meaning 'no cases' is a lie."""
    return num / den if den else None


@dataclass
class ArmScore:
    arm: str = ""
    model: str = ""
    cases_total: int = 0
    answered: int = 0

    action_scored: int = 0
    action_correct: int = 0
    unsafe_clears: int = 0
    unverified_clears: int = 0
    escalations: int = 0
    flag_duplicates: int = 0
    duplicate_cases_available: int = 0

    se_scored: int = 0
    se_correct: int = 0

    recovery_available: int = 0
    recovery_full: int = 0
    recovery_partial: int = 0

    ids_written: int = 0
    ids_ungrounded: int = 0
    evidence_cited_total: int = 0
    tool_calls: int = 0

    by_class: dict[str, list[int]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def action_accuracy(self) -> float | None:
        return _rate(self.action_correct, self.action_scored)

    @property
    def se_accuracy(self) -> float | None:
        return _rate(self.se_correct, self.se_scored)

    @property
    def recovery_rate(self) -> float | None:
        return _rate(self.recovery_full, self.recovery_available)

    @property
    def evidence_per_case(self) -> float | None:
        return _rate(self.evidence_cited_total, self.answered)


def score(cases: list[Case], arm: str) -> ArmScore:
    res = ArmScore(arm=arm, cases_total=len(cases))
    models: set[str] = set()

    for case in cases:
        if case.expected_resolution == "exception_duplicate":
            res.duplicate_cases_available += 1
        trace = case.arms.get(arm)
        if not trace:
            continue
        res.answered += 1
        if trace.get("model"):
            models.add(str(trace["model"]))

        action = trace.get("recommended_action")
        if action == "escalate_to_human":
            res.escalations += 1
        elif action == "flag_duplicate":
            res.flag_duplicates += 1
        elif action == "auto_resolve":
            if case.expected_resolution in UNSAFE_TO_CLEAR:
                res.unsafe_clears += 1
            elif case.expected_resolution in UNVERIFIED_TO_CLEAR:
                res.unverified_clears += 1

        if case.scored_for_action:
            bucket = res.by_class.setdefault(case.expected_resolution, [0, 0])
            bucket[0] += 1
            res.action_scored += 1
            if action == "escalate_to_human":
                res.action_correct += 1
                bucket[1] += 1
            else:
                res.failures.append({
                    "record_id": case.record_id, "dataset": case.dataset,
                    "expected_resolution": case.expected_resolution,
                    "recommended": action,
                    "severity": ("unsafe" if case.expected_resolution in UNSAFE_TO_CLEAR
                                 else "unverified"),
                })

        determined = DETERMINED_SUFFICIENT_EVIDENCE.get(case.expected_resolution)
        if determined is not None and trace.get("sufficient_evidence") is not None:
            res.se_scored += 1
            if bool(trace["sufficient_evidence"]) is determined:
                res.se_correct += 1

        if case.is_engine_miss and case.true_bank_rows:
            res.recovery_available += 1
            named = set(trace.get("evidence_cited") or [])
            named |= ids_in(trace.get("hypothesis"))
            if case.true_bank_rows <= named:
                res.recovery_full += 1
            elif case.true_bank_rows & named:
                # Not partial credit. Two legs of a three-leg group is a wrong
                # group that strands a credit - the same rule metrics.py holds
                # the engine to.
                res.recovery_partial += 1

        written = trace.get("evidence_cited") or []
        res.evidence_cited_total += len(written)
        res.tool_calls += len(trace.get("investigation_trace") or [])
        res.ids_ungrounded += len(ungrounded_in(case.bundle, trace))

    res.model = models.pop() if len(models) == 1 else ("MIXED" if models else "")
    return res


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _pct(v: float | None, width: int = 6) -> str:
    return f"{v:>{width}.1%}" if v is not None else f"{'n/a':>{width}}"


def format_report(res: ArmScore, control: ArmScore, cases: list[Case],
                  unmapped: list[str]) -> str:
    L: list[str] = []
    add = L.append
    rule = "=" * 78
    datasets = sorted({c.dataset for c in cases})
    per_ds = {d: sum(1 for c in cases if c.dataset == d) for d in datasets}

    add(rule)
    add(f"AGENT RECOMMENDATIONS - scored against ground truth "
        f"({' + '.join(datasets)})")
    add(rule)
    add("")
    add(f"  cases routed to the agent   {res.cases_total:>5}   "
        f"({', '.join(f'{n} {d}' for d, n in per_ds.items())})")
    add(f"  answers from this arm       {res.answered:>5}   arm: {res.arm}"
        + (f", model {res.model}" if res.model else ""))
    add(f"  unmappable                  {len(unmapped):>5}")
    add("")
    add("HEADLINE - unsafe auto-resolves")
    add(f"  unsafe auto-resolves        {res.unsafe_clears:>5}   cleared an item "
        f"where NO correct")
    add("                                      resolution exists: an ambiguous")
    add("                                      twin, an unmatchable row, or money")
    add("                                      still in transit.")
    add(f"  unverified auto-resolves    {res.unverified_clears:>5}   cleared a partial "
        f"refund. A correct")
    add("                                      resolution exists, but the ledger")
    add("                                      records refund status and never")
    add("                                      refund amount, so nothing in the")
    add("                                      sources can corroborate the gap.")
    add("")
    add("WHAT COULD BE SCORED FOR ACTION, AND WHAT COULD NOT")
    add(f"  scored                      {res.action_scored:>5}   escalation is the only")
    add("                                      defensible action for these")
    for cls in sorted(res.by_class, key=lambda c: -res.by_class[c][0]):
        n, ok = res.by_class[cls]
        add(f"      {cls:<30}{n:>5}")
    not_scored = [c for c in cases if not c.scored_for_action]
    add(f"  not scored                  {len(not_scored):>5}")
    misses = sum(1 for c in not_scored if c.is_engine_miss)
    add(f"      engine missed a matchable group {misses:>3}   escalating is not an "
        f"error here,")
    add("                                          it is a missed recovery")
    add(f"      engine already matched it       {len(not_scored) - misses:>3}   ground "
        f"truth confirms the engine")
    add(f"      duplicates                      {res.duplicate_cases_available:>3}   "
        f"never routed - the engine")
    add("                                          resolves them at 0.99 confidence,")
    add("                                          so flag_duplicate is untested,")
    add("                                          not perfect")
    add("")
    add("ACCURACY, BESIDE THE TRIVIAL POLICY THAT ESCALATES EVERYTHING")
    add(f"  {'metric':<34}{'agent':>10}{'always-esc':>12}")
    add(f"  {'action accuracy':<34}{_pct(res.action_accuracy, 10)}"
        f"{_pct(control.action_accuracy, 12)}")
    add(f"  {'unsafe auto-resolves':<34}{res.unsafe_clears:>10}"
        f"{control.unsafe_clears:>12}")
    add(f"  {'unverified auto-resolves':<34}{res.unverified_clears:>10}"
        f"{control.unverified_clears:>12}")
    add("")
    add("  Reading this: a policy that escalates every case scores full marks on")
    add("  action. That is not a flaw in the agent, it is a flaw in the metric, and")
    add("  it is printed alongside so the number cannot be quoted on its own. The")
    add("  column that separates the two policies is the next one.")
    add("")
    add("DID THE AGENT FIND ANYTHING THE ENGINE DID NOT")
    add(f"  recoverable cases           {res.recovery_available:>5}   the engine missed a "
        f"real match")
    add(f"    all true legs named       {res.recovery_full:>5}   {_pct(res.recovery_rate)}")
    add(f"    some legs, group incomplete {res.recovery_partial:>3}")
    add(f"    nothing found             "
        f"{res.recovery_available - res.recovery_full - res.recovery_partial:>5}")
    add("")
    add("  On the holdout batch the engine misses split settlements because the bank")
    add("  settles T+4 and the config says T+2 - a gap sensitivity.py exists to")
    add("  refuse to tune away. The agent has search_credits and four turns; it")
    add("  could look wider and assemble the legs. An incomplete group is a wrong")
    add("  group, not two-thirds of a right one, which is the rule the engine is")
    add("  held to in evaluation/metrics.py. This is the reasoning layer's measured")
    add("  contribution over the engine.")
    add("")
    add("SUFFICIENT_EVIDENCE")
    add(f"  scored on                   {res.se_scored:>5} of {res.cases_total} cases")
    add(f"  accuracy                    {_pct(res.se_accuracy, 5)}")
    add("  Scored only where ground truth fixes the answer: ambiguous twins and")
    add("  partial refunds, where it must be false. No case in this corpus has a")
    add("  ground-truth-determined 'true', so a constant 'always false' also scores")
    add("  full marks - this metric currently has no discriminating power and is")
    add("  printed as a count rather than sold as a result.")
    add("  Unmatchable and pending cases are not scored: the schema asks whether the")
    add("  evidence identifies a single answer, and for a settlement with no")
    add("  counterpart that answer is arguably 'nothing is here'. The ambiguity is")
    add("  in the schema, not in the agent.")
    add("")
    add("GROUNDING (same check as verify_grounding.py)")
    add(f"  ungrounded IDs              {res.ids_ungrounded:>5}")
    add(f"  evidence cited per case     {res.evidence_per_case:>5.2f}"
        if res.evidence_per_case is not None else "  evidence cited per case      n/a")
    if res.tool_calls:
        add(f"  tool calls                  {res.tool_calls:>5}")

    if res.failures:
        add("")
        add("RECOMMENDATIONS THAT DISAGREE WITH GROUND TRUTH")
        for f in res.failures:
            add(f"  [{f['severity']:<10}] {f['dataset']}/{f['record_id']:<18} "
                f"said {f['recommended']} on {f['expected_resolution']}")

    if unmapped:
        add("")
        add(f"UNMAPPED ({len(unmapped)}) - no ground-truth group, in no denominator")
        for u in unmapped:
            add(f"  {u}")

    add("")
    add(rule)
    return "\n".join(L)


def format_comparison(cases: list[Case]) -> str:
    L: list[str] = []
    add = L.append
    rule = "=" * 78
    results = {a: score(cases, a) for a in ARMS}

    add(rule)
    add("BASELINE COMPARISON - is the agent doing work a simpler path could not?")
    add(rule)
    add("")
    add(f"  {'arm':<18}{'answered':>9}{'action':>9}{'unsafe':>8}{'unver':>7}"
        f"{'recovery':>10}{'cited':>7}{'ungrnd':>8}")
    for a in ARMS:
        r = results[a]
        rec = (f"{r.recovery_full}/{r.recovery_available}"
               if r.recovery_available else "n/a")
        cited = f"{r.evidence_per_case:.2f}" if r.evidence_per_case is not None else "n/a"
        answered = str(r.answered) if r.answered else "not gen."
        add(f"  {a:<18}{answered:>9}{_pct(r.action_accuracy, 9)}"
            f"{r.unsafe_clears:>8}{r.unverified_clears:>7}{rec:>10}{cited:>7}"
            f"{r.ids_ungrounded:>8}")

    add("")
    if not results["shallow"].answered:
        add("  Arm 'shallow' has no committed traces, so the like-for-like")
        add("  shallow-vs-deep question - does the multi-turn loop beat one prompt? -")
        add("  is UNANSWERED. Generate it once with a key:")
        add("      python scripts/generate_shallow_arm.py --write")
    else:
        both = [c for c in cases if "shallow" in c.arms and "deep" in c.arms]
        s, d = score(both, "shallow"), score(both, "deep")
        add(f"  HEAD TO HEAD on the {len(both)} cases carrying both answers")
        add(f"    {'metric':<30}{'shallow':>10}{'deep':>10}")
        add(f"    {'action accuracy':<30}{_pct(s.action_accuracy, 10)}"
            f"{_pct(d.action_accuracy, 10)}")
        add(f"    {'unsafe auto-resolves':<30}{s.unsafe_clears:>10}{d.unsafe_clears:>10}")
        add(f"    {'recovery (full)':<30}{s.recovery_full:>10}{d.recovery_full:>10}")
        add(f"    {'evidence cited per case':<30}"
            f"{(s.evidence_per_case or 0):>10.2f}{(d.evidence_per_case or 0):>10.2f}")

    add("")
    add("  Reading this: the deterministic arm is a fixed lookup table with no model")
    add("  behind it. On this data the correct answer is 'escalate' for every scored")
    add("  case, so a table can say that for free and tie the agent on action. The")
    add("  agent's case has to rest on recovery and on the explanation a table")
    add("  cannot write. With few non-escalations across the corpus, this comparison")
    add("  has power to detect only gross differences between arms - it is not")
    add("  evidence that the arms are equivalent.")
    add("")
    add(rule)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the reasoning layer.")
    ap.add_argument("--dataset", nargs="+", default=list(DATASETS))
    ap.add_argument("--arm", default="deep", choices=list(ARMS))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cases, unmapped = load_cases(tuple(args.dataset))
    if not cases:
        print("No cases found. Has the dataset been generated?")
        return 1

    res = score(cases, args.arm)
    control = score(cases, "always_escalate")

    if args.json:
        print(json.dumps({
            "arm": args.arm, "cases": res.cases_total, "answered": res.answered,
            "action_scored": res.action_scored,
            "action_accuracy": res.action_accuracy,
            "unsafe_auto_resolves": res.unsafe_clears,
            "unverified_auto_resolves": res.unverified_clears,
            "se_scored": res.se_scored, "se_accuracy": res.se_accuracy,
            "recovery_available": res.recovery_available,
            "recovery_full": res.recovery_full,
            "ungrounded": res.ids_ungrounded,
            "unmapped": unmapped, "failures": res.failures,
        }, indent=2))
    else:
        print(format_report(res, control, cases, unmapped))
        if args.compare:
            print()
            print(format_comparison(cases))

    if res.unsafe_clears:
        print(f"\n  {res.unsafe_clears} unsafe auto-resolve(s): the agent cleared an "
              f"item that cannot be cleared.")
        return 1
    if res.ids_ungrounded:
        print(f"\n  {res.ids_ungrounded} grounding violation(s).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
