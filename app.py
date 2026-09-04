"""ReconAgent - Streamlit interface.

    streamlit run app.py

Runs the same pipeline as `run_demo.py`. Nothing is computed here that is not
computed there; this is a view over the reconciliation, not a second
implementation of it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.llm import describe_provider, detect_provider_name
from agent.qa import ReconciliationQA
from agent.reasoner import ExceptionReasoner, build_evidence_bundle
from audit.trail import AuditTrail
from cash.position import compute_position
from core.loader import load_batch
from core.matcher import reconcile
from core.normalize import format_inr, paise_to_rupees
from evaluation import agent_eval as ae
from evaluation.metrics import evaluate

ROOT = Path(__file__).resolve().parent

STATUS_LABEL = {
    "matched": "Matched",
    "matched_split": "Matched (split)",
    "duplicate": "Duplicate",
    "unresolved": "Exception",
}

REASON_LABEL = {
    "identifier_match_amount_discrepancy": "UTR matched, amount disagreed",
    "ambiguous_candidates": "Several credits qualified",
    "contested_candidate": "One credit, two claimants",
    "ambiguous_split": "Several subsets summed to net",
    "contested_split_legs": "Split legs claimed twice",
    "weak_amount_date_evidence": "Match too likely to be coincidence",
    "no_candidate_found": "No credit within thresholds",
    "no_settlement_counterpart": "Credit with no settlement",
    "duplicate_settlement_row": "Webhook retry",
}

st.set_page_config(page_title="ReconAgent", page_icon="R", layout="wide")


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def run_pipeline(dataset: str, use_llm: bool):
    batch = load_batch(ROOT / "data" / dataset)
    report = reconcile(batch)

    trail = AuditTrail(directory=ROOT / "audit_trail")
    for r in report.results:
        trail.append_decision(r)

    reasoner = ExceptionReasoner(report.config, use_llm=use_llm)
    outcome = reasoner.investigate(report, batch, trail=trail)

    ev = evaluate(report, ROOT / "data" / dataset, batch=batch,
                  reasoning_seconds=outcome.stats.elapsed_seconds if use_llm else None,
                  llm_free_groups=outcome.stats.never_touched_llm)
    pos = compute_position(report, batch)
    # Score the reasoning layer for this batch too. It is cheap here because the
    # reconciliation is already done, and a measurement that only exists in a
    # terminal is one a reviewer will never see.
    agent_cases, _ = ae.load_cases((dataset,))
    agent = ae.score(agent_cases, "deep")
    control = ae.score(agent_cases, "always_escalate")
    return {"batch": batch, "report": report, "outcome": outcome, "ev": ev,
            "pos": pos, "trail": trail, "dataset": dataset,
            "agent": agent, "control": control}


def results_frame(state) -> pd.DataFrame:
    report, batch = state["report"], state["batch"]
    settlements = {s.transaction_id: s for s in batch.settlements}
    rows = []
    for r in report.results:
        s = settlements.get(r.record_id)
        rows.append({
            "Record": r.record_id,
            "Status": STATUS_LABEL.get(r.status, r.status),
            "How": r.match_type or (REASON_LABEL.get(r.exception_reason or "",
                                                     r.exception_reason) or ""),
            "Confidence": round(r.confidence, 3),
            "Amount": paise_to_rupees(s.net_paise) if s else "",
            "Bank rows": ", ".join(r.linked_ids["bank"]) or "-",
            "Ledger": ", ".join(r.linked_ids["ledger"]) or "-",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("ReconAgent")
st.sidebar.caption("Three-way reconciliation with an honest exception list")

dataset = st.sidebar.selectbox(
    "Dataset", ["dev", "holdout"],
    help="holdout is the sealed evaluation set: different seed, harder mix, and a "
         "bank narration format the parser was never tuned on",
)

has_key = detect_provider_name() is not None
use_llm = st.sidebar.toggle(
    "Reasoning layer", value=True,
    help="Explains exceptions. It cannot change a single match - turn it off and "
         "every metric below is identical.",
)

if has_key:
    st.sidebar.success(f"Live reasoning via {describe_provider()}")
else:
    st.sidebar.info("No API key - reasoning replays from committed traces. "
                    "All matching and every metric is unaffected.")

if st.sidebar.button("Run reconciliation", type="primary", use_container_width=True):
    with st.spinner(f"Reconciling {dataset}..."):
        st.session_state.state = run_pipeline(dataset, use_llm)

if "state" not in st.session_state:
    st.session_state.state = run_pipeline(dataset, use_llm)

state = st.session_state.state
ev, pos, report, outcome = state["ev"], state["pos"], state["report"], state["outcome"]

st.sidebar.divider()
st.sidebar.metric("Rows processed", ev.total_rows)
st.sidebar.metric("Reconciliation groups", ev.total_groups)
# No rows/sec here: these batches match in a couple of milliseconds, and a rate
# divided out of that measures the interpreter rather than the engine. Throughput
# is a benchmark.py question, not a demo-header question.
st.sidebar.caption(f"Engine: {ev.deterministic_seconds * 1000:.1f} ms "
                   f"(too fast at this size to quote a rate)")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

st.title(f"Reconciliation - {state['dataset']} batch")

tab_overview, tab_results, tab_exceptions, tab_cash, tab_ask = st.tabs(
    ["Overview", "Results", "Exceptions", "Cash position", "Ask"])


# ---- Overview ------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Auto-match rate", f"{ev.auto_match_rate:.1%}",
              help="Groups resolved with no human involvement")
    c2.metric("Precision", f"{ev.precision:.1%}",
              help="Of what it auto-matched, the share that was exactly right")
    c3.metric("Recall", f"{ev.recall:.1%}",
              help="Of what was matchable, the share it found")
    # Not throughput. These batches match in a couple of milliseconds, and a rate
    # divided out of that measures the interpreter - the overstatement DEVLOG
    # entry 9 retracts. A headline metric card is the worst possible place to
    # keep printing a number the README takes back, so this shows the honest
    # figure instead and sends throughput to the ramp that can measure it.
    c4.metric("Adversarial pairs conflated",
              f"{ev.adversarial_conflated} / {ev.adversarial_pairs}",
              help="Near-duplicate transactions the engine could have merged and did not. "
                   "Throughput is deliberately not shown: a batch this size finishes in "
                   "milliseconds, so a rows/sec figure would measure the interpreter. "
                   "Run benchmark.py --rows 250 5000 25000 for a real ramp.")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("False positives")
        st.caption("Money bound to the wrong counterpart, or a match forced. "
                   "This is the expensive error.")
        st.metric("False positives", ev.false_positives)
        st.metric("Adversarial pairs conflated",
                  f"{ev.adversarial_conflated} / {ev.adversarial_pairs}",
                  help="Near-duplicate transactions wrongly merged")
        if ev.false_positives == 0 and ev.adversarial_conflated == 0:
            st.success("No incorrect link was asserted anywhere in this batch.")
        else:
            st.error(f"{ev.false_positives} incorrect links asserted.")
            for d in ev.false_positive_detail[:5]:
                st.write(f"**{d['group_id']}** - {d['kind']}")
                st.caption(f"expected {d['expected']} / predicted {d['predicted']}")

    with right:
        st.subheader("False negatives")
        st.caption("Matchable, but held back for a human. This is the cheap error, "
                   "and it is deliberately preferred.")
        st.metric("False negatives", ev.false_negatives)
        st.metric("Exceptions correctly held",
                  f"{ev.exceptions_correctly_held} / {ev.exceptions_expected}")
        if ev.false_negatives:
            st.info("Each of these is a case the engine could have guessed at and "
                    "chose not to. On the held-out batch these are split settlements "
                    "whose legs arrive later than the configured T+2 window.")

    st.divider()
    st.divider()
    st.subheader("Is the reasoning layer any good?")
    st.caption("Its recommendations scored against the same ground truth the "
               "engine is scored against - `python -m evaluation.agent_eval`.")
    agent, control = state["agent"], state["control"]
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Unsafe auto-resolves", agent.unsafe_clears,
              help="Cleared an item where NO correct resolution exists: an "
                   "ambiguous twin, an unmatchable row, or money still in "
                   "transit. This is the number that fails the build.")
    g2.metric("Unverified auto-resolves", agent.unverified_clears,
              help="Cleared a partial refund. A correct resolution exists, but "
                   "the ledger records refund status and never refund amount, "
                   "so nothing in the sources can corroborate the gap.")
    g3.metric("Recommendation accuracy",
              f"{agent.action_accuracy:.1%}" if agent.action_accuracy is not None else "n/a",
              delta=(f"{(agent.action_accuracy - control.action_accuracy):+.1%} vs "
                     f"always-escalate"
                     if None not in (agent.action_accuracy, control.action_accuracy)
                     else None),
              delta_color="normal",
              help="Against a policy that escalates every case and looks at "
                   "nothing. If the trivial policy ties it, the number is not "
                   "evidence the agent is doing work.")
    g4.metric("Recovered engine misses",
              f"{agent.recovery_full} / {agent.recovery_available}",
              help="Groups the engine failed to match where the agent named the "
                   "complete true set of bank legs. An incomplete group is a "
                   "wrong group, not partial credit.")
    if agent.recovery_available and not agent.recovery_full:
        st.info("The agent has not yet recovered a single match the engine "
                "missed. Published here rather than omitted - it is the honest "
                "answer to whether the reasoning layer earns its place on "
                "evidence-finding, and today it does not.")

    st.divider()
    st.subheader("Where the model is and is not used")
    m1, m2, m3 = st.columns(3)
    stats = outcome.stats
    m1.metric("Groups reaching a model", stats.sent_to_llm,
              f"{stats.deep_investigations} investigated with tools"
              if stats.deep_investigations else None, delta_color="off")
    m2.metric("Handled with no model", stats.never_touched_llm,
              f"{stats.llm_free_fraction:.0%} of the batch")
    m3.metric("Live API calls", stats.live_calls)
    st.caption(
        "Matching is fully deterministic. The reasoning layer only explains what "
        "the engine already decided, and clean matches never reach it - which is "
        "both a cost argument and a correctness one."
    )

    st.divider()
    st.subheader("Accuracy by case type")
    st.dataframe(pd.DataFrame([
        {"Case": c, "n": b.total, "Correct": b.correct,
         "False pos": b.false_positive, "False neg": b.false_negative,
         "Accuracy": f"{b.accuracy:.0%}"}
        for c, b in sorted(ev.by_case_type.items(), key=lambda kv: -kv[1].total)
    ]), use_container_width=True, hide_index=True)


# ---- Results -------------------------------------------------------------
with tab_results:
    df = results_frame(state)
    choices = st.multiselect("Filter by status", sorted(df["Status"].unique()),
                             default=sorted(df["Status"].unique()))
    view = df[df["Status"].isin(choices)] if choices else df
    st.caption(f"{len(view)} of {len(df)} groups")
    st.dataframe(view, use_container_width=True, hide_index=True, height=560)


# ---- Exceptions ----------------------------------------------------------
with tab_exceptions:
    unresolved = report.unresolved
    st.caption(f"{len(unresolved)} items the engine would not resolve on its own.")

    # Lead with cases where the engine had a candidate in front of it and
    # declined. An orphan credit with nothing nearby is a trivial absence; a
    # settlement with a credit just outside tolerance is a judgement call.
    nets = {x.transaction_id: x.net_paise for x in state["batch"].settlements}

    def sort_key(r):
        inv = outcome.investigations.get(r.record_id, {})
        return (not r.near_misses, inv.get("sufficient_evidence", True),
                -nets.get(r.record_id, 0), r.record_id)

    ordered = sorted(unresolved, key=sort_key)

    # Two cases carry the whole argument, and hunting for them in a list of
    # thirty during a demo is how a three-minute walkthrough becomes six. They
    # are pinned to the top and labelled, so the contrast can be shown in two
    # clicks: one the data cannot decide, and one where the agent accounts for
    # the discrepancy arithmetically and says so.
    FEATURED = {
        "ambiguous_candidates": "refuses - two identical credits, no UTR",
        "identifier_match_amount_discrepancy": "UTR matches, amount short",
    }

    def is_featured(r):
        inv = outcome.investigations.get(r.record_id, {})
        if r.exception_reason == "ambiguous_candidates":
            return True
        return inv.get("recommended_action") == "auto_resolve"

    ordered = sorted(ordered, key=lambda r: not is_featured(r))

    def label_for(r):
        base = (f"{r.record_id}  -  "
                f"{REASON_LABEL.get(r.exception_reason or '', r.exception_reason)}")
        if not is_featured(r):
            return base
        inv = outcome.investigations.get(r.record_id, {})
        if inv.get("recommended_action") == "auto_resolve":
            return f"[start here] {base}  -  agent accounts for the gap"
        return f"[start here] {base}  -  undecidable from the data"

    labels = {label_for(r): r for r in ordered}
    picked = st.selectbox(
        "Exception", list(labels),
        help="The two cases marked [start here] are the contrast worth showing: "
             "one the data cannot decide at any tolerance, and one where the "
             "agent closes the arithmetic and clears it.")
    r = labels[picked]
    inv = outcome.investigations.get(r.record_id, {})

    a, b = st.columns([3, 2])
    with a:
        st.subheader("What the engine did")
        for line in r.rule_trace:
            st.markdown(f"- `{line}`")

        if r.near_misses:
            st.subheader("Candidates it considered and declined")
            st.dataframe(pd.DataFrame([{
                "Record": nm.record_id,
                "Amount differs by": (paise_to_rupees(nm.amount_delta_paise)
                                      if nm.amount_delta_paise is not None else ""),
                "Days after settlement": (nm.date_delta_days
                                          if nm.date_delta_days is not None else ""),
                "Why declined": nm.reason,
            } for nm in r.near_misses]), use_container_width=True, hide_index=True)

    with b:
        st.subheader("Agent investigation")
        if not inv:
            st.info("No investigation recorded for this case.")
        else:
            src = inv.get("source", "unknown")
            if src == "unavailable":
                st.warning("Reasoning layer unavailable - the engine's rule trace "
                           "still records exactly why this did not match.")
            else:
                st.caption(f"source: {src}")
            st.write(inv.get("hypothesis", ""))
            # The verdict is the point of this panel, so it gets its own row
            # rather than another line of prose. sufficient_evidence is the
            # field that makes "I cannot tell" a first-class answer.
            # st.metric truncates a long value to fit its column, and
            # "escalate_to_human" rendered as "escal..." - the single most
            # important word on the panel, unreadable. Short labels, with the
            # full value in the tooltip.
            SHORT = {"escalate_to_human": "escalate",
                     "auto_resolve": "auto-resolve",
                     "flag_duplicate": "flag dup"}
            action = inv.get("recommended_action", "-")
            v1, v2, v3 = st.columns(3)
            v1.metric("Action", SHORT.get(action, action), help=str(action))
            v2.metric("Confidence", str(inv.get("confidence", "-")))
            v3.metric("Evidence sufficient",
                      "yes" if inv.get("sufficient_evidence") else "no",
                      help="Whether the agent judged the evidence enough to "
                           "decide. 'no' is a correct and valued answer.")
            if inv.get("rupee_impact") and inv["rupee_impact"] != "unknown":
                # The model is free to write "Rs 48,615.93" or just "48615.93",
                # and it does both. Prefixing unconditionally rendered "Rs Rs
                # 48,615.93" on screen, so only add the unit when it is absent.
                impact = str(inv["rupee_impact"]).strip()
                if not impact.lower().startswith(("rs", "₹")):
                    impact = f"Rs {impact}"
                st.write(f"**Amount at stake:** {impact}")
            if inv.get("evidence_cited"):
                st.write(f"**Cites:** {', '.join(inv['evidence_cited'])}")

            if inv.get("sufficient_evidence") is False:
                st.error(
                    "The agent judged the evidence insufficient to decide this case "
                    "and deferred to human review rather than forcing a match."
                )

        trace = inv.get("investigation_trace") if inv else None
        if trace:
            st.markdown("**How it investigated**")
            st.caption(f"{len(trace)} turns using read-only query tools. The agent "
                       f"chose each step; none of these tools can change a match.")
            for st_ in trace:
                params = {k: v for k, v in (st_.get("params") or {}).items() if v}
                with st.expander(f"Turn {st_['turn']} - {st_['action']}"
                                 + (f" {params}" if params else "")):
                    st.write(st_["thought"])
                    if st_.get("observation"):
                        st.json(st_["observation"])

        with st.expander("Evidence the agent started from"):
            st.json(build_evidence_bundle(r, state["batch"]))


# ---- Cash position -------------------------------------------------------
with tab_cash:
    st.caption(f"Books closed as of {pos.as_of.isoformat()}, "
               f"settlement window T+{pos.window_days}")

    st.subheader("In the account")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reconciled and explained", format_inr(pos.confirmed_paise))
    c2.metric("Arrived but unreconciled", format_inr(pos.held_for_review_paise))
    c3.metric("Unattributed inflows", format_inr(pos.unattributed_paise))
    st.progress(pos.explained_fraction,
                text=f"{pos.explained_fraction:.0%} of money in the account is fully explained")

    st.divider()
    st.subheader("Expected, not yet received")
    c1, c2 = st.columns(2)
    c1.metric(f"In flight (within T+{pos.window_days})", format_inr(pos.in_flight_paise),
              f"{len(pos.in_flight)} settlements", delta_color="off")
    c2.metric("Overdue", format_inr(pos.overdue_paise),
              f"{len(pos.overdue)} settlements", delta_color="inverse")
    st.caption("Kept apart deliberately: a settlement at T+1 with no credit is "
               "normal, and the same settlement at T+9 is an incident.")

    if pos.schedule:
        st.write("**Expected arrival schedule**")
        st.bar_chart(pd.DataFrame(
            [{"date": d.isoformat(), "expected": v / 100} for d, v in sorted(pos.schedule.items())]
        ).set_index("date"))

    if pos.overdue:
        st.write("**Overdue receivables**")
        st.dataframe(pd.DataFrame([{
            "Transaction": i.transaction_id,
            "Customer": i.customer,
            "Amount": paise_to_rupees(i.net_paise),
            "Settled": i.settlement_date.isoformat(),
            "Days outstanding": i.days_outstanding,
        } for i in pos.overdue]), use_container_width=True, hide_index=True)

    st.divider()
    st.metric("Needs a human today", format_inr(pos.at_risk_paise))
    if pos.unattributed_paise and pos.overdue_paise:
        st.info(
            f"{format_inr(pos.unattributed_paise)} of unattributed credits sits alongside "
            f"{format_inr(pos.overdue_paise)} of overdue receivables. These are plausibly "
            "the same money - and that is exactly the pairing the engine declined to "
            "assert on the evidence available."
        )


# ---- Ask -----------------------------------------------------------------
with tab_ask:
    st.subheader("Why did this transaction reconcile, or not?")
    st.caption("Answered by retrieving the audit trail for that record. If a "
               "transaction is not in the trail, the answer says so rather than "
               "inventing one.")

    examples = [r.record_id for r in sorted(report.unresolved, key=lambda x: x.record_id)[:3]]
    if examples:
        st.caption("Try: " + " . ".join(f"`{e}`" for e in examples))

    question = st.text_input("Question", placeholder="why didn't pay_xxxxxxxxxx reconcile?")
    phrase = st.checkbox("Let the model phrase the answer", value=False,
                         disabled=not has_key,
                         help="Reads the same retrieved record either way. Off, the "
                              "answer is rendered directly from the trail.")
    if question:
        qa = ReconciliationQA(state["trail"].path)
        ans = qa.ask(question, use_llm=phrase)
        if ans.found:
            st.success(f"source: {ans.source}")
            st.write(ans.answer)
            with st.expander("The audit record this came from"):
                st.json(ans.record)
        else:
            st.warning(ans.answer)
