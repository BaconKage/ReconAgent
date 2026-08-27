"""ReconAgent - one-command reconciliation run.

    python run_demo.py                      # reconcile the dev batch
    python run_demo.py --dataset holdout    # the sealed evaluation set
    python run_demo.py --ask pay_xxxxxxxxxx # why did this reconcile?

Runs with or without ANTHROPIC_API_KEY. Without one, the deterministic engine
produces every match and metric exactly as it otherwise would, and the agent's
explanations are replayed from committed traces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.qa import ReconciliationQA
from agent.reasoner import ExceptionReasoner
from audit.trail import AuditTrail
from cash.position import compute_position, format_position
from core.loader import load_batch
from core.matcher import reconcile
from core.normalize import format_inr
from evaluation.metrics import evaluate, format_report

ROOT = Path(__file__).resolve().parent
RULE = "=" * 78


def hr(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a reconciliation over a synthetic batch.")
    ap.add_argument("--dataset", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the reasoning layer entirely (engine only)")
    ap.add_argument("--ask", metavar="QUESTION",
                    help="ask why a transaction did or did not reconcile, then exit")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip scoring against ground truth")
    ap.add_argument("--exceptions", type=int, default=5,
                    help="how many exceptions to print in full (default 5)")
    args = ap.parse_args()

    if args.ask:
        qa = ReconciliationQA()
        if qa.trail_path is None:
            print("No audit trail found. Run `python run_demo.py` first.")
            return 1
        ans = qa.ask(args.ask)
        hr(f"Q: {args.ask}")
        print(f"[source: {ans.source}]\n")
        print(ans.answer)
        return 0

    data_dir = ROOT / "data" / args.dataset
    if not data_dir.exists():
        print(f"Dataset not found: {data_dir}\nRun: python -m data.generator")
        return 1

    # ---- 1. load -----------------------------------------------------
    batch = load_batch(data_dir)
    hr(f"ReconAgent - {args.dataset} batch")
    print(f"Loaded {batch.summary()}")
    if batch.rejected:
        print(f"  {len(batch.rejected)} rows could not be parsed and were skipped")

    # ---- 2. deterministic engine -------------------------------------
    report = reconcile(batch)
    counts = report.by_status()
    resolved = sum(counts.get(k, 0) for k in ("matched", "matched_split"))

    hr("Deterministic engine")
    print(f"  matched            {counts.get('matched', 0):>4}")
    print(f"  matched (split)    {counts.get('matched_split', 0):>4}")
    print(f"  duplicates flagged {counts.get('duplicate', 0):>4}")
    print(f"  exceptions         {counts.get('unresolved', 0):>4}")
    print(f"  {'-' * 30}")
    print(f"  groups             {len(report.results):>4}")
    print(f"\n  {report.elapsed_seconds * 1000:.1f} ms for {report.rows_processed} rows "
          f"({report.throughput:,.0f} rows/sec, no LLM involved)")

    # ---- 3. audit trail ----------------------------------------------
    trail = AuditTrail(directory=ROOT / "audit_trail")
    for r in report.results:
        trail.append_decision(r)

    # ---- 4. reasoning layer ------------------------------------------
    reasoner = ExceptionReasoner(report.config, use_llm=not args.no_llm)
    outcome = reasoner.investigate(report, batch, trail=trail)
    s = outcome.stats

    hr("Reasoning layer")
    mode = ("disabled (--no-llm)" if args.no_llm
            else "cached traces (no API key)" if s.unavailable
            else f"live ({s.live_calls} calls, {s.served_from_cache} from cache)")
    print(f"  mode                     {mode}")
    print(f"  groups reaching a model  {s.sent_to_llm} of {s.total_records}")
    print(f"  handled with no model    {s.never_touched_llm} ({s.llm_free_fraction:.0%})")
    if s.api_errors:
        print(f"  api errors (degraded)    {s.api_errors}")
    print(f"\n  Matching does not depend on any of this. The numbers above are")
    print(f"  identical whether or not a key is present.")

    # ---- 5. exception list -------------------------------------------
    hr(f"Exception list - {len(report.unresolved)} items needing attention")
    by_reason: dict[str, int] = {}
    for r in report.unresolved:
        by_reason[r.exception_reason or "unknown"] = by_reason.get(r.exception_reason or "unknown", 0) + 1
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3}  {reason}")

    escalated = [rid for rid, inv in outcome.investigations.items()
                 if inv.get("recommended_action") == "escalate_to_human"]
    refused = [rid for rid, inv in outcome.investigations.items()
               if inv.get("sufficient_evidence") is False]
    print(f"\n  agent recommends human review for {len(escalated)}")
    print(f"  agent judged evidence insufficient to decide for {len(refused)}")

    # Lead with the cases the agent declined to force - those are the ones
    # worth a human's time, and the ones worth showing.
    ordered = sorted(
        report.unresolved,
        key=lambda r: (outcome.investigations.get(r.record_id, {})
                       .get("sufficient_evidence", True), r.record_id),
    )
    for r in ordered[:args.exceptions]:
        inv = outcome.investigations.get(r.record_id, {})
        print(f"\n  {'-' * 74}")
        print(f"  {r.record_id}   [{r.exception_reason}]")
        for line in r.rule_trace:
            print(f"      . {line}")
        if inv:
            print(f"      agent    : {inv.get('hypothesis', '')}")
            print(f"      action   : {inv.get('recommended_action')} "
                  f"(confidence {inv.get('confidence')}, "
                  f"evidence sufficient: {inv.get('sufficient_evidence')})")
            if inv.get("evidence_cited"):
                print(f"      cites    : {', '.join(inv['evidence_cited'])}")

    # ---- 6. evaluation against ground truth --------------------------
    gt_path = data_dir / "ground_truth.json"
    if gt_path.exists() and not args.no_eval:
        ev = evaluate(report, data_dir, batch=batch,
                      reasoning_seconds=s.elapsed_seconds if s.sent_to_llm else None,
                      llm_free_groups=s.never_touched_llm)
        print()
        print(format_report(ev))

    # ---- 7. forward cash position ------------------------------------
    print()
    print(format_position(compute_position(report, batch)))

    hr("Audit trail")
    print(f"  {len(trail)} entries written to {trail.path}")
    print(f"\n  Ask about any transaction:")
    print(f"    python run_demo.py --ask \"why didn't {ordered[0].record_id} reconcile?\""
          if ordered else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
