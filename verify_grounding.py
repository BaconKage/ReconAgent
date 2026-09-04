"""Check every record ID the model wrote against the evidence it was shown.

The central claim of the reasoning layer is that it explains rather than invents.
That claim was quoted in the README before anything measured it, which is exactly
the mistake entry 9 of DEVLOG.md is about - so this script exists to make the
number reproducible instead of asserted.

    python verify_grounding.py [--dataset dev holdout] [--show-grounded]

An ID is **grounded** if it appears somewhere the model could legitimately have
read it:

* the evidence bundle the engine handed it, or
* the observation returned by a read-only tool it called during the
  investigation - the model genuinely saw those rows.

An ID is **model-written** if it appears anywhere the model produced text:

* ``hypothesis`` - the explanation itself,
* ``evidence_cited`` - the IDs it claims to be relying on,
* each turn's ``thought`` - its stated reason for a query,
* each turn's ``params`` - the arguments it chose, which is where a
  hallucinated lookup would surface first.

Counting ``params`` and ``thought`` makes this stricter than a check on the final
answer alone. A model that invents a plausible-looking bank row and then asks a
tool about it has hallucinated, even if the tool returns nothing and the invented
ID never reaches the conclusion. That is the failure this is meant to catch.

Exit code is non-zero if anything is ungrounded, so it can run in CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.cache import TraceCache, evidence_key
from agent.investigator import system_prompt as investigation_system_prompt
from agent.prompts import system_prompt
from agent.reasoner import build_evidence_bundle, select_cases
from evaluation.grounding import permitted_ids, written_ids
from core.config import DEFAULT_CONFIG
from core.loader import load_batch
from core.matcher import reconcile
from core.normalize import make_stdout_safe

make_stdout_safe()

ROOT = Path(__file__).resolve().parent

def build_case_index(datasets: list[str]) -> dict[str, dict]:
    """Map every cache key to the evidence bundle that produced it.

    Keyed by the cache key, not by record id. Five bank row ids - BNK_00002,
    BNK_00027, BNK_00032, BNK_00035, BNK_00056 - exist in both dev and holdout
    with different rows behind them, so a record-id index silently let the
    holdout bundle overwrite the dev one and checked those traces against
    evidence they were never shown. It reported zero either way, which is
    exactly why it went unnoticed: a check that is accidentally right is
    indistinguishable from one that is right on purpose.

    The cache key is a hash of the bundle plus the instructions, so it is unique
    per (case, prompt) by construction and cannot collide across datasets.
    """
    index: dict[str, dict] = {}
    shallow = system_prompt(DEFAULT_CONFIG)
    deep = shallow + investigation_system_prompt(DEFAULT_CONFIG)
    for name in datasets:
        batch = load_batch(ROOT / "data" / name)
        report = reconcile(batch, DEFAULT_CONFIG)
        for case in select_cases(report):
            bundle = build_evidence_bundle(case, batch, DEFAULT_CONFIG)
            for instructions in (shallow, deep):
                index[evidence_key(bundle, instructions)] = bundle
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["dev", "holdout"])
    ap.add_argument("--show-grounded", action="store_true",
                    help="list every checked ID, not just the failures")
    args = ap.parse_args()

    cache = TraceCache()
    index = build_case_index(args.dataset)

    checked = ungrounded = 0
    per_field: dict[str, int] = {}
    traces_checked = 0
    unresolved: list[str] = []
    failures: list[tuple[str, str, str]] = []

    for key, trace in cache._data.items():
        case_id = trace.get("case_id")
        bundle = index.get(key)
        if bundle is None:
            # A trace whose case is not in these datasets cannot be checked
            # here. Reported rather than dropped - a silent skip would let the
            # denominator quietly shrink to whatever passes.
            unresolved.append(case_id or "(no case_id)")
            continue

        traces_checked += 1
        allowed = permitted_ids(bundle, trace)
        for field, ids in written_ids(trace).items():
            for rid in sorted(ids):
                checked += 1
                per_field[field] = per_field.get(field, 0) + 1
                if rid not in allowed:
                    ungrounded += 1
                    failures.append((case_id, field, rid))
                elif args.show_grounded:
                    print(f"  ok   {case_id:<18} {field:<15} {rid}")

    print("=" * 74)
    print("GROUNDING - every record ID the model wrote, checked against its evidence")
    print("=" * 74)
    print(f"  datasets                {', '.join(args.dataset)}")
    print(f"  traces in cache         {len(cache)}")
    print(f"  traces checked          {traces_checked}")
    if unresolved:
        print(f"  traces not checkable    {len(unresolved)} (case absent from these datasets)")
    print()
    print(f"  record IDs written      {checked}")
    for field in ("hypothesis", "evidence_cited", "thought", "tool_params"):
        if per_field.get(field):
            print(f"    in {field:<20} {per_field[field]}")
    print()
    print(f"  ungrounded              {ungrounded}")

    if failures:
        print("\n  FAILURES - an ID the model wrote that it was never shown:")
        for case_id, field, rid in failures:
            print(f"    {case_id:<18} {field:<15} {rid}")
        print("\n  Ungrounded IDs found. This is the anti-hallucination claim failing.")
        return 1

    print("\n  Every record ID the model wrote appears in the evidence it was shown"
          "\n  or in a tool result it received. Nothing was invented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
