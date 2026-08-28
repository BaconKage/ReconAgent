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
import json
import re
import sys
from pathlib import Path

from agent.cache import TraceCache
from agent.reasoner import build_evidence_bundle, select_cases
from core.config import DEFAULT_CONFIG
from core.loader import load_batch
from core.matcher import reconcile
from core.normalize import make_stdout_safe

make_stdout_safe()

ROOT = Path(__file__).resolve().parent

#: The three record-ID shapes this domain uses. Anything matching one of these
#: in model-written text is a factual claim about a record, and is checkable.
ID_PATTERN = re.compile(r"\b(?:pay_[a-z0-9]+|order_[a-z0-9]+|BNK_\d+)\b")


def ids_in(obj: object) -> set[str]:
    """Every record ID appearing anywhere in a nested structure.

    Serialising and regexing rather than walking the shape deliberately: the
    point is to catch an ID *wherever* it appears, including in a narration
    string or a nested tool observation, and a structural walk would need
    updating every time either shape changed.
    """
    if obj is None:
        return set()
    blob = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return set(ID_PATTERN.findall(blob))


def permitted_ids(bundle: dict, trace: dict) -> set[str]:
    """What the model was allowed to know about, from its own point of view."""
    seen = ids_in(bundle)
    for step in trace.get("investigation_trace") or []:
        # Only the observation - the params are the model's own choice and are
        # therefore a claim to be checked, not a source to be trusted.
        seen |= ids_in(step.get("observation"))
    return seen


def written_ids(trace: dict) -> dict[str, set[str]]:
    """What the model asserted, split by where it asserted it."""
    out = {
        "hypothesis": ids_in(trace.get("hypothesis")),
        "evidence_cited": set(trace.get("evidence_cited") or []),
        "thought": set(),
        "tool_params": set(),
    }
    for step in trace.get("investigation_trace") or []:
        out["thought"] |= ids_in(step.get("thought"))
        out["tool_params"] |= ids_in(step.get("params"))
    return out


def build_case_index(datasets: list[str]) -> dict[str, dict]:
    """Map every case ID to the evidence bundle the reasoner would have built.

    Rebuilt from the committed CSVs through the real ``build_evidence_bundle``,
    so this verifies against the bundle the model actually got rather than a
    reconstruction that could drift from it.
    """
    index: dict[str, dict] = {}
    for name in datasets:
        batch = load_batch(ROOT / "data" / name)
        report = reconcile(batch, DEFAULT_CONFIG)
        for case in select_cases(report):
            index[case.record_id] = build_evidence_bundle(case, batch, DEFAULT_CONFIG)
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

    for trace in cache._data.values():
        case_id = trace.get("case_id")
        bundle = index.get(case_id)
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
