"""Generate the one-shot arm, so shallow-vs-deep can be measured like for like.

`evaluation/agent_eval.py` compares what the reasoning layer recommends against a
deterministic lookup table. It cannot yet answer the sharper question a reviewer
will ask - *why a multi-turn agent instead of one prompt?* - because no case
carries both a one-shot answer and a deep one. Every committed trace was written
during a deep-enabled run, so all 104 are keyed under the deep instructions.

This script fills the gap. It runs the same cases with `deep=False`, which changes
the instructions string and therefore the cache key, so the one-shot answers land
in their own slots and cannot overwrite anything. Verified before writing, and
asserted again at run time.

    python scripts/generate_shallow_arm.py              # dry run, no calls
    python scripts/generate_shallow_arm.py --write      # needs an API key

Roughly 18 batched calls across both datasets. Commit the result: once the traces
are in, the comparison replays offline forever and CI needs no key.

This is a build tool, deliberately outside `evaluation/` - that package reads
ground truth and must stay free of anything that can reach the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This script lives one directory down, so the repo root is not on the path when
# it is invoked directly. Everything else in the project runs from the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.cache import TraceCache, evidence_key  # noqa: E402
from agent.investigator import system_prompt as investigation_system_prompt  # noqa: E402
from agent.llm import describe_provider  # noqa: E402
from agent.prompts import system_prompt  # noqa: E402
from agent.reasoner import ExceptionReasoner, build_evidence_bundle, select_cases  # noqa: E402
from core.config import DEFAULT_CONFIG  # noqa: E402
from core.loader import load_batch  # noqa: E402
from core.matcher import reconcile  # noqa: E402
from core.normalize import make_stdout_safe  # noqa: E402

make_stdout_safe()

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("dev", "holdout")


def plan() -> tuple[int, int, int]:
    """(cases, already present as shallow, would collide with an existing key).

    The collision count is the safety check. It must be zero: a shallow key that
    already exists in the cache would mean this run overwrites a deep answer, and
    the whole comparison rests on the two arms being separately addressable.
    """
    cache = TraceCache()
    shallow_i = system_prompt(DEFAULT_CONFIG)
    deep_i = shallow_i + investigation_system_prompt(DEFAULT_CONFIG)

    total = present = collisions = 0
    for name in DATASETS:
        batch = load_batch(ROOT / "data" / name)
        report = reconcile(batch, DEFAULT_CONFIG)
        for case in select_cases(report):
            bundle = build_evidence_bundle(case, batch, DEFAULT_CONFIG)
            total += 1
            s_key = evidence_key(bundle, shallow_i)
            if s_key in cache:
                present += 1
            if s_key == evidence_key(bundle, deep_i):
                collisions += 1
    return total, present, collisions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="actually call the model and write traces (needs a key)")
    ap.add_argument("--dataset", nargs="+", default=list(DATASETS))
    args = ap.parse_args()

    total, present, collisions = plan()
    print(f"  cases                       {total}")
    print(f"  already have a shallow answer {present}")
    print(f"  would overwrite a deep answer {collisions}")

    if collisions:
        print("\n  REFUSING: a shallow key collides with a deep one. Writing would")
        print("  destroy the arm this comparison depends on.")
        return 1

    if not args.write:
        print(f"\n  Dry run. {total - present} case(s) would be sent to the model.")
        print("  Re-run with --write to generate them.")
        return 0

    print(f"\n  provider: {describe_provider()}")
    cache = TraceCache()
    before = len(cache)

    for name in args.dataset:
        batch = load_batch(ROOT / "data" / name)
        report = reconcile(batch, DEFAULT_CONFIG)
        # deep=False is the whole point: it changes the instructions, hence the
        # key, hence which slot the answer lands in.
        reasoner = ExceptionReasoner(report.config, cache=cache,
                                     use_llm=True, deep=False)
        outcome = reasoner.investigate(report, batch)
        s = outcome.stats
        print(f"  {name:<9} {len(outcome.investigations):>3} cases, "
              f"{s.live_calls} live call(s), {s.api_errors} error(s)"
              + (" - PROVIDER UNAVAILABLE" if s.unavailable else ""))
        if s.unavailable or s.api_errors:
            print("\n  Stopping: placeholders are never cached, so a partial run")
            print("  leaves the cache consistent - but the arm would be incomplete.")
            return 1

    cache.save()
    print(f"\n  cache {before} -> {len(cache)} entries. Commit agent/cache/traces.json,")
    print("  then: python -m evaluation.agent_eval --compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
