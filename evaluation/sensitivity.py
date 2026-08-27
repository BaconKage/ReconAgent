"""Threshold sensitivity analysis.

The held-out run exposed a recall gap on split settlements: that batch models a
bank on a T+4 settlement cycle, while the engine is configured for T+2, so legs
arriving on day three and four fall outside the window and the split cannot be
assembled.

The tempting response is to widen the window until the number improves. That
would be tuning on the evaluation set, and it would also be the wrong instinct:
the date window is a *domain* parameter - your bank's actual settlement SLA - not
a hyperparameter to be fitted. Widening it globally buys recall and pays for it
in candidate collisions, which is precisely how false positives are born.

So instead of changing the default, this module measures the trade. It sweeps the
two thresholds across both datasets and reports what each setting costs and buys.
The shipped configuration is unchanged by anything found here; `core/config.py`
has not been modified since the first commit, which is checkable in git history.

    python -m evaluation.sensitivity
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from core.config import DEFAULT_CONFIG, MatchConfig
from core.loader import load_batch
from core.matcher import reconcile
from evaluation.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("dev", "holdout")


def sweep(param: str, values: list, datasets=DATASETS) -> list[dict]:
    rows = []
    batches = {d: load_batch(ROOT / "data" / d) for d in datasets}
    for v in values:
        cfg = replace(DEFAULT_CONFIG, **{param: v})
        row: dict = {"value": v, "is_default": getattr(DEFAULT_CONFIG, param) == v}
        for d in datasets:
            rep = reconcile(batches[d], cfg)
            ev = evaluate(rep, ROOT / "data" / d, batch=batches[d])
            row[d] = {
                "precision": ev.precision,
                "recall": ev.recall,
                "fp": ev.false_positives,
                "fn": ev.false_negatives,
                "adv_conflated": ev.adversarial_conflated,
                "auto_rate": ev.auto_match_rate,
            }
        rows.append(row)
    return rows


def render(title: str, param: str, rows: list[dict], unit: str = "") -> str:
    out = [
        "",
        "=" * 78,
        title,
        "=" * 78,
        f"  {param:<14}{'dev P':>8}{'dev R':>8}{'dev FP':>8}   "
        f"{'hold P':>8}{'hold R':>8}{'hold FP':>8}{'hold adv':>10}",
    ]
    for r in rows:
        mark = " <- shipped" if r["is_default"] else ""
        d, h = r["dev"], r["holdout"]
        out.append(
            f"  {str(r['value']) + unit:<14}"
            f"{d['precision']:>8.0%}{d['recall']:>8.0%}{d['fp']:>8}   "
            f"{h['precision']:>8.0%}{h['recall']:>8.0%}{h['fp']:>8}{h['adv_conflated']:>10}"
            f"{mark}"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep matching thresholds.")
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead of a table")
    args = ap.parse_args()

    window = sweep("date_window_days", [1, 2, 3, 4, 5, 7])
    tolerance = sweep("amount_tolerance_paise", [0, 10, 50, 100, 500, 5000])
    legs = sweep("max_split_legs", [2, 3, 4, 5, 6])

    if args.json:
        import json
        print(json.dumps({"date_window_days": window,
                          "amount_tolerance_paise": tolerance,
                          "max_split_legs": legs}, indent=2))
        return

    print(render("SETTLEMENT DATE WINDOW - how many days a credit may lag",
                 "window (days)", window, "d"))
    print("""
  Reading this: the shipped T+2 leaves recall on the table for the held-out
  batch, which models a slower-settling bank. Widening it recovers that recall.
  The default is NOT changed here, because the right value is a property of the
  bank you are reconciling against, not of this test set - and because a window
  wide enough to catch every straggler is also wide enough to let two unrelated
  credits look like candidates for the same settlement.""")

    print(render("AMOUNT TOLERANCE - how far a credit may sit from the net",
                 "tolerance", tolerance, "p"))
    print("""
  Tolerance is the knob that actually endangers precision. Widening it pulls
  genuinely unrelated transactions into the same candidate set, which is what
  manufactures near-duplicate collisions.""")

    print(render("MAX SPLIT LEGS - subset-sum search bound", "max legs", legs))
    print("""
  More legs means more subsets, and with enough subsets something always sums to
  the target by coincidence.""")


if __name__ == "__main__":
    main()
