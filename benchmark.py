"""Throughput measurement at realistic batch sizes.

The headline number in the README was measured over 252 rows in under two
milliseconds. That is not a throughput measurement - at that size you are timing
interpreter noise, not the algorithm. This script generates progressively larger
batches from the same seeded generator and measures the deterministic engine on
each, so the reported figure means something.

It ramps rather than jumping straight to the largest size, and abandons the ramp
once a single run exceeds `--budget` seconds. A matcher with quadratic behaviour
will not politely take twice as long on twice the data, and waiting out a run
that was never going to finish tells you nothing you did not already know at the
previous size.

    python benchmark.py
    python benchmark.py --rows 1000 5000 25000 --budget 120

Generated batches are written to a temporary directory and deleted; nothing large
is committed to the repository.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from core.loader import load_batch
from core.matcher import reconcile
from core.normalize import make_stdout_safe
from data.generator import DEV_MIX, ReconDataGenerator, write_dataset

make_stdout_safe()

#: The dev mix produces this many CSV rows; used to convert a target row count
#: into a multiplier for the case mix.
ROWS_PER_UNIT_MIX = 252

#: Shortest run worth using as the denominator of a scaling ratio. Below this,
#: timer resolution dominates and the ratio describes the interpreter rather
#: than the algorithm. See DEVLOG entry 11.
MEASURABLE_SECONDS = 0.05


def scaled_mix(target_rows: int) -> dict:
    """Scale the dev case mix to land near `target_rows` total rows."""
    factor = max(1, round(target_rows / ROWS_PER_UNIT_MIX))
    return {case: count * factor for case, count in DEV_MIX.items()}


def generate(target_rows: int, out_dir: Path, seed: int = 7) -> dict:
    gen = ReconDataGenerator(seed, scaled_mix(target_rows), max_lag_days=2)
    return write_dataset(out_dir, gen)


def measure(directory: Path, repeats: int = 1) -> dict:
    load_times, match_times = [], []
    report = None
    batch = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        batch = load_batch(directory)
        t1 = time.perf_counter()
        report = reconcile(batch)
        t2 = time.perf_counter()
        load_times.append(t1 - t0)
        match_times.append(t2 - t1)
    return {
        "rows": batch.total_rows,
        "settlements": len(batch.settlements),
        "banks": len(batch.banks),
        "groups": len(report.results),
        "load_s": statistics.median(load_times),
        "match_s": statistics.median(match_times),
        "rows_per_s": batch.total_rows / statistics.median(match_times),
        "matched": report.by_status().get("matched", 0)
        + report.by_status().get("matched_split", 0),
        "unresolved": report.by_status().get("unresolved", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure engine throughput by batch size.")
    ap.add_argument("--rows", type=int, nargs="+",
                    default=[250, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000])
    ap.add_argument("--budget", type=float, default=180.0,
                    help="abandon the ramp once one run exceeds this many seconds")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--keep", action="store_true", help="keep the generated batches")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="recon_scale_"))
    print(f"generating into {workdir}\n")
    header = (f"{'target':>9}{'rows':>9}{'settle':>9}{'banks':>9}"
              f"{'gen s':>9}{'load s':>9}{'match s':>10}{'rows/s':>12}{'matched':>9}")
    print(header)
    print("-" * len(header))

    results = []
    for target in args.rows:
        out = workdir / f"n{target}"
        t0 = time.perf_counter()
        generate(target, out)
        gen_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        stats = measure(out, repeats=args.repeats)
        wall = time.perf_counter() - t0

        stats["target"] = target
        stats["gen_s"] = gen_s
        results.append(stats)
        # Same rule as the scaling verdict below: a rate is only printed when the
        # run was long enough for it to mean anything. Quoting 177,000 rows/sec
        # off a sub-millisecond run is the exact overstatement this script was
        # written to correct.
        rate = (f"{stats['rows_per_s']:>12,.0f}"
                if stats["match_s"] >= MEASURABLE_SECONDS else f"{'-':>12}")
        print(f"{target:>9,}{stats['rows']:>9,}{stats['settlements']:>9,}"
              f"{stats['banks']:>9,}{gen_s:>9.1f}{stats['load_s']:>9.2f}"
              f"{stats['match_s']:>10.2f}{rate}"
              f"{stats['matched']:>9,}")

        if not args.keep:
            shutil.rmtree(out, ignore_errors=True)
        if stats["match_s"] > args.budget:
            print(f"\n  stopping: matching took {stats['match_s']:.0f}s, over the "
                  f"{args.budget:.0f}s budget.")
            break

    # Scaling behaviour is the actual finding, so state it rather than leaving it
    # to be eyeballed off the table.
    if len(results) >= 2:
        print("\nscaling between consecutive sizes (rows x, match time x):")
        for a, b in zip(results, results[1:]):
            rx = b["rows"] / a["rows"]
            tx = b["match_s"] / a["match_s"] if a["match_s"] else float("inf")
            # A ratio against a baseline too short to measure is not a scaling
            # result. The 250-row batch matches in under a millisecond, so
            # dividing by it reports timer noise - and would print "quadratic"
            # for an engine that is nothing of the sort. Refusing to classify
            # here is the same discipline as the engine refusing an ambiguous
            # match: the measurement does not support a verdict.
            if a["match_s"] < MEASURABLE_SECONDS:
                verdict = f"(baseline under {MEASURABLE_SECONDS * 1000:.0f} ms - not measurable)"
            else:
                verdict = ("linear" if tx < rx * 1.4 else
                           "super-linear" if tx < rx * rx * 0.6 else "quadratic")
            print(f"  {a['rows']:>7,} -> {b['rows']:>7,}   "
                  f"rows x{rx:>5.1f}   time x{tx:>7.1f}   {verdict}")

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
