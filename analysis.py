"""Analysis CLI for the IB human-vs-LLM experiment.

Loads all model logs from the checkpoint + any completed run directories,
optionally merges human results, computes IB rate-distortion points,
saves tables, and generates plots.

Usage:
    python3 analysis.py
    python3 analysis.py --human data/human_results.csv
    python3 analysis.py --out results/analysis/ --mode separate
"""

import argparse
import csv
import json
from pathlib import Path

from features.analysis import (
    compute_ib_points,
    human_baseline_points,
    pivot_for_paper,
    plot_memory_vs_reasoning,
    plot_per_family,
    plot_rate_distortion,
    plot_scaling,
)
from features.human_study import load_human_results
from features.results_io import read_jsonl

PROJECT_ROOT = Path(__file__).parent
CHECKPOINT   = PROJECT_ROOT / "results" / "full" / "separate" / "checkpoint.jsonl"
RESULTS_DIR  = PROJECT_ROOT / "results"
DEFAULT_OUT  = PROJECT_ROOT / "results" / "analysis"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_model_rows(results_dir: Path, mode: str = "separate") -> list[dict]:
    """Collect model question rows for a specific mode only.

    Uses mode in the dedupe key so rows from different modes never collide.
    Only loads files that belong to the requested mode.
    """
    rows: list[dict] = []
    seen: set[tuple] = set()

    def _add(batch: list[dict]) -> None:
        for r in batch:
            if r.get("mode", "separate") != mode:
                continue
            key = (r.get("model",""), r.get("mode",""), r.get("n_examples", r.get("n_groups",0)),
                   r.get("trial",0), r.get("group_id",0), r.get("target_position",0))
            if key not in seen:
                seen.add(key)
                rows.append(r)

    # 1. Checkpoint for the requested mode
    cp = results_dir / "full" / mode / "checkpoint.jsonl"
    if cp.exists():
        _add([json.loads(l) for l in cp.open() if l.strip()])
        print(f"Loaded checkpoint     : {cp}  ({len(rows)} rows so far)")

    # 2. gz logs only from the matching mode directory
    mode_dir = results_dir / "model_runs" / "full" / mode
    if mode_dir.exists():
        for gz in sorted(mode_dir.rglob("logs.jsonl.gz")):
            batch = read_jsonl(gz)
            before = len(rows)
            _add(batch)
            added = len(rows) - before
            if added:
                print(f"Loaded run log        : {gz}  (+{added} new rows)")

    return rows


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_ib_points(ib_points: list[dict], out_dir: Path) -> None:
    path = out_dir / "ib_points.json"
    with path.open("w") as f:
        json.dump(ib_points, f, indent=2)
    print(f"Saved IB points       → {path}")


def save_summary_table(ib_points: list[dict], out_dir: Path) -> None:
    rows = pivot_for_paper(ib_points)
    if not rows:
        return
    path = out_dir / "summary_table.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary table   → {path}")


def save_accuracy_table(ib_points: list[dict], out_dir: Path) -> None:
    """Long-format CSV easy to load into pandas / R for further analysis."""
    path = out_dir / "accuracy_by_n.csv"
    fields = [
        "source", "model", "mode", "n_examples",
        "memory_accuracy", "reasoning_accuracy", "reasoning_given_memory",
        "memory_distortion", "reasoning_distortion",
        "n_questions",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ib_points)
    print(f"Saved accuracy table  → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--human", type=Path, default=None,
                   help="Path to human results CSV or JSONL.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output directory. Default: {DEFAULT_OUT}")
    p.add_argument("--mode", default="separate",
                   choices=["separate", "combined", "summarize"],
                   help="Which model run mode to plot.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation (just save JSON/CSV tables).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    # 1. Load model rows
    print("\n── Loading model data ─────────────────────────────────────")
    model_rows = load_all_model_rows(RESULTS_DIR, mode=args.mode)
    print(f"Total model rows      : {len(model_rows)}")

    # 2. Load human rows if provided
    human_rows: list[dict] = []
    if args.human and args.human.exists():
        human_rows = load_human_results(args.human)
        print(f"Total human rows      : {len(human_rows)}")
    elif args.human:
        print(f"[warn] Human file not found: {args.human}")

    all_rows = model_rows + human_rows

    # 3. Compute IB points + inject human baseline
    print("\n── Computing IB rate-distortion points ────────────────────")
    ib_points = compute_ib_points(all_rows)
    baseline = human_baseline_points()
    ib_points += baseline
    print(f"IB points computed    : {len(ib_points)} ({len(baseline)} human baseline points injected)")

    # Quick text summary
    print(f"\n── Accuracy summary ({args.mode} mode) ───────────────────────")
    header = f"{'Model':<45}  N    mem%    rea%  rea|mem%"
    print(header)
    print("─" * len(header))
    for pt in sorted(ib_points, key=lambda x: (x["source"], x["model"], x["n_examples"])):
        if pt["mode"] not in (args.mode, "human"):
            continue
        print(
            f"{pt['model']:<45} {pt['n_examples']:>2}"
            f"  {pt['memory_accuracy']:>6.1%}"
            f"  {pt['reasoning_accuracy']:>6.1%}"
            f"  {pt['reasoning_given_memory']:>7.1%}"
        )

    # 4. Save tables
    print("\n── Saving tables ───────────────────────────────────────────")
    save_ib_points(ib_points, out_dir)
    save_summary_table(ib_points, out_dir)
    save_accuracy_table(ib_points, out_dir)

    # 5. Generate plots
    if not args.no_plots:
        print("\n── Generating plots ────────────────────────────────────────")
        plot_rate_distortion(ib_points, plots_dir, mode_filter=args.mode)
        plot_memory_vs_reasoning(ib_points, plots_dir, mode_filter=args.mode)
        plot_scaling(ib_points, plots_dir, mode_filter=args.mode)
        plot_per_family(ib_points, plots_dir, mode_filter=args.mode)

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
