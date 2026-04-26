"""LLM evaluation pipeline for the IB memory-vs-reasoning experiment.

For each (model, N) configuration, runs N_TRIALS trials.
A trial:
  1. Samples N entries, ONE from each of N different groups.
     Each entry contributes exactly one pairing fact (A paired with B)
     and one attribute fact (B lives in City) to the prompt.
  2. Builds a single shuffled fact block of N pairs + N attributes (2N lines).
  3. Asks memory AND reasoning questions about EVERY one of the N people.
     Score = (correct answers) / N — 100% means the model remembered everyone.
  4. Scores via case-insensitive exact match on extracted answer token.

This directly measures: given N facts, how many can the model recall/reason over?
Varying N gives the rate-distortion curve: more facts = harder = lower accuracy.

Outputs:
  results/<run>/<mode>/summary.json  -- per (model, N) accuracies
  results/<run>/<mode>/logs.jsonl    -- one JSON line per question
  results/<run>/<mode>/results.csv   -- flat table for plotting

Usage:
  export OPENROUTER_API_KEY=...
  python3 run_experiment.py --smoke              # 1 model x sizes x 10 trials
  python3 run_experiment.py                      # full grid
  python3 run_experiment.py --models gemma-3-4b  # substring filter on MODELS
  python3 run_experiment.py --smoke --combined   # both Qs in one call
  python3 run_experiment.py --smoke --summarize  # compress-then-query
"""

import argparse
import json
import os
import sys

from ib_eval.client import OpenRouterAuthError
from ib_eval.config import (
    DATASET_PATH,
    MODELS,
    N_EXAMPLES_PER_PROMPT,
    N_TRIALS,
    RESULTS_DIR,
    SMOKE_MODEL,
    SMOKE_TRIALS,
)
from ib_eval.io import load_dataset, save_results
from ib_eval.trials import run_grid


def select_models(filter_substr: str | None) -> list[str]:
    if not filter_substr:
        return MODELS
    matches = [m for m in MODELS if filter_substr in m]
    if not matches:
        raise SystemExit(
            f"No models matched filter '{filter_substr}'. Known: {MODELS}"
        )
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Smoke test: 1 model ({SMOKE_MODEL}) x sizes x {SMOKE_TRIALS} trials.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Use combined mode: both questions in one API call (run_trial_combined).",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Use summarize mode: compress facts first, then query from summary (run_trial_summarize).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Substring filter on model IDs (e.g. 'gemma-3-4b').",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help=f"Override trial count per (model, size). Default {N_TRIALS} ({SMOKE_TRIALS} for smoke).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set in environment.")

    if args.smoke:
        models = [SMOKE_MODEL]
        n_trials = args.trials or SMOKE_TRIALS
    else:
        models = select_models(args.models)
        n_trials = args.trials or N_TRIALS

    dataset, groups = load_dataset(DATASET_PATH)
    avg_k = len(dataset) / len(groups)
    print(
        f"Loaded {len(dataset)} entries ({len(groups)} groups, avg {avg_k:.1f} facts/group)"
    )
    print(f"Models: {models}")
    print(f"Sizes:  {N_EXAMPLES_PER_PROMPT}  | trials per (model, size): {n_trials}")
    calls_per_fact = 3 if args.summarize else 2 if args.combined else 2
    calls_per_trial = round(avg_k * calls_per_fact + (1 if args.summarize else 0))
    print(
        f"Approx API calls planned: ~{len(models) * len(N_EXAMPLES_PER_PROMPT) * n_trials * calls_per_trial}"
    )

    try:
        summary, logs = run_grid(
            models,
            N_EXAMPLES_PER_PROMPT,
            n_trials,
            dataset,
            groups,
            api_key,
            combined=args.combined,
            summarize=args.summarize,
        )
    except OpenRouterAuthError as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        print(
            "Check that OPENROUTER_API_KEY is set to a valid key from "
            "https://openrouter.ai/keys (no quotes, no trailing whitespace).",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.summarize:
        mode_tag = "summarize"
    elif args.combined:
        mode_tag = "combined"
    else:
        mode_tag = "separate"
    run_tag = "smoke" if args.smoke else "full"
    out_dir = RESULTS_DIR / run_tag / mode_tag
    save_results(summary, logs, out_dir)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
