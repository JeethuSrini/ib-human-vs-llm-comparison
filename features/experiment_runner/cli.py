"""Command-line orchestration for model runs and human manifests."""

import argparse
import json
import os
import sys
from pathlib import Path

from features.human_study import export_trial_manifest
from features.model_inference import OpenRouterAuthError
from features.results_io import build_model_run_dir, load_dataset, make_run_id, save_results
from features.trial_generation import run_grid
from shared.config import (
    DATASET_PATH,
    MODELS,
    N_EXAMPLES_PER_PROMPT,
    N_TRIALS,
    RESULTS_DIR,
    SMOKE_MODEL,
    SMOKE_TRIALS,
)


def select_models(filter_text: str | None) -> list[str]:
    """Select configured models, optionally filtering by substring."""
    if not filter_text:
        return MODELS
    selected = [model_id for model_id in MODELS if filter_text in model_id]
    if not selected:
        raise SystemExit(f"No configured models match --models={filter_text!r}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Smoke test: 1 model ({SMOKE_MODEL}) x sizes x {SMOKE_TRIALS} trials.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Use combined mode: both questions in one API call.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Use summarize mode: compress facts first, then query from summary.",
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
    parser.add_argument(
        "--export-human-manifest",
        type=Path,
        default=None,
        help="Write a trial manifest JSON for a human study before any model run.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only export the human manifest; do not call model APIs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    if args.export_human_manifest:
        manifest = export_trial_manifest(
            groups, N_EXAMPLES_PER_PROMPT, n_trials, args.export_human_manifest
        )
        print(f"Wrote human trial manifest with {len(manifest)} trials -> {args.export_human_manifest}")
        if args.manifest_only:
            return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set in environment.")

    calls_per_fact = 3 if args.summarize else 2
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
    except OpenRouterAuthError as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
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
    run_id = make_run_id()
    out_dir = build_model_run_dir(RESULTS_DIR, run_tag, mode_tag, run_id=run_id)
    metadata = {
        "run_id": run_id,
        "run_type": run_tag,
        "mode": mode_tag,
        "models": models,
        "sizes": N_EXAMPLES_PER_PROMPT,
        "trials_per_condition": n_trials,
        "dataset_path": str(DATASET_PATH),
    }
    save_results(summary, logs, out_dir, metadata=metadata)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
