"""Export reproducible trial manifests for human participant studies."""

import json
from pathlib import Path
from typing import Any

from features.trial_generation.prompt_builder import build_fact_block
from features.trial_generation.trial_sampler import fact_order_rng, make_trial_specs
from shared.config import SEED
from shared.types import TrialSpec


def build_trial_manifest(
    groups: dict[int, list[dict]],
    sizes: list[int],
    n_trials: int,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Create model/human-shared trial manifests from sampled trial specs."""
    sampled_specs = make_trial_specs(groups, sizes, n_trials, seed=seed)
    manifest: list[dict[str, Any]] = []
    for n_examples, trials in sampled_specs.items():
        for trial_idx, entries in enumerate(trials):
            fact_block = build_fact_block(entries, fact_order_rng(seed, n_examples, trial_idx))
            spec = TrialSpec(
                trial_id=f"n{n_examples}-t{trial_idx}",
                trial=trial_idx,
                n_examples=n_examples,
                entries=entries,
                fact_block=fact_block,
            ).to_dict()
            spec["questions"] = [
                {
                    "group_id": entry["group_id"],
                    "target_position": entry["target_position"],
                    "target_fact": entry["target_fact"],
                    "memory_question": entry["memory_question"],
                    "reasoning_question": entry["reasoning_question"],
                    "memory_answer": entry["memory_answer"],
                    "reasoning_answer": entry["reasoning_answer"],
                }
                for entry in entries
            ]
            manifest.append(spec)
    return manifest


def export_trial_manifest(
    groups: dict[int, list[dict]],
    sizes: list[int],
    n_trials: int,
    out_path: Path,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Write a trial manifest for Prolific/PsyToolkit/browser study use."""
    manifest = build_trial_manifest(groups, sizes, n_trials, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
