"""Trial sampling and deterministic manifest preparation."""

import random

from shared.config import SEED


def sample_trial_entries(
    groups: dict[int, list[dict]],
    n: int,
    rng: random.Random,
    max_attempts: int = 200,
) -> list[dict]:
    """Sample N entries from distinct groups, avoiding target-name collisions."""
    group_ids = list(groups.keys())

    for _ in range(max_attempts):
        chosen_gids = rng.sample(group_ids, n)
        entries = [rng.choice(groups[gid]) for gid in chosen_gids]
        left_names = [entry["target_fact"][0] for entry in entries]
        right_names = [entry["target_fact"][1] for entry in entries]
        if len(set(left_names)) == n and len(set(right_names)) == n:
            return entries

    chosen_gids = rng.sample(group_ids, n)
    return [rng.choice(groups[gid]) for gid in chosen_gids]


def make_trial_specs(
    groups: dict[int, list[dict]],
    sizes: list[int],
    n_trials: int,
    seed: int = SEED,
) -> dict[int, list[list[dict]]]:
    """Pre-sample trial entries so every model and human run shares items."""
    rng = random.Random(seed)
    trial_specs: dict[int, list[list[dict]]] = {}
    for n_examples in sizes:
        trial_specs[n_examples] = [
            sample_trial_entries(groups, n_examples, rng)
            for _ in range(n_trials)
        ]
    return trial_specs


def fact_order_rng(seed: int, n_examples: int, trial_idx: int) -> random.Random:
    """Stable fact-order RNG shared by humans and all model providers."""
    return random.Random(f"{seed}|{n_examples}|{trial_idx}")
