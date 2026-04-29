"""Reusable aggregation for comparing human and model result logs."""

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from features.human_study import load_human_results
from features.results_io import read_jsonl


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1 + z * z / total
    center = (p_hat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denom
    return round(center - margin, 4), round(center + margin, 4)


def aggregate_rows(rows: Iterable[dict]) -> list[dict]:
    """Aggregate rows by source/model/mode/n_examples with confidence intervals."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("source", "model"),
            row.get("model", "unknown"),
            row.get("mode", "unknown"),
            int(row.get("n_examples", row.get("n_groups", 0))),
        )
        buckets[key].append(row)

    summary: list[dict] = []
    for (source, model, mode, n_examples), bucket in sorted(buckets.items()):
        usable = [row for row in bucket if not row.get("memory_error") and not row.get("reasoning_error")]
        total = len(usable)
        memory_successes = sum(1 for row in usable if row.get("memory_correct"))
        reasoning_successes = sum(1 for row in usable if row.get("reasoning_correct"))
        both_successes = sum(
            1 for row in usable
            if row.get("memory_correct") and row.get("reasoning_correct")
        )
        memory_ci = wilson_interval(memory_successes, total)
        reasoning_ci = wilson_interval(reasoning_successes, total)
        summary.append({
            "source": source,
            "model": model,
            "mode": mode,
            "n_examples": n_examples,
            "n_questions": total,
            "memory_accuracy": round(memory_successes / total, 4) if total else 0.0,
            "memory_ci_low": memory_ci[0],
            "memory_ci_high": memory_ci[1],
            "reasoning_accuracy": round(reasoning_successes / total, 4) if total else 0.0,
            "reasoning_ci_low": reasoning_ci[0],
            "reasoning_ci_high": reasoning_ci[1],
            "reasoning_given_memory": round(both_successes / memory_successes, 4)
            if memory_successes else 0.0,
        })
    return summary


def compare_human_and_model_logs(
    model_log_path: Path,
    human_result_path: Path,
) -> list[dict]:
    """Load model JSONL plus human CSV/JSONL and return comparable aggregates."""
    model_rows = read_jsonl(model_log_path)
    human_rows = load_human_results(human_result_path)
    return aggregate_rows([*model_rows, *human_rows])
