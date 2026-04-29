"""Human participant result ingestion and normalization."""

import csv
import json
from pathlib import Path
from typing import Any

from features.scoring import score

HUMAN_REQUIRED_FIELDS = {
    "participant_id",
    "trial",
    "n_examples",
    "group_id",
    "target_position",
    "memory_prediction",
    "reasoning_prediction",
    "memory_truth",
    "reasoning_truth",
}


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_human_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one human response row to the model log schema."""
    missing = HUMAN_REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"Human row is missing required fields: {sorted(missing)}")

    memory_correct = _parse_bool(row.get("memory_correct"))
    reasoning_correct = _parse_bool(row.get("reasoning_correct"))
    if memory_correct is None:
        memory_correct = score(row["memory_prediction"], row["memory_truth"])
    if reasoning_correct is None:
        reasoning_correct = score(row["reasoning_prediction"], row["reasoning_truth"])

    normalized = dict(row)
    normalized.update({
        "source": "human",
        "model": row.get("model", "human"),
        "mode": row.get("mode", "human"),
        "condition": row.get("condition", "human"),
        "trial": int(row["trial"]),
        "n_examples": int(row["n_examples"]),
        "n_groups": int(row.get("n_groups") or row["n_examples"]),
        "group_id": int(row["group_id"]),
        "target_position": int(row["target_position"]),
        "memory_correct": memory_correct,
        "reasoning_correct": reasoning_correct,
        "memory_error": row.get("memory_error") or None,
        "reasoning_error": row.get("reasoning_error") or None,
    })
    return normalized


def load_human_results(path: Path) -> list[dict[str, Any]]:
    """Load CSV or JSONL human study rows into the shared result schema."""
    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return [normalize_human_row(row) for row in rows]
