"""Dataset loading and result persistence."""

import csv
import json
from pathlib import Path


def load_dataset(path: Path) -> tuple[list[dict], dict[int, list[dict]]]:
    """Load dataset and return (flat_entries, groups).

    groups maps group_id -> list of entries sharing the same prompt.
    All entries in a group cover a different target fact from the same prompt.
    """
    with path.open() as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"Dataset at {path} is empty.")
    groups: dict[int, list[dict]] = {}
    for entry in data:
        gid = entry["group_id"]
        groups.setdefault(gid, []).append(entry)
    return data, groups


def save_results(summary: dict, logs: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "logs.jsonl").open("w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")
    csv_fields = [
        "model", "n_examples", "trial",
        "memory_correct", "reasoning_correct",
        "memory_prediction", "reasoning_prediction",
        "memory_truth", "reasoning_truth",
        "memory_error", "reasoning_error",
        "elapsed_seconds",
    ]
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for entry in logs:
            writer.writerow(entry)
    print(f"\nWrote summary -> {out_dir / 'summary.json'}")
    print(f"Wrote logs    -> {out_dir / 'logs.jsonl'}")
    print(f"Wrote csv     -> {out_dir / 'results.csv'}")
