"""Result persistence for model and human experiment outputs."""

import csv
import gzip
import json
from pathlib import Path
from typing import Any

RESULT_CSV_FIELDS = [
    "source", "model", "mode", "condition", "participant_id",
    "n_examples", "n_groups", "trial", "group_id", "target_position",
    "memory_correct", "reasoning_correct",
    "memory_prediction", "reasoning_prediction",
    "memory_truth", "reasoning_truth",
    "memory_error", "reasoning_error", "elapsed_seconds",
]


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read plain or gzip-compressed JSON lines."""
    rows: list[dict[str, Any]] = []
    with _open_text(path, "r") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows as compact JSON lines, optionally gzip-compressed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def save_results(
    summary: dict[str, Any],
    logs: list[dict[str, Any]],
    out_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist one categorized run directory with compact logs and metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    metadata_path = out_dir / "metadata.json"
    if metadata is not None:
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    write_jsonl(logs, out_dir / "logs.jsonl.gz")
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for entry in logs:
            writer.writerow(entry)
    print(f"\nWrote run      -> {out_dir}")
    print(f"Wrote summary  -> {out_dir / 'summary.json'}")
    if metadata is not None:
        print(f"Wrote metadata -> {metadata_path}")
    print(f"Wrote logs     -> {out_dir / 'logs.jsonl.gz'}")
    print(f"Wrote csv      -> {out_dir / 'results.csv'}")
