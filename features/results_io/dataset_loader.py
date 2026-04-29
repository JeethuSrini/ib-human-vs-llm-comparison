"""Dataset loading for the IB experiment."""

import json
from pathlib import Path


def load_dataset(path: Path) -> tuple[list[dict], dict[int, list[dict]]]:
    """Load dataset and return flat entries plus group_id-indexed entries."""
    with path.open() as handle:
        data = json.load(handle)
    if not data:
        raise ValueError(f"Dataset at {path} is empty.")
    groups: dict[int, list[dict]] = {}
    for entry in data:
        groups.setdefault(entry["group_id"], []).append(entry)
    return data, groups
