"""Categorized filesystem paths for experiment outputs."""

from datetime import datetime, timezone
from pathlib import Path


def make_run_id(started_at: datetime | None = None) -> str:
    """Return a sortable UTC run id for result directories."""
    timestamp = started_at or datetime.now(timezone.utc)
    return timestamp.strftime("%Y%m%dT%H%M%SZ")


def build_model_run_dir(
    results_dir: Path,
    run_tag: str,
    mode_tag: str,
    run_id: str | None = None,
) -> Path:
    """Group model outputs by run type, mode, and concrete run id."""
    return results_dir / "model_runs" / run_tag / mode_tag / (run_id or make_run_id())


def build_human_manifest_dir(results_dir: Path, run_id: str | None = None) -> Path:
    """Group exported human-study manifests separately from model outputs."""
    return results_dir / "human_manifests" / (run_id or make_run_id())
