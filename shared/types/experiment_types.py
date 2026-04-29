"""Typed contracts shared across experiment features."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrialSpec:
    """A reproducible trial shown to either humans or models."""

    trial_id: str
    trial: int
    n_examples: int
    entries: list[dict[str, Any]]
    fact_block: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuestionResult:
    """Normalized row emitted by model and human study pipelines."""

    source: str
    model: str
    mode: str
    n_examples: int
    trial: int
    group_id: int
    target_position: int
    target_fact: list[str]
    memory_truth: str
    reasoning_truth: str
    memory_prediction: str
    reasoning_prediction: str
    memory_correct: bool
    reasoning_correct: bool
    memory_raw: str = ""
    reasoning_raw: str = ""
    memory_error: str | None = None
    reasoning_error: str | None = None
    elapsed_seconds: float | None = None
    participant_id: str | None = None
    condition: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {})
        data.update(extra)
        return data


RunSummary = dict[str, dict[str, dict[str, float | int]]]
