"""Analysis feature exports."""

from .human_model_comparison import (
    aggregate_rows,
    compare_human_and_model_logs,
    wilson_interval,
)
from .ib_curves import compute_ib_points, human_baseline_points, pivot_for_paper
from .plot import plot_memory_vs_reasoning, plot_per_family, plot_rate_distortion, plot_scaling

__all__ = [
    "aggregate_rows",
    "compare_human_and_model_logs",
    "wilson_interval",
    "compute_ib_points",
    "human_baseline_points",
    "pivot_for_paper",
    "plot_rate_distortion",
    "plot_memory_vs_reasoning",
    "plot_scaling",
    "plot_per_family",
]
