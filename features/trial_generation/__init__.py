"""Trial generation feature exports."""

from .prompt_builder import build_fact_block, build_prompt, group_names
from .trial_runner import run_grid, run_trial, run_trial_combined, run_trial_summarize
from .trial_sampler import fact_order_rng, make_trial_specs, sample_trial_entries

__all__ = [
    "build_fact_block",
    "build_prompt",
    "fact_order_rng",
    "group_names",
    "make_trial_specs",
    "run_grid",
    "run_trial",
    "run_trial_combined",
    "run_trial_summarize",
    "sample_trial_entries",
]
