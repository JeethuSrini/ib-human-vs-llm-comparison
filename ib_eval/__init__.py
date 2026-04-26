"""IB human-vs-LLM LLM evaluation package."""

from .client import OpenRouterAuthError, OpenRouterError, call_openrouter
from .config import (
    DATASET_PATH,
    MAX_TOKENS,
    MODELS,
    N_EXAMPLES_PER_PROMPT,
    N_TRIALS,
    OPENROUTER_URL,
    RESULTS_DIR,
    SEED,
    SMOKE_MODEL,
    SMOKE_TRIALS,
    SYSTEM_INSTRUCTION,
    TEMPERATURE,
)
from .io import load_dataset, save_results
from .trials import run_grid, run_trial, run_trial_combined, run_trial_summarize
from .scoring import build_vocab, extract_answer, score
from .prompts import build_fact_block, build_prompt, group_names, sample_trial_entries

__all__ = [
    "OpenRouterAuthError",
    "OpenRouterError",
    "call_openrouter",
    "DATASET_PATH",
    "MAX_TOKENS",
    "MODELS",
    "N_EXAMPLES_PER_PROMPT",
    "N_TRIALS",
    "OPENROUTER_URL",
    "RESULTS_DIR",
    "SEED",
    "SMOKE_MODEL",
    "SMOKE_TRIALS",
    "SYSTEM_INSTRUCTION",
    "TEMPERATURE",
    "load_dataset",
    "save_results",
    "run_grid",
    "run_trial",
    "run_trial_combined",
    "run_trial_summarize",
    "build_vocab",
    "extract_answer",
    "score",
    "build_fact_block",
    "build_prompt",
    "group_names",
    "sample_trial_entries",
]
