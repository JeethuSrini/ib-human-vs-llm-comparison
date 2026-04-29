"""Runtime constants and paths for the IB human-vs-LLM study."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "ib_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "results"

SEED = 20260426

MODELS = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
    "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.2-11b-vision-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen3-8b",
    "qwen/qwen3-14b",
    "qwen/qwen3-32b",
]
N_EXAMPLES_PER_PROMPT = [3, 8, 15]
N_TRIALS = 100

SMOKE_MODEL = "google/gemma-3-4b-it"
SMOKE_TRIALS = 10

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_INSTRUCTION = (
    "You will be given a list of facts and a question. Answer with only a "
    "single word: the name. Do not explain. Do not include punctuation."
)
TEMPERATURE = 0.0
MAX_TOKENS = 16
REQUEST_TIMEOUT = 60
