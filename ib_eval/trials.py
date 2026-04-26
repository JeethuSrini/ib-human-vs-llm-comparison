"""Trial runners and grid evaluation."""

import re
import time
import random
from typing import Any

import requests

from .client import OpenRouterAuthError, OpenRouterError, call_openrouter
from .config import SEED
from .prompts import build_fact_block, build_prompt, sample_trial_entries
from .scoring import build_vocab, extract_answer, score


def _make_question_result(
    model_id: str,
    mode: str,
    n_groups: int,
    entry: dict,
    mem_raw: str,
    rea_raw: str,
    mem_err: str | None,
    rea_err: str | None,
    vocab: set[str],
    elapsed: float,
    extra: dict | None = None,
) -> dict:
    """Build a single per-question log entry."""
    target_a, target_b = entry["target_fact"]
    mem_pred = extract_answer(mem_raw, vocab, exclude={target_a})
    rea_pred = extract_answer(rea_raw, vocab, exclude={target_a, target_b})
    result = {
        "model": model_id,
        "mode": mode,
        "n_groups": n_groups,
        "n_examples": n_groups,
        "group_id": entry["group_id"],
        "target_position": entry["target_position"],
        "target_fact": entry["target_fact"],
        "memory_truth": entry["memory_answer"],
        "reasoning_truth": entry["reasoning_answer"],
        "memory_prediction": mem_pred,
        "reasoning_prediction": rea_pred,
        "memory_correct": score(mem_pred, entry["memory_answer"]),
        "reasoning_correct": score(rea_pred, entry["reasoning_answer"]),
        "memory_raw": mem_raw,
        "reasoning_raw": rea_raw,
        "memory_error": mem_err,
        "reasoning_error": rea_err,
        "elapsed_seconds": round(elapsed, 3),
    }
    if extra:
        result.update(extra)
    return result


def run_trial(
    model_id: str,
    entries: list[dict],
    rng: random.Random,
    api_key: str,
    session: requests.Session,
    vocab: set[str],
) -> list[dict]:
    """Separate-call mode: 2 API calls per person (memory + reasoning).

    Builds one fact block from all N entries (2N lines total — one pairing
    and one attribute per person). Then asks memory and reasoning questions
    about every person. Score = correct / N.
    """
    fact_block = build_fact_block(entries, rng)
    n = len(entries)
    results = []

    for entry in entries:
        t0 = time.time()
        try:
            mem_raw = call_openrouter(
                model_id, build_prompt(fact_block, entry["memory_question"]),
                api_key, session,
            )
            mem_err = None
        except OpenRouterAuthError:
            raise
        except OpenRouterError as e:
            mem_raw = ""; mem_err = str(e)

        try:
            rea_raw = call_openrouter(
                model_id, build_prompt(fact_block, entry["reasoning_question"]),
                api_key, session,
            )
            rea_err = None
        except OpenRouterAuthError:
            raise
        except OpenRouterError as e:
            rea_raw = ""; rea_err = str(e)

        results.append(_make_question_result(
            model_id, "separate", n, entry,
            mem_raw, rea_raw, mem_err, rea_err, vocab,
            time.time() - t0,
        ))
    return results


def run_trial_combined(
    model_id: str,
    entries: list[dict],
    rng: random.Random,
    api_key: str,
    session: requests.Session,
    vocab: set[str],
) -> list[dict]:
    """Combined mode: both Qs for each person in one API call."""
    fact_block = build_fact_block(entries, rng)
    n = len(entries)
    results = []

    for entry in entries:
        combined_prompt = (
            f"{fact_block}\n\n"
            f"Answer both questions below with a single word each.\n\n"
            f"Q1: {entry['memory_question']}\n"
            f"Q2: {entry['reasoning_question']}\n\n"
            f"A1:\nA2:"
        )
        t0 = time.time()
        try:
            raw = call_openrouter(model_id, combined_prompt, api_key, session)
            err = None
        except OpenRouterAuthError:
            raise
        except OpenRouterError as e:
            raw = ""; err = str(e)

        mem_raw, rea_raw = "", ""
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        for line in lines:
            low = line.lower()
            if low.startswith("a1:") or low.startswith("1.") or low.startswith("1:"):
                mem_raw = re.sub(r"^(a1:|1\.|1:)\s*", "", line, flags=re.IGNORECASE).strip()
            elif low.startswith("a2:") or low.startswith("2.") or low.startswith("2:"):
                rea_raw = re.sub(r"^(a2:|2\.|2:)\s*", "", line, flags=re.IGNORECASE).strip()
        if not mem_raw and not rea_raw and len(lines) >= 2:
            mem_raw, rea_raw = lines[0], lines[1]
        elif not mem_raw and lines:
            mem_raw = lines[0]

        results.append(_make_question_result(
            model_id, "combined", n, entry,
            mem_raw, rea_raw, err, err, vocab,
            time.time() - t0, extra={"combined_raw": raw},
        ))
    return results


def run_trial_summarize(
    model_id: str,
    entries: list[dict],
    rng: random.Random,
    api_key: str,
    session: requests.Session,
    vocab: set[str],
) -> list[dict]:
    """Summarize-then-query: compress N facts into T, then query all N from T."""
    fact_block = build_fact_block(entries, rng)
    n = len(entries)

    summarize_prompt = (
        f"{fact_block}\n\n"
        f"You will later be asked two types of questions based on the facts above:\n"
        f"  1. Memory questions: direct recall, e.g. \"What is Alen paired with?\"\n"
        f"  2. Reasoning questions: multi-step, e.g. \"Where does the person paired with Alen live?\"\n\n"
        f"Summarize the facts above as concisely as possible while preserving ALL "
        f"pairing relationships and city assignments so both question types remain answerable.\n\n"
        f"Example of good summary format:\n"
        f"  Alen-Brik(Colvex), Cora-Deni(Arvon), Eron-Fila(Belso)\n"
        f"  meaning: Alen paired with Brik who lives in Colvex, etc.\n\n"
        f"Write only the summary, nothing else."
    )

    try:
        summary_raw = call_openrouter(
            model_id, summarize_prompt, api_key, session,
            max_tokens=512,
            system="You are a concise assistant. Follow the user's instructions exactly.",
        )
        summary_err = None
    except OpenRouterAuthError:
        raise
    except OpenRouterError as e:
        summary_raw = ""; summary_err = str(e)

    def query_from_summary(question: str) -> tuple[str, str | None]:
        if summary_err:
            return "", summary_err
        prompt = (
            f"Here is a summary of some facts:\n{summary_raw}\n\n"
            f"{question}\n"
            f"Answer with only a single word: the name."
        )
        try:
            return call_openrouter(model_id, prompt, api_key, session), None
        except OpenRouterAuthError:
            raise
        except OpenRouterError as e:
            return "", str(e)

    results = []
    for entry in entries:
        t0 = time.time()
        mem_raw, mem_err = query_from_summary(entry["memory_question"])
        rea_raw, rea_err = query_from_summary(entry["reasoning_question"])
        results.append(_make_question_result(
            model_id, "summarize", n, entry,
            mem_raw, rea_raw, mem_err, rea_err, vocab,
            time.time() - t0, extra={"summary": summary_raw},
        ))
    return results


def run_grid(
    models: list[str],
    sizes: list[int],
    n_trials: int,
    dataset: list[dict],
    groups: dict[int, list[dict]],
    api_key: str,
    combined: bool = False,
    summarize: bool = False,
) -> tuple[dict, list[dict]]:
    summary: dict[str, dict[str, Any]] = {}
    logs: list[dict] = []
    session = requests.Session()
    rng = random.Random(SEED)
    vocab = build_vocab(dataset)

    # Pre-sample trial entries so all models see identical prompts.
    # Each trial spec is a list of N entries (one per person in the prompt).
    trial_specs: dict[int, list[list[dict]]] = {}
    for n in sizes:
        specs = []
        for _ in range(n_trials):
            specs.append(sample_trial_entries(groups, n, rng))
        trial_specs[n] = specs

    trial_fn = run_trial_summarize if summarize else (
        run_trial_combined if combined else run_trial
    )

    for model_id in models:
        print(f"\n=== {model_id} ===", flush=True)
        summary[model_id] = {}
        for n in sizes:
            mem_correct = 0
            rea_correct = 0
            both_correct = 0
            recorded = 0
            t_start = time.time()

            for trial_idx, entries in enumerate(trial_specs[n]):
                trial_rng = random.Random(f"{SEED}|{model_id}|{n}|{trial_idx}")

                question_results = trial_fn(
                    model_id, entries,
                    trial_rng, api_key, session, vocab,
                )

                for qr in question_results:
                    qr["trial"] = trial_idx
                    logs.append(qr)
                    if qr["memory_error"] is None and qr["reasoning_error"] is None:
                        recorded += 1
                        if qr["memory_correct"]:
                            mem_correct += 1
                            if qr["reasoning_correct"]:
                                both_correct += 1
                        if qr["reasoning_correct"]:
                            rea_correct += 1

                if (trial_idx + 1) % 10 == 0 or trial_idx == n_trials - 1:
                    elapsed = time.time() - t_start
                    print(
                        f"  N={n:2d}  trial {trial_idx + 1:3d}/{n_trials}  "
                        f"mem={mem_correct}/{recorded}  "
                        f"rea={rea_correct}/{recorded}  "
                        f"({elapsed:.1f}s)",
                        flush=True,
                    )

            mem_acc = mem_correct / recorded if recorded else 0.0
            rea_acc = rea_correct / recorded if recorded else 0.0
            cond_acc = both_correct / mem_correct if mem_correct else 0.0
            summary[model_id][str(n)] = {
                "memory_accuracy": round(mem_acc, 4),
                "reasoning_accuracy": round(rea_acc, 4),
                "reasoning_given_memory": round(cond_acc, 4),
                "n_questions_recorded": recorded,
                "n_trials_attempted": n_trials,
            }

    return summary, logs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

