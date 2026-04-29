"""Model trial runners and grid evaluation."""

import random
import re
import time
from typing import Any

import requests

from features.model_inference import OpenRouterAuthError, OpenRouterError, call_openrouter
from features.scoring import build_vocab, extract_answer, score
from features.trial_generation.prompt_builder import build_fact_block, build_prompt
from features.trial_generation.trial_sampler import fact_order_rng, make_trial_specs
from shared.config import SEED


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
    """Build a single normalized per-question log entry."""
    target_a, target_b = entry["target_fact"]
    mem_pred = extract_answer(mem_raw, vocab, exclude={target_a})
    rea_pred = extract_answer(rea_raw, vocab, exclude={target_a, target_b})
    result = {
        "source": "model",
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
    """Separate-call mode: two API calls per person."""
    fact_block = build_fact_block(entries, rng)
    n_examples = len(entries)
    results = []

    for entry in entries:
        start = time.time()
        try:
            mem_raw = call_openrouter(
                model_id,
                build_prompt(fact_block, entry["memory_question"]),
                api_key,
                session,
            )
            mem_err = None
        except OpenRouterAuthError:
            raise
        except OpenRouterError as exc:
            mem_raw = ""
            mem_err = str(exc)

        try:
            rea_raw = call_openrouter(
                model_id,
                build_prompt(fact_block, entry["reasoning_question"]),
                api_key,
                session,
            )
            rea_err = None
        except OpenRouterAuthError:
            raise
        except OpenRouterError as exc:
            rea_raw = ""
            rea_err = str(exc)

        results.append(_make_question_result(
            model_id, "separate", n_examples, entry,
            mem_raw, rea_raw, mem_err, rea_err, vocab,
            time.time() - start,
        ))
    return results


def _split_combined_response(raw: str) -> tuple[str, str]:
    mem_raw, rea_raw = "", ""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    for line in lines:
        lower_line = line.lower()
        if lower_line.startswith(("a1:", "1.", "1:")):
            mem_raw = re.sub(r"^(a1:|1\.|1:)\s*", "", line, flags=re.IGNORECASE).strip()
        elif lower_line.startswith(("a2:", "2.", "2:")):
            rea_raw = re.sub(r"^(a2:|2\.|2:)\s*", "", line, flags=re.IGNORECASE).strip()
    if not mem_raw and not rea_raw and len(lines) >= 2:
        return lines[0], lines[1]
    if not mem_raw and lines:
        mem_raw = lines[0]
    return mem_raw, rea_raw


def run_trial_combined(
    model_id: str,
    entries: list[dict],
    rng: random.Random,
    api_key: str,
    session: requests.Session,
    vocab: set[str],
) -> list[dict]:
    """Combined mode: ask memory and reasoning questions in one API call."""
    fact_block = build_fact_block(entries, rng)
    n_examples = len(entries)
    results = []

    for entry in entries:
        combined_prompt = (
            f"{fact_block}\n\n"
            f"Answer both questions below with a single word each.\n\n"
            f"Q1: {entry['memory_question']}\n"
            f"Q2: {entry['reasoning_question']}\n\n"
            f"A1:\nA2:"
        )
        start = time.time()
        try:
            raw = call_openrouter(model_id, combined_prompt, api_key, session)
            err = None
        except OpenRouterAuthError:
            raise
        except OpenRouterError as exc:
            raw = ""
            err = str(exc)

        mem_raw, rea_raw = _split_combined_response(raw)
        results.append(_make_question_result(
            model_id, "combined", n_examples, entry,
            mem_raw, rea_raw, err, err, vocab,
            time.time() - start, extra={"combined_raw": raw},
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
    """Summarize-then-query mode: compress facts, then answer from summary."""
    fact_block = build_fact_block(entries, rng)
    n_examples = len(entries)
    summarize_prompt = (
        f"{fact_block}\n\n"
        "You will later be asked two types of questions based on the facts above:\n"
        "  1. Memory questions: direct recall, e.g. \"What is Alen paired with?\"\n"
        "  2. Reasoning questions: multi-step, e.g. \"Where does the person paired with Alen live?\"\n\n"
        "Summarize the facts above as concisely as possible while preserving ALL "
        "pairing relationships and city assignments so both question types remain answerable.\n\n"
        "Example of good summary format:\n"
        "  Alen-Brik(Colvex), Cora-Deni(Arvon), Eron-Fila(Belso)\n"
        "  meaning: Alen paired with Brik who lives in Colvex, etc.\n\n"
        "Write only the summary, nothing else."
    )

    try:
        summary_raw = call_openrouter(
            model_id,
            summarize_prompt,
            api_key,
            session,
            max_tokens=512,
            system="You are a concise assistant. Follow the user's instructions exactly.",
        )
        summary_err = None
    except OpenRouterAuthError:
        raise
    except OpenRouterError as exc:
        summary_raw = ""
        summary_err = str(exc)

    def query_from_summary(question: str) -> tuple[str, str | None]:
        if summary_err:
            return "", summary_err
        prompt = (
            f"Here is a summary of some facts:\n{summary_raw}\n\n"
            f"{question}\n"
            "Answer with only a single word: the name."
        )
        try:
            return call_openrouter(model_id, prompt, api_key, session), None
        except OpenRouterAuthError:
            raise
        except OpenRouterError as exc:
            return "", str(exc)

    results = []
    for entry in entries:
        start = time.time()
        mem_raw, mem_err = query_from_summary(entry["memory_question"])
        rea_raw, rea_err = query_from_summary(entry["reasoning_question"])
        results.append(_make_question_result(
            model_id, "summarize", n_examples, entry,
            mem_raw, rea_raw, mem_err, rea_err, vocab,
            time.time() - start, extra={"summary": summary_raw},
        ))
    return results


def summarize_question_results(question_results: list[dict], n_trials: int) -> dict[str, float | int]:
    """Aggregate normalized question rows for one model and N condition."""
    recorded_rows = [
        row for row in question_results
        if row["memory_error"] is None and row["reasoning_error"] is None
    ]
    recorded = len(recorded_rows)
    mem_correct = sum(1 for row in recorded_rows if row["memory_correct"])
    rea_correct = sum(1 for row in recorded_rows if row["reasoning_correct"])
    both_correct = sum(
        1 for row in recorded_rows
        if row["memory_correct"] and row["reasoning_correct"]
    )
    mem_acc = mem_correct / recorded if recorded else 0.0
    rea_acc = rea_correct / recorded if recorded else 0.0
    cond_acc = both_correct / mem_correct if mem_correct else 0.0
    return {
        "memory_accuracy": round(mem_acc, 4),
        "reasoning_accuracy": round(rea_acc, 4),
        "reasoning_given_memory": round(cond_acc, 4),
        "n_questions_recorded": recorded,
        "n_trials_attempted": n_trials,
    }


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
    """Run every selected model over each size and trial condition."""
    summary: dict[str, dict[str, Any]] = {}
    logs: list[dict] = []
    session = requests.Session()
    vocab = build_vocab(dataset)
    trial_specs = make_trial_specs(groups, sizes, n_trials, seed=SEED)
    trial_fn = run_trial_summarize if summarize else (run_trial_combined if combined else run_trial)

    for model_id in models:
        print(f"\n=== {model_id} ===", flush=True)
        summary[model_id] = {}
        for n_examples in sizes:
            condition_rows: list[dict] = []
            start = time.time()
            for trial_idx, entries in enumerate(trial_specs[n_examples]):
                trial_rng = fact_order_rng(SEED, n_examples, trial_idx)
                question_results = trial_fn(model_id, entries, trial_rng, api_key, session, vocab)
                for row in question_results:
                    row["trial"] = trial_idx
                    logs.append(row)
                    condition_rows.append(row)

                if (trial_idx + 1) % 10 == 0 or trial_idx == n_trials - 1:
                    partial = summarize_question_results(condition_rows, n_trials)
                    recorded = int(partial["n_questions_recorded"])
                    mem_count = sum(1 for row in condition_rows if row.get("memory_correct"))
                    rea_count = sum(1 for row in condition_rows if row.get("reasoning_correct"))
                    print(
                        f"  N={n_examples:2d}  trial {trial_idx + 1:3d}/{n_trials}  "
                        f"mem={mem_count}/{recorded}  rea={rea_count}/{recorded}  "
                        f"({time.time() - start:.1f}s)",
                        flush=True,
                    )
            summary[model_id][str(n_examples)] = summarize_question_results(condition_rows, n_trials)

    return summary, logs
