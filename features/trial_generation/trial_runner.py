"""Model trial runners and grid evaluation."""

import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from tqdm import tqdm

# Concurrency within a single trial (memory+reasoning calls fired together).
MAX_WORKERS = 3

# Concurrency across trials (multiple trials running simultaneously).
# Each trial itself uses up to MAX_WORKERS threads, so total concurrent
# API calls = TRIAL_WORKERS * MAX_WORKERS. Keep total ≤ 30 to avoid 429s.
TRIAL_WORKERS = 1

from features.model_inference import (
    OpenRouterAuthError,
    OpenRouterError,
    OpenRouterRateLimitError,
    call_openrouter,
)
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
    """Separate-call mode: memory and reasoning calls fired concurrently per trial."""
    fact_block = build_fact_block(entries, rng)
    n_examples = len(entries)

    # Build all (entry_idx, question_type, prompt) tasks
    tasks = []
    for i, entry in enumerate(entries):
        tasks.append((i, "mem", build_prompt(fact_block, entry["memory_question"])))
        tasks.append((i, "rea", build_prompt(fact_block, entry["reasoning_question"])))

    raw: dict[tuple, tuple[str, str | None]] = {}

    def _call(idx: int, qtype: str, prompt: str) -> tuple[int, str, str, str | None]:
        try:
            result = call_openrouter(model_id, prompt, api_key, session)
            return idx, qtype, result, None
        except (OpenRouterAuthError, OpenRouterRateLimitError):
            raise
        except OpenRouterError as exc:
            return idx, qtype, "", str(exc)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_call, i, qt, p): (i, qt) for i, qt, p in tasks}
        for fut in as_completed(futures):
            idx, qtype, result, err = fut.result()
            raw[(idx, qtype)] = (result, err)

    elapsed = time.time() - t0
    results = []
    for i, entry in enumerate(entries):
        mem_raw, mem_err = raw.get((i, "mem"), ("", "missing"))
        rea_raw, rea_err = raw.get((i, "rea"), ("", "missing"))
        results.append(_make_question_result(
            model_id, "separate", n_examples, entry,
            mem_raw, rea_raw, mem_err, rea_err, vocab,
            round(elapsed / len(entries), 3),
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
    """Combined mode: all N combined-question calls fired concurrently."""
    fact_block = build_fact_block(entries, rng)
    n_examples = len(entries)

    prompts = [
        (i, (
            f"{fact_block}\n\n"
            f"Answer both questions below with a single word each.\n\n"
            f"Q1: {entry['memory_question']}\n"
            f"Q2: {entry['reasoning_question']}\n\n"
            f"A1:\nA2:"
        ))
        for i, entry in enumerate(entries)
    ]

    raw_map: dict[int, tuple[str, str | None]] = {}

    def _call(idx: int, prompt: str) -> tuple[int, str, str | None]:
        try:
            return idx, call_openrouter(model_id, prompt, api_key, session), None
        except (OpenRouterAuthError, OpenRouterRateLimitError):
            raise
        except OpenRouterError as exc:
            return idx, "", str(exc)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_call, i, p): i for i, p in prompts}
        for fut in as_completed(futures):
            idx, result, err = fut.result()
            raw_map[idx] = (result, err)

    elapsed = time.time() - t0
    results = []
    for i, entry in enumerate(entries):
        combined_raw, err = raw_map.get(i, ("", "missing"))
        mem_raw, rea_raw = _split_combined_response(combined_raw)
        results.append(_make_question_result(
            model_id, "combined", n_examples, entry,
            mem_raw, rea_raw, err, err, vocab,
            round(elapsed / len(entries), 3), extra={"combined_raw": combined_raw},
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

    # After the summary call, fire all 2N query calls concurrently
    tasks = []
    for i, entry in enumerate(entries):
        for qtype, question in [("mem", entry["memory_question"]),
                                 ("rea", entry["reasoning_question"])]:
            if summary_err:
                tasks.append((i, qtype, None))
            else:
                prompt = (
                    f"Here is a summary of some facts:\n{summary_raw}\n\n"
                    f"{question}\n"
                    "Answer with only a single word: the name."
                )
                tasks.append((i, qtype, prompt))

    raw_map: dict[tuple, tuple[str, str | None]] = {}

    def _query(idx: int, qtype: str, prompt: str | None) -> tuple[int, str, str, str | None]:
        if prompt is None:
            return idx, qtype, "", summary_err
        try:
            return idx, qtype, call_openrouter(model_id, prompt, api_key, session), None
        except (OpenRouterAuthError, OpenRouterRateLimitError):
            raise
        except OpenRouterError as exc:
            return idx, qtype, "", str(exc)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_query, i, qt, p): (i, qt) for i, qt, p in tasks}
        for fut in as_completed(futures):
            idx, qtype, result, err = fut.result()
            raw_map[(idx, qtype)] = (result, err)

    elapsed = time.time() - t0
    results = []
    for i, entry in enumerate(entries):
        mem_raw, mem_err = raw_map.get((i, "mem"), ("", "missing"))
        rea_raw, rea_err = raw_map.get((i, "rea"), ("", "missing"))
        results.append(_make_question_result(
            model_id, "summarize", n_examples, entry,
            mem_raw, rea_raw, mem_err, rea_err, vocab,
            round(elapsed / len(entries), 3), extra={"summary": summary_raw},
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


def _load_checkpoint(checkpoint_path: "Path") -> tuple[list[dict], set[tuple]]:
    """Load completed rows from a checkpoint file.

    Returns (rows, done_keys) where done_keys is a set of
    (model, n_examples, trial_idx) tuples already completed.
    """
    import json
    from pathlib import Path

    cp = Path(checkpoint_path)
    if not cp.exists():
        return [], set()
    rows = []
    with cp.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    done_keys: set[tuple] = set()
    for row in rows:
        done_keys.add((row["model"], row["n_examples"], row["trial"]))
    return rows, done_keys


def _append_checkpoint(checkpoint_path: "Path", rows: list[dict]) -> None:
    """Append new rows to the checkpoint file (one JSON line each)."""
    import json
    from pathlib import Path

    cp = Path(checkpoint_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_grid(
    models: list[str],
    sizes: list[int],
    n_trials: int,
    dataset: list[dict],
    groups: dict[int, list[dict]],
    api_key: str,
    combined: bool = False,
    summarize: bool = False,
    checkpoint_path: "Path | None" = None,
) -> tuple[dict, list[dict]]:
    """Run every selected model over each size and trial condition.

    Supports resuming from a crash: if checkpoint_path exists, already-completed
    (model, n_examples, trial) combos are skipped automatically.
    New results are appended to the checkpoint after each completed trial.
    """
    summary: dict[str, dict[str, Any]] = {}
    session = requests.Session()
    vocab = build_vocab(dataset)
    trial_specs = make_trial_specs(groups, sizes, n_trials, seed=SEED)
    trial_fn = run_trial_summarize if summarize else (run_trial_combined if combined else run_trial)

    # Load any previously completed rows
    prior_rows, done_keys = (
        _load_checkpoint(checkpoint_path) if checkpoint_path else ([], set())
    )
    if done_keys:
        print(
            f"[resume] Found {len(prior_rows)} completed rows across "
            f"{len(done_keys)} trials in checkpoint — skipping those.",
            flush=True,
        )
    logs: list[dict] = list(prior_rows)

    total_trials = len(models) * len(sizes) * n_trials
    completed_trials = len(done_keys)
    all_mem_correct = sum(1 for r in prior_rows if r.get("memory_correct"))
    all_rea_correct = sum(1 for r in prior_rows if r.get("reasoning_correct"))
    all_recorded = len([r for r in prior_rows if r["memory_error"] is None and r["reasoning_error"] is None])

    bar = tqdm(
        total=total_trials,
        initial=completed_trials,
        desc="Overall",
        unit="trial",
        dynamic_ncols=True,
        colour="green",
    )

    import threading
    checkpoint_lock = threading.Lock()

    def _run_one_trial(
        model_id: str, n_examples: int, trial_idx: int, entries: list[dict]
    ) -> list[dict]:
        trial_rng = fact_order_rng(SEED, n_examples, trial_idx)
        results = trial_fn(model_id, entries, trial_rng, api_key, session, vocab)
        for row in results:
            row["trial"] = trial_idx
        return results

    for model_id in models:
        summary[model_id] = {}
        for n_examples in sizes:
            condition_rows: list[dict] = [
                r for r in prior_rows
                if r["model"] == model_id and r["n_examples"] == n_examples
            ]

            pending = [
                (trial_idx, entries)
                for trial_idx, entries in enumerate(trial_specs[n_examples])
                if (model_id, n_examples, trial_idx) not in done_keys
            ]

            with ThreadPoolExecutor(max_workers=TRIAL_WORKERS) as pool:
                future_map = {
                    pool.submit(_run_one_trial, model_id, n_examples, t_idx, entries): t_idx
                    for t_idx, entries in pending
                }

                for fut in as_completed(future_map):
                    try:
                        question_results = fut.result()
                    except (OpenRouterAuthError, OpenRouterRateLimitError):
                        bar.close()
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise
                    except OpenRouterError as exc:
                        bar.set_postfix_str(f"skip: {exc}"[:50])
                        bar.update(1)
                        continue

                    with checkpoint_lock:
                        for row in question_results:
                            logs.append(row)
                            condition_rows.append(row)
                            if row["memory_error"] is None and row["reasoning_error"] is None:
                                all_recorded += 1
                                if row["memory_correct"]:
                                    all_mem_correct += 1
                                if row["reasoning_correct"]:
                                    all_rea_correct += 1

                        if checkpoint_path:
                            _append_checkpoint(checkpoint_path, question_results)

                    model_short = model_id.split("/")[-1]
                    bar.set_postfix(
                        model=model_short,
                        N=n_examples,
                        mem=f"{all_mem_correct/max(all_recorded,1):.0%}",
                        rea=f"{all_rea_correct/max(all_recorded,1):.0%}",
                    )
                    bar.update(1)

            summary[model_id][str(n_examples)] = summarize_question_results(
                condition_rows, n_trials
            )

    bar.close()

    return summary, logs
