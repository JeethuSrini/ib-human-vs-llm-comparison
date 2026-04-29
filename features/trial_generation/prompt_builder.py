"""Prompt and fact-block construction for IB trials."""

import random


def group_names(entries: list[dict]) -> set[str]:
    """Return all person and city names used in one prompt group."""
    names: set[str] = set()
    for left_name, right_name in entries[0]["facts"]:
        names.add(left_name)
        names.add(right_name)
    for city in entries[0]["attributes"].values():
        names.add(city)
    return names


def build_fact_block(entries: list[dict], rng: random.Random) -> str:
    """Build a shuffled fact block from N sampled entries."""
    lines: list[str] = []
    for entry in entries:
        left_name, right_name = entry["target_fact"]
        city = entry["reasoning_answer"]
        lines.append(f"{left_name} is paired with {right_name}")
        lines.append(f"{right_name} lives in {city}")
    rng.shuffle(lines)
    return "Facts:\n" + "\n".join(lines)


def build_prompt(fact_block: str, question: str) -> str:
    """Join the fact block and one question exactly as shown to a model."""
    return f"{fact_block}\n\n{question}"
