"""Prompt construction and trial sampling."""

import random


def group_names(entries: list[dict]) -> set[str]:
    """All person and city names used in a group's prompt."""
    names: set[str] = set()
    for a, b in entries[0]["facts"]:
        names.add(a)
        names.add(b)
    for city in entries[0]["attributes"].values():
        names.add(city)
    return names


def sample_trial_entries(
    groups: dict[int, list[dict]],
    n: int,
    rng: random.Random,
    max_attempts: int = 200,
) -> list[dict]:
    """Sample N entries, one from each of N distinct groups, with no name collisions."""
    group_ids = list(groups.keys())

    for _ in range(max_attempts):
        chosen_gids = rng.sample(group_ids, n)
        entries = [rng.choice(groups[gid]) for gid in chosen_gids]
        a_names = [e["target_fact"][0] for e in entries]
        b_names = [e["target_fact"][1] for e in entries]
        if len(set(a_names)) == n and len(set(b_names)) == n:
            return entries

    chosen_gids = rng.sample(group_ids, n)
    return [rng.choice(groups[gid]) for gid in chosen_gids]


def build_fact_block(entries: list[dict], rng: random.Random) -> str:
    """Build a shuffled fact block from N entries."""
    lines: list[str] = []
    for e in entries:
        a, b = e["target_fact"]
        city = e["reasoning_answer"]   # B's city
        lines.append(f"{a} is paired with {b}")
        lines.append(f"{b} lives in {city}")
    rng.shuffle(lines)
    return "Facts:\n" + "\n".join(lines)


def build_prompt(fact_block: str, question: str) -> str:
    return f"{fact_block}\n\n{question}"
