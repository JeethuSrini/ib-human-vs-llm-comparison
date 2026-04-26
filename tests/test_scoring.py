import random

import pytest

from ib_eval.prompts import build_fact_block, build_prompt, sample_trial_entries
from ib_eval.scoring import extract_answer, score


def test_score_case_and_whitespace():
    assert score("Veki", "veki")
    assert score("  Veki  ", "veki")
    assert not score("veko", "veki")


def test_extract_last_vocab_token():
    vocab = {"veki", "colvex", "jelo"}
    raw = "The answer is probably Veki"
    assert extract_answer(raw, vocab, exclude=set()) == "Veki"
    # verbose: last vocab hit
    assert extract_answer("Maybe Jelo but actually Veki", vocab, exclude=set()) == "Veki"


def test_extract_exclude():
    assert extract_answer("Jelo and Veki", {"veki", "jelo"}, exclude={"Jelo"}) == "Veki"


def test_build_fact_block_line_count():
    entries = [
        {
            "target_fact": ["A", "B"],
            "reasoning_answer": "City1",
        },
        {
            "target_fact": ["C", "D"],
            "reasoning_answer": "City2",
        },
    ]
    rng = random.Random(0)
    block = build_fact_block(entries, rng)
    lines = [l for l in block.splitlines() if l.strip() and not l.startswith("Facts:")]
    assert len(lines) == 4
    assert "A is paired with B" in block
    assert "B lives in City1" in block


def test_build_prompt_joins():
    assert build_prompt("Facts:\nA", "Q?") == "Facts:\nA\n\nQ?"


def test_sample_trial_determinism():
    # minimal fake groups: two group ids, one entry each
    g0 = {
        "group_id": 0,
        "target_position": 0,
        "target_fact": ("A0", "B0"),
        "memory_answer": "B0",
        "reasoning_answer": "C0",
    }
    g1 = {
        "group_id": 1,
        "target_position": 0,
        "target_fact": ("A1", "B1"),
        "memory_answer": "B1",
        "reasoning_answer": "C1",
    }
    groups = {0: [g0], 1: [g1]}
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    a = sample_trial_entries(groups, 2, rng1)
    b = sample_trial_entries(groups, 2, rng2)
    assert [x["group_id"] for x in a] == [x["group_id"] for x in b]
