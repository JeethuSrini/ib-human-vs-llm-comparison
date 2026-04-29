import random

from features.scoring import extract_answer, score
from features.trial_generation import build_fact_block, build_prompt, sample_trial_entries


def test_score_case_and_whitespace():
    assert score("Veki", "veki")
    assert score("  Veki  ", "veki")
    assert not score("veko", "veki")


def test_extract_last_vocab_token():
    vocab = {"veki", "colvex", "jelo"}
    raw = "The answer is probably Veki"
    assert extract_answer(raw, vocab, exclude=set()) == "Veki"
    assert extract_answer("Maybe Jelo but actually Veki", vocab, exclude=set()) == "Veki"


def test_extract_exclude():
    assert extract_answer("Jelo and Veki", {"veki", "jelo"}, exclude={"Jelo"}) == "Veki"


def test_build_fact_block_line_count():
    entries = [
        {"target_fact": ["A", "B"], "reasoning_answer": "City1"},
        {"target_fact": ["C", "D"], "reasoning_answer": "City2"},
    ]
    block = build_fact_block(entries, random.Random(0))
    lines = [line for line in block.splitlines() if line.strip() and not line.startswith("Facts:")]
    assert len(lines) == 4
    assert "A is paired with B" in block
    assert "B lives in City1" in block


def test_build_prompt_joins():
    assert build_prompt("Facts:\nA", "Q?") == "Facts:\nA\n\nQ?"


def test_sample_trial_determinism():
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
    first = sample_trial_entries(groups, 2, random.Random(42))
    second = sample_trial_entries(groups, 2, random.Random(42))
    assert [entry["group_id"] for entry in first] == [entry["group_id"] for entry in second]
