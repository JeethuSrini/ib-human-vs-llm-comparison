from features.human_study import build_trial_manifest, normalize_human_row


def test_build_trial_manifest_is_reproducible():
    entry = {
        "group_id": 0,
        "target_position": 0,
        "target_fact": ["A", "B"],
        "memory_question": "What is A paired with?",
        "memory_answer": "B",
        "reasoning_question": "Where does the person paired with A live?",
        "reasoning_answer": "City",
    }
    groups = {0: [entry]}
    first = build_trial_manifest(groups, [1], 1, seed=123)
    second = build_trial_manifest(groups, [1], 1, seed=123)
    assert first == second
    assert first[0]["trial_id"] == "n1-t0"
    assert first[0]["questions"][0]["memory_answer"] == "B"


def test_normalize_human_row_scores_missing_correctness():
    row = normalize_human_row({
        "participant_id": "p1",
        "trial": "0",
        "n_examples": "1",
        "group_id": "0",
        "target_position": "0",
        "memory_prediction": " B ",
        "reasoning_prediction": "city",
        "memory_truth": "B",
        "reasoning_truth": "City",
    })
    assert row["source"] == "human"
    assert row["memory_correct"] is True
    assert row["reasoning_correct"] is True
