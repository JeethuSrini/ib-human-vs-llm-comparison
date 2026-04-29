from features.analysis import aggregate_rows, wilson_interval


def test_wilson_interval_handles_empty_total():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_aggregate_rows_groups_by_source_model_mode_and_n():
    rows = [
        {
            "source": "model",
            "model": "m1",
            "mode": "separate",
            "n_examples": 3,
            "memory_correct": True,
            "reasoning_correct": False,
        },
        {
            "source": "model",
            "model": "m1",
            "mode": "separate",
            "n_examples": 3,
            "memory_correct": True,
            "reasoning_correct": True,
        },
    ]
    summary = aggregate_rows(rows)
    assert summary[0]["memory_accuracy"] == 1.0
    assert summary[0]["reasoning_accuracy"] == 0.5
    assert summary[0]["reasoning_given_memory"] == 0.5
