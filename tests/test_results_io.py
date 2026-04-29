from datetime import datetime, timezone

from features.results_io import build_model_run_dir, make_run_id, read_jsonl, write_jsonl


def test_make_run_id_is_sortable_utc_timestamp():
    stamp = datetime(2026, 4, 29, 7, 30, 0, tzinfo=timezone.utc)
    assert make_run_id(stamp) == "20260429T073000Z"


def test_build_model_run_dir_categorizes_outputs(tmp_path):
    path = build_model_run_dir(tmp_path, "smoke", "separate", run_id="r1")
    assert path == tmp_path / "model_runs" / "smoke" / "separate" / "r1"


def test_read_write_compressed_jsonl(tmp_path):
    path = tmp_path / "logs.jsonl.gz"
    rows = [{"a": 1}, {"b": 2}]
    write_jsonl(rows, path)
    assert read_jsonl(path) == rows
