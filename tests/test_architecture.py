from pathlib import Path


def test_feature_modules_do_not_import_removed_compatibility_package():
    for path in Path("features").rglob("*.py"):
        source = path.read_text()
        assert "from ib_eval" not in source
        assert "import ib_eval" not in source


def test_project_uses_single_main_entrypoint():
    assert Path("main.py").exists()
    assert not Path("run_experiment.py").exists()
    assert not Path("generate_dataset.py").exists()
    assert not Path("ib_eval").exists()


def test_feature_modules_have_descriptive_file_names():
    generic_names = {"utils.py", "helpers.py", "common.py"}
    feature_files = {path.name for path in Path("features").rglob("*.py")}
    assert feature_files.isdisjoint(generic_names)


def test_results_folder_is_generated_not_checked_in():
    gitignore = Path(".gitignore").read_text()
    assert "results/" in gitignore
