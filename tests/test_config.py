from pathlib import Path

import pytest

from sentinelpay.utils.config import PROJECT_ROOT, load_settings


def test_base_and_local_layers_merge():
    s = load_settings(env="local")
    assert s.env == "local"
    assert s.spark.master == "local[4]"  # from local.yaml
    assert s.data.history_days == 182  # from base.yaml


def test_env_var_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAMPLE_SIZE", "1234")
    monkeypatch.setenv("SEED", "7")
    s = load_settings(env="local")
    assert s.data.sample_size == 1234
    assert s.seed == 7


def test_explicit_overrides_win_over_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAMPLE_SIZE", "1234")
    s = load_settings(env="local", overrides={"data": {"sample_size": 99}})
    assert s.data.sample_size == 99


def test_relative_paths_resolve_to_project_root():
    s = load_settings(env="local")
    assert Path(s.paths.data_root) == PROJECT_ROOT / "data"


def test_databricks_paths_are_volumes():
    s = load_settings(env="databricks")
    assert s.paths.bronze.startswith("/Volumes/")
