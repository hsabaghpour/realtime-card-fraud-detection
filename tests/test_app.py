"""GUI smoke test: every page renders without exceptions against real pipeline output."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.spark

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def pipeline_outputs(spark, settings, monkeypatch):
    """Run the tiny pipeline and point the app at its outputs."""
    from sentinelpay.data import generator
    from sentinelpay.pipeline import bronze, silver

    generator.run(settings, None)
    bronze.run(spark, settings)
    silver.run(spark, settings)
    monkeypatch.setenv("SENTINELPAY_DATA_ROOT", settings.paths.data_root)
    monkeypatch.setenv("SENTINELPAY_REPORTS_DIR", settings.paths.reports_dir)


def test_data_explorer_renders(pipeline_outputs):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == "🔎 Data Explorer"
    assert len(at.tabs) == 4


def test_each_fraud_pattern_example_renders(pipeline_outputs):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    for option in at.radio(key="pattern").options:
        at.radio(key="pattern").set_value(option).run()
        assert not at.exception, f"{option}: {at.exception}"
        at.number_input(key="example").set_value(2).run()
        assert not at.exception, f"{option} example 2: {at.exception}"


def test_missing_outputs_shows_hint(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SENTINELPAY_DATA_ROOT", str(tmp_path / "nothing"))
    monkeypatch.setenv("SENTINELPAY_REPORTS_DIR", str(tmp_path / "nothing"))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert "make demo" in at.warning[0].value
