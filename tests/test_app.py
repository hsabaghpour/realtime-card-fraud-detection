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
    from sentinelpay.features import spark_features
    from sentinelpay.pipeline import bronze, silver
    from sentinelpay.training import promote, train

    generator.run(settings, None)
    bronze.run(spark, settings)
    silver.run(spark, settings)
    spark_features.run(spark, settings)
    train.run(settings)
    promote.run(settings)
    monkeypatch.setenv("SENTINELPAY_DATA_ROOT", settings.paths.data_root)
    monkeypatch.setenv("SENTINELPAY_REPORTS_DIR", settings.paths.reports_dir)
    monkeypatch.setenv("SENTINELPAY_MLFLOW_URI", settings.mlflow.tracking_uri)


def test_data_explorer_renders(pipeline_outputs):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == "🔎 Data Explorer"
    assert len(at.tabs) == 5


def test_every_feature_chart_renders(pipeline_outputs):
    """Each of the 26 features (flags, counts, deciles, categories) must chart cleanly."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    for option in at.selectbox(key="feature").options:
        at.selectbox(key="feature").set_value(option).run()
        assert not at.exception, f"{option}: {at.exception}"


def test_each_fraud_pattern_example_renders(pipeline_outputs):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    for option in at.radio(key="pattern").options:
        at.radio(key="pattern").set_value(option).run()
        assert not at.exception, f"{option}: {at.exception}"
        at.number_input(key="example").set_value(2).run()
        assert not at.exception, f"{option} example 2: {at.exception}"


def test_model_performance_page_renders(pipeline_outputs):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.switch_page("views/model_performance.py").run()
    assert not at.exception, at.exception
    assert at.title[0].value == "📈 Model Performance"
    assert any("champion" in i.value.lower() for i in at.info)


def test_threshold_page_reacts_to_slider_and_costs(pipeline_outputs):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.switch_page("views/threshold.py").run()
    assert not at.exception, at.exception

    def alerts_per_day() -> str:
        return next(m.value for m in at.metric if m.label == "Alerts per day")

    at.select_slider(key="threshold").set_value(0.001).run()  # lowest threshold -> many alerts
    low = float(alerts_per_day().replace(",", ""))
    at.select_slider(key="threshold").set_value(0.99).run()  # highest -> few
    high = float(alerts_per_day().replace(",", ""))
    assert low > high

    at.button[0].click().run()  # "Jump to cost-optimal"
    assert not at.exception
    at.number_input[1].set_value(500.0).run()  # very expensive reviews
    assert not at.exception


def test_missing_outputs_shows_hint(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SENTINELPAY_DATA_ROOT", str(tmp_path / "nothing"))
    monkeypatch.setenv("SENTINELPAY_REPORTS_DIR", str(tmp_path / "nothing"))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert "make demo" in at.warning[0].value
