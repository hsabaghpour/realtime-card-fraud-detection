"""Shared fixtures: a small local Spark session and isolated settings."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sentinelpay.utils.config import Settings, load_settings

TEST_SAMPLE_SIZE = 5_000


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with a tiny sample and all outputs under a temp directory."""
    return load_settings(
        env="local",
        overrides={
            "data": {"sample_size": TEST_SAMPLE_SIZE},
            "spark": {"master": "local[2]", "shuffle_partitions": 2},
            "paths": {
                "data_root": str(tmp_path / "data"),
                "reports_dir": str(tmp_path / "reports"),
            },
        },
    )


@pytest.fixture(scope="session")
def spark() -> Iterator:
    """One Delta-enabled local Spark session for the whole test run."""
    from sentinelpay.utils.spark import get_spark

    base = load_settings(
        env="local", overrides={"spark": {"master": "local[2]", "shuffle_partitions": 2}}
    )
    session = get_spark(base, app_name="sentinelpay-tests")
    yield session
    session.stop()
