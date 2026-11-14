from pathlib import Path

import pytest

from sentinelpay import cli


def test_parser_lists_all_steps():
    parser = cli.build_parser()
    for cmd in ("generate", "bronze", "silver", "demo"):
        assert parser.parse_args([cmd]).command == cmd


def test_rejects_unknown_drift_scenario():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["generate", "--drift-scenario", "bogus"])


def test_generate_command_writes_landing(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINELPAY_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("SENTINELPAY_REPORTS_DIR", str(tmp_path / "reports"))
    assert cli.main(["--sample-size", "3000", "generate", "--drift-scenario", "none"]) == 0
    assert list((tmp_path / "data/landing/history").glob("*_n3000.parquet"))
    assert list((tmp_path / "data/landing/production/scenario=none").glob("*_n3000.parquet"))
    assert not Path(tmp_path / "data/landing/production/scenario=spending_shift").exists()
