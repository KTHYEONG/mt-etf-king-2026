from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.core.paths import DataPaths


def test_bronze_silver_gold_state_and_guard(tmp_path: Path) -> None:
    """SCENARIO-01-05: DataPaths 매핑 및 traversal guard."""
    root = tmp_path / "data_root"
    dp = DataPaths(root=root)
    assert dp.bronze("etp/etf_bydd_trd", date(2026, 8, 27)) == root / "raw/krx/etp/etf_bydd_trd/2026/20260827.json.gz"
    assert dp.silver("etf_daily") == root / "normalized/etf_daily.parquet"
    assert dp.gold("features_etf") == root / "features/features_etf.parquet"
    assert dp.state("krx_quota") == root / "state/krx_quota.json"

    with pytest.raises(ValueError, match="path traversal not allowed"):
        dp.bronze("../../etc", date(2026, 8, 27))
    with pytest.raises(ValueError, match="absolute path not allowed"):
        dp.silver("/abs")
    with pytest.raises(ValueError, match="path traversal not allowed"):
        dp.silver("../escape")
    with pytest.raises(ValueError, match="absolute path not allowed"):
        dp.gold("/abs/table")

    # No directory is created by these calls
    assert not (root / "raw").exists()
    assert not (root / "normalized").exists()
    assert not (root / "features").exists()
    assert not (root / "state").exists()


def test_SCENARIO_DSR_05_bronze_gz_and_results(tmp_path: Path) -> None:  # noqa: N802
    """SCENARIO_DSR_05: DataPaths.bronze ends with .json.gz; results guard."""
    root = tmp_path / "data_root"
    dp = DataPaths(root=root)
    p = dp.bronze("etp/etf_bydd_trd", date(2026, 8, 27))
    assert p.name.endswith(".json.gz")
    assert p.suffixes == [".json", ".gz"] or p.name.endswith(".json.gz")
    assert p == root / "raw/krx/etp/etf_bydd_trd/2026/20260827.json.gz"
    # results
    assert dp.results("run_a") == root / "docs/results" / "run_a"
    with pytest.raises(ValueError, match="path traversal not allowed"):
        dp.results("..")
    with pytest.raises(ValueError, match="absolute path not allowed"):
        dp.results("/abs")
    with pytest.raises(ValueError, match="path traversal not allowed"):
        dp.results("../escape")


def test_paths_trace_under_results_and_guard(tmp_path: Path) -> None:
    dp = DataPaths(root=tmp_path)
    assert dp.trace("run_a") == tmp_path / "docs/results" / "run_a" / "trace"
    assert not (tmp_path / "docs/results").exists()
    with pytest.raises(ValueError, match="path traversal"):
        dp.trace("..")
    with pytest.raises(ValueError, match="path traversal"):
        dp.trace("../escape")
    with pytest.raises(ValueError, match="absolute path"):
        dp.trace("/abs")
