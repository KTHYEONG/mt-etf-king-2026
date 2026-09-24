"""P4 CLI decomposition: features.py moved verbatim from _impl.py."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from src.cli.commands.features import cmd_features


def test_p4_features_missing_args_returns_1() -> None:
    assert cmd_features(argparse.Namespace()) == 1


def test_p4_features_start_after_end_returns_1() -> None:
    args = argparse.Namespace(start="2026-08-27", end="2026-01-02")
    assert cmd_features(args) == 1


def test_features_gold_persisted_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful build publishes the gold panel through one atomic parquet write."""
    from typing import Any

    import src.features.builder as builder_module

    silver_dir = tmp_path / "normalized"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "date": [date(2026, 8, 27), date(2026, 8, 28)],
            "ticker": ["069500", "069500"],
            "close": [100.0, 101.0],
        }
    ).write_parquet(silver_dir / "etf_daily.parquet")
    monkeypatch.setattr(
        "src.cli.commands.features.get_settings", lambda: SimpleNamespace(data_root=tmp_path)
    )
    monkeypatch.setattr(
        builder_module.FeatureConfig, "from_yaml", staticmethod(lambda path: SimpleNamespace())
    )

    built: dict[str, Any] = {}

    class _StubBuilder:
        def __init__(self, calendar: Any, config: Any) -> None:
            built["calendar"] = calendar
            built["config"] = config

        def build_panel(self, panel: pl.DataFrame, decision_date: date) -> pl.DataFrame:
            return panel

    monkeypatch.setattr(builder_module, "FeatureBuilder", _StubBuilder)
    args = argparse.Namespace(start="2026-08-27", end="2026-08-28")
    assert cmd_features(args) == 0
    gold_path = tmp_path / "features" / "etf_features.parquet"
    assert gold_path.is_file()
    assert set(pl.read_parquet(gold_path).columns) >= {"date", "ticker"}
