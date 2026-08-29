"""SCENARIO-07-03"""
from __future__ import annotations

from datetime import date

import polars as pl

from src.alpha.cluster import ClusterChoice
from src.alpha.leadership import SectorScoreWeights
from src.alpha.theme import build_theme_panel


def test_SCENARIO_07_03_build_theme_panel_rs_percentile() -> None:  # noqa: N802
    """SCENARIO-07-03"""
    choices = [
        ClusterChoice(ticker="A", index_key="idx_a", theme="T1"),
        ClusterChoice(ticker="B", index_key="idx_b", theme="T2"),
    ]
    snapshot = pl.DataFrame(
        [
            {
                "ticker": "A",
                "mom_20": 0.10,
                "mom_5": 0.12,
                "close": 100.0,
                "ma_20": 90.0,
            },
            {
                "ticker": "B",
                "mom_20": 0.20,
                "mom_5": 0.22,
                "close": 100.0,
                "ma_20": 90.0,
            },
        ]
    )
    weights = SectorScoreWeights(rs=0.45, accel=0.30, breadth=0.25)
    panel = build_theme_panel(snapshot, snapshot, choices, date(2024, 6, 1), weights)
    metrics = panel.metrics_for("T2")
    assert metrics is not None
    assert 0.0 <= metrics.rs <= 1.0
    assert metrics.rs == 1.0
