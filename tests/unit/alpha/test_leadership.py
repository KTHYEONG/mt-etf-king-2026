"""SCENARIO-07-06 SCENARIO-07-08"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from src.alpha.base import DecisionContext
from src.alpha.cluster import ClusterResolver
from src.alpha.leadership import SectorLeadershipModel, SectorScoreWeights
from src.universe.tournament import TournamentRules
from tests.unit.alpha.conftest import make_master, transition_config


def test_SCENARIO_07_06_empty_snapshot_returns_empty_score() -> None:  # noqa: N802
    """SCENARIO-07-06"""
    model = SectorLeadershipModel(
        master=make_master({}),
        resolver=ClusterResolver(make_master({})),
        weights=SectorScoreWeights(rs=0.45, accel=0.30, breadth=0.25),
        transition_config=transition_config(),
        history=None,
    )
    ctx = DecisionContext(
        decision_date=date(2024, 6, 1),
        regime=None,
        capital=1_000_000_000.0,
        held={},
        rules=TournamentRules.from_yaml(Path("configs/tournament.yaml")),
    )
    assert model.score(pl.DataFrame(), ctx) == {}


def test_SCENARIO_07_08_sector_score_weights_from_yaml() -> None:  # noqa: N802
    """SCENARIO-07-08"""
    weights = SectorScoreWeights.from_yaml(
        {"rs": 0.45, "accel": 0.30, "breadth": 0.25, "breakout": 0.0, "flow": 0.0}
    )
    assert abs(weights.rs + weights.accel + weights.breadth - 1.0) < 1e-9
    assert abs(weights.sector_score(1.0, 1.0, 1.0) - 1.0) < 1e-9
