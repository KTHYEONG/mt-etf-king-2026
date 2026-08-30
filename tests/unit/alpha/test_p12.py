from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from src.alpha.base import DecisionContext
from src.alpha.baselines import BASELINES
from src.alpha.cluster import ClusterResolver
from src.alpha.leadership import (
    SectorLeadershipModel,
    SectorScoreWeights,
    filter_scores_by_theme_state,
)
from src.universe.tournament import TournamentRules
from tests.unit.alpha.conftest import make_attr, make_master, transition_config


def test_filter_scores_by_theme_state_tradable_only() -> None:
    scores = {"A": 1.0, "B": 0.5}
    theme_states = {"A": "LEADING", "B": "BREAKDOWN"}
    filtered = filter_scores_by_theme_state(scores, theme_states)
    assert filtered == {"A": 1.0}


def test_theme_states_by_representative_after_score() -> None:
    attrs = {"AAA": make_attr("AAA", theme="T1", index_key="idx_a")}
    master = make_master(attrs)
    resolver = ClusterResolver(master, max_per_theme=2)
    model = SectorLeadershipModel(
        master=master,
        resolver=resolver,
        weights=SectorScoreWeights(rs=0.45, accel=0.30, breadth=0.25),
        transition_config=transition_config(),
        history=None,
    )
    snapshot = pl.DataFrame(
        [
            {
                "ticker": "AAA",
                "mom_20": 0.20,
                "mom_5": 0.12,
                "close": 100.0,
                "ma_20": 90.0,
                "trading_value": 5_000_000_000.0,
            }
        ]
    )
    ctx = DecisionContext(
        decision_date=date(2024, 6, 1),
        regime=None,
        capital=1_000_000_000.0,
        held={},
        rules=TournamentRules.from_yaml(Path("configs/tournament.yaml")),
    )
    _ = model.score(snapshot, ctx)
    states = model.theme_states_by_representative()
    assert states
    assert "AAA" in states
    assert isinstance(states["AAA"], str) and states["AAA"]


def test_p12_in_baselines_registry() -> None:
    assert "P12" in BASELINES
    model = BASELINES["P12"]()
    assert getattr(model, "name", None) == "P12"
    assert hasattr(model, "score")
    assert hasattr(model, "allocate")
    assert callable(getattr(model, "theme_states_by_representative", None))
