# ruff: noqa
import inspect
from datetime import date
import polars as pl
from src.strategies.registry import STRATEGIES as BASELINES
from src.alpha.base import DecisionContext
from src.universe.tournament import TournamentRules
from pathlib import Path

def test_p16_registered_keeps_p14_alpha() -> None:
    assert "portfolio.convexity_hold" in BASELINES and "portfolio.lottery_exposure" in BASELINES and "portfolio.tail_concentration" in BASELINES
    p16 = BASELINES["portfolio.convexity_hold"]()
    p14 = BASELINES["portfolio.lottery_exposure"]()
    p15 = BASELINES["portfolio.tail_concentration"]()
    assert p16.name == "portfolio.convexity_hold"
    assert p14.name == "portfolio.lottery_exposure"
    assert p15.name == "portfolio.tail_concentration"
    assert p16.scores_path_independent is True
    assert getattr(p16, "convexity_config").enabled is True
    assert getattr(p16, "convexity_config").skip_capacity_route is True
    import src.portfolio.builders_convexity as baselines_mod
    src = inspect.getsource(baselines_mod.make_portfolio_convexity_hold)
    assert "ConvexityHoldConfig" in src
    assert "family_canonical_scores" in src
    assert "filter_scores_by_theme_state" not in src
    snap = pl.DataFrame({"ticker": ["069500", "122630"], "mom_20": [0.10, 0.22]})
    ctx = DecisionContext(decision_date=date(2024, 6, 1), regime=None, capital=1_000_000_000.0, held={}, rules=TournamentRules.from_yaml(Path("configs/tournament.yaml")))
    assert p16.score(snap, ctx) == p14.score(snap, ctx)
