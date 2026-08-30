# ruff: noqa
import inspect
from datetime import date
import polars as pl
from src.alpha.baselines import BASELINES
from src.alpha.base import DecisionContext
from src.universe.tournament import TournamentRules
from pathlib import Path

def test_p16_registered_keeps_p14_alpha() -> None:
    assert "P16" in BASELINES and "P14" in BASELINES and "P15" in BASELINES
    p16 = BASELINES["P16"]()
    p14 = BASELINES["P14"]()
    p15 = BASELINES["P15"]()
    assert p16.name == "P16"
    assert p14.name == "P14"
    assert p15.name == "P15"
    assert p16.scores_path_independent is True
    assert getattr(p16, "convexity_config").enabled is True
    assert getattr(p16, "convexity_config").skip_capacity_route is True
    import src.alpha.baselines as baselines_mod
    src = inspect.getsource(baselines_mod._make_p16)
    assert "ConvexityHoldConfig" in src
    assert "family_canonical_scores" in src
    assert "filter_scores_by_theme_state" not in src
    snap = pl.DataFrame({"ticker": ["069500", "122630"], "mom_20": [0.10, 0.22]})
    ctx = DecisionContext(decision_date=date(2024, 6, 1), regime=None, capital=1_000_000_000.0, held={}, rules=TournamentRules.from_yaml(Path("configs/tournament.yaml")))
    assert p16.score(snap, ctx) == p14.score(snap, ctx)
