"""Facade fidelity for the sticky model runner split (P5)."""

from __future__ import annotations


def test_model_runner_canonical_home() -> None:
    import src.strategies.sticky.model as facade
    from src.strategies.sticky.model_runner import StickyLeaderModel

    assert facade.StickyLeaderModel is StickyLeaderModel
    assert StickyLeaderModel.__module__ == "src.strategies.sticky.model_runner"


def test_score_crash_rebound_route_excludes_uncapacitated_candidates() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.sticky.model_config import StickyLeaderConfig
    from src.strategies.sticky.model_runner import StickyLeaderModel
    from src.tournament.objective.cutoff_auc import CUTOFF_AUC_IS_PRODUCTION_GATE

    assert CUTOFF_AUC_IS_PRODUCTION_GATE is True, "attack-sleeve routing must be the production path"

    capital = 1_000_000_000.0
    phi = 0.01
    # required ADV for min_fill_ratio=0.25 at sleeve_weight 0.95 => 1e9*0.95*0.25/0.01 = 23.75e9
    # Given: THIN wins on mom_20 but can never be filled; DEEP is the only tradable +2X
    snapshot = pl.DataFrame(
        {
            "ticker": ["THIN", "DEEP"],
            "name": ["SOL 소형주단일종목레버리지", "KODEX 코스닥150 레버리지"],
            "mom_60": [0.20, 0.10],
            "mom_20": [0.90, 0.30],
            "trading_value": [2_000_000_000.0, 500_000_000_000.0],
            "volume_expansion": [1.0, 1.0],
        }
    )
    model = StickyLeaderModel(
        name="sticky.mom60_raw",
        config=StickyLeaderConfig(mom_col="mom_60", min_fill_ratio=0.25),
    )
    ctx = DecisionContext(
        decision_date=date(2026, 9, 8),
        regime=None,
        capital=capital,
        held={},
        rules=SimpleNamespace(initial_capital=capital, max_order_to_adv=phi),  # type: ignore[arg-type]
        championship_sleeve="CRASH_REBOUND",
    )

    # When
    scores = model.score(snapshot, ctx)

    # Then: the untradable leader is gone; the route still returns the rebound ranking
    assert isinstance(scores, dict)
    assert "THIN" not in scores, "ADV 20억 종목은 용량필터를 통과할 수 없다"
    assert set(scores) == {"DEEP"}


def test_score_crash_rebound_route_returns_cash_when_no_capacitated_candidate() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.portfolio.intent import CASH_INTENT
    from src.strategies.sticky.model_config import StickyLeaderConfig
    from src.strategies.sticky.model_runner import StickyLeaderModel

    capital = 1_000_000_000.0
    # Given: DEEP is capacitated and keeps the mom60 route alive (mom_60 present) but is
    # absent from rebound_leader_scores because mom_20 is null; THIN is the only rebound
    # candidate and is far below the 23.75e9 ADV requirement.
    snapshot = pl.DataFrame(
        {
            "ticker": ["THIN", "DEEP"],
            "name": ["SOL 소형주단일종목레버리지", "KODEX 코스닥150 레버리지"],
            "mom_60": [0.20, 0.10],
            "mom_20": [0.90, None],
            "trading_value": [2_000_000_000.0, 500_000_000_000.0],
            "volume_expansion": [1.0, 1.0],
        }
    )
    model = StickyLeaderModel(
        name="sticky.mom60_raw",
        config=StickyLeaderConfig(mom_col="mom_60", min_fill_ratio=0.25),
    )
    ctx = DecisionContext(
        decision_date=date(2026, 9, 8),
        regime=None,
        capital=capital,
        held={},
        rules=SimpleNamespace(initial_capital=capital, max_order_to_adv=0.01),  # type: ignore[arg-type]
        championship_sleeve="CRASH_REBOUND",
    )

    # When
    result = model.score(snapshot, ctx)

    # Then: fail-closed to cash, never to an unfillable ticker
    assert result is CASH_INTENT
