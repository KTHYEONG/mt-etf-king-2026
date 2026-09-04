from __future__ import annotations


def test_sticky_model_caches_snapshot_filtering() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.sticky.model import StickyLeaderConfig, StickyLeaderModel

    model = StickyLeaderModel(config=StickyLeaderConfig())
    snapshot = pl.DataFrame(
        {"ticker": ["A"], "name": ["KODEX 반도체레버리지"], "mom_60": [0.2]}
    )
    context = DecisionContext(decision_date=date(2026, 1, 2), regime=None, capital=1.0, held={}, rules=None)
    first = model.score(snapshot, context)
    second = model.score(snapshot, context)
    assert type(first) is type(second)
    assert len(model._filtered_scores_by_snapshot) == 1
