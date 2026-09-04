from __future__ import annotations


def test_purged_date_walk_forward_excludes_label_and_embargo_dates() -> None:
    from datetime import date, timedelta
    from src.alpha.champion_ranker import PurgedDateWalkForward

    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(360)]
    splitter = PurgedDateWalkForward(n_folds=3, label_horizon=36, embargo=36, min_train_sessions=72)
    folds = splitter.split(dates)

    assert len(folds) == 3
    for fold in folds:
        assert fold.train_dates
        assert fold.test_dates
        assert fold.train_dates[-1] < fold.test_dates[0]
        assert dates.index(fold.test_dates[0]) - dates.index(fold.train_dates[-1]) >= 72
        assert set(fold.train_dates).isdisjoint(fold.test_dates)


def test_oos_score_store_rejects_unscored_or_ineligible_date() -> None:
    from datetime import date
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import OosScoreStore

    store = OosScoreStore(pl.DataFrame({'decision_date': [date(2026, 1, 5)], 'source_ticker': ['ONE'], 'score': [0.75], 'fold_id': [0], 'trained_through': [date(2025, 12, 31)], 'is_evaluation': [True]}))

    assert store.scores_for(date(2026, 1, 5), {'ONE', 'TWO'}) == {'ONE': 0.75}
    with pytest.raises(ValueError, match='OOS'):
        store.scores_for(date(2026, 1, 6), {'ONE'})
