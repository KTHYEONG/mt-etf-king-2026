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


def test_champion_ranker_records_label_rank_as_training_target() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.champion_ranker import ChampionTailRanker

    train = pl.DataFrame({'decision_date': [date(2026, 1, 2), date(2026, 1, 2)], 'source_ticker': ['TWO_A', 'TWO_B'], 'family_key': ['A', 'B'], 'mom_5': [0.1, 0.2], 'label_return': [0.9, 0.35], 'label_tail_utility': [1.0, 0.1], 'label_rank': [1.0, 0.5]})

    artifact = ChampionTailRanker(('mom_5',), min_data_in_leaf=100).fit(train)

    assert artifact.target_column == 'label_rank'
    assert artifact.trained_through == date(2026, 1, 2)


def test_champion_ranker_rejects_missing_or_nonfinite_rank_target() -> None:
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import ChampionTailRanker
    from datetime import date

    base = {'decision_date': [date(2026, 1, 2)], 'mom_5': [0.1], 'label_return': [0.2]}
    with pytest.raises(ValueError, match='label_rank'):
        ChampionTailRanker(('mom_5',)).fit(pl.DataFrame(base))
    bad = {**base, 'label_rank': [float('nan')]}
    with pytest.raises(ValueError, match='no finite'):
        ChampionTailRanker(('mom_5',)).fit(pl.DataFrame(bad))
