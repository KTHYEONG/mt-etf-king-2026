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


def test_tail_grade_objective_maps_thresholds_to_integer_gain() -> None:
    import numpy as np
    from src.alpha.champion_ranker import TailGradeObjective

    objective = TailGradeObjective(thresholds=(0.30, 0.40, 0.50, 0.60), weights=(0.10, 0.25, 0.45, 0.20))
    grades = objective.grade([0.30, 0.31, 0.41, 0.51, 0.61])

    assert grades.dtype == np.int32
    assert grades.tolist() == [0, 1, 2, 3, 4]
    assert objective.label_gain == (0.0, 0.10, 0.35, 0.80, 1.0)


def test_champion_ranker_fails_closed_when_lightgbm_training_raises() -> None:
    from datetime import date
    from unittest.mock import patch
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import ChampionTailRanker, ChampionTrainingError, TailGradeObjective

    train = pl.DataFrame({'decision_date': [date(2026, 1, 2)] * 2, 'source_ticker': ['A', 'B'], 'family_key': ['A', 'B'], 'mom_5': [0.1, 0.2], 'label_return': [0.2, 0.5]})
    objective = TailGradeObjective((0.30, 0.40, 0.50, 0.60), (0.10, 0.25, 0.45, 0.20))

    with patch('lightgbm.train', side_effect=RuntimeError('backend failed')):  # noqa: SIM117
        with pytest.raises(ChampionTrainingError, match='backend failed'):
            ChampionTailRanker(('mom_5',), objective=objective, min_data_in_leaf=100).fit(train)


def test_champion_artifact_rejects_fallback_backend() -> None:
    from datetime import date
    from types import SimpleNamespace
    from src.alpha.champion_ranker import is_valid_champion_artifact

    artifact = SimpleNamespace(backend='fallback-mean', lgbm_version='fallback-mean', trained_through=date(2025, 12, 31), feature_config_hash='x', model_config_hash='y', panel_hash='z', model={'mean': 0.5})

    assert is_valid_champion_artifact(artifact) is False


def test_tail_grade_objective_rejects_malformed_vectors() -> None:
    import pytest
    from src.alpha.champion_ranker import TailGradeObjective

    with pytest.raises(ValueError, match='non-empty'):
        TailGradeObjective(thresholds=(), weights=())
    with pytest.raises(ValueError, match='equal length'):
        TailGradeObjective(thresholds=(0.3,), weights=(0.1, 0.2))
    with pytest.raises(ValueError, match='positive finite'):
        TailGradeObjective(thresholds=(0.0,), weights=(1.0,))
    with pytest.raises(ValueError, match='strictly ascending'):
        TailGradeObjective(thresholds=(0.4, 0.3), weights=(0.5, 0.5))
    with pytest.raises(ValueError, match='non-negative'):
        TailGradeObjective(thresholds=(0.3,), weights=(-1.0,))
    with pytest.raises(ValueError, match='positive total'):
        TailGradeObjective(thresholds=(0.3,), weights=(0.0,))


def test_champion_ranker_fit_with_objective_uses_integer_grades() -> None:
    from datetime import date
    import polars as pl
    from src.alpha.champion_ranker import ChampionTailRanker, TailGradeObjective, is_valid_champion_artifact

    days = [date(2026, 1, 2)] * 4 + [date(2026, 1, 5)] * 4
    train = pl.DataFrame({
        'decision_date': days,
        'source_ticker': ['A', 'B', 'C', 'D'] * 2,
        'family_key': ['A', 'B', 'C', 'D'] * 2,
        'mom_5': [0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45],
        'label_return': [0.1, 0.35, 0.45, 0.7, 0.12, 0.36, 0.46, 0.71],
    })
    objective = TailGradeObjective((0.30, 0.40, 0.50, 0.60), (0.10, 0.25, 0.45, 0.20))

    artifact = ChampionTailRanker(('mom_5',), objective=objective, min_data_in_leaf=100).fit(train)

    assert artifact.backend == 'lightgbm'
    assert artifact.trained_through == date(2026, 1, 5)
    assert is_valid_champion_artifact(artifact) is True


def test_champion_ranker_objective_fit_rejects_missing_label_or_columns() -> None:
    from datetime import date
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import ChampionTailRanker, ChampionTrainingError, TailGradeObjective

    objective = TailGradeObjective((0.30,), (1.0,))
    ranker = ChampionTailRanker(('mom_5',), objective=objective, min_data_in_leaf=100)
    with pytest.raises(ChampionTrainingError, match='missing label'):
        ranker.fit(pl.DataFrame({'decision_date': [date(2026, 1, 2)], 'mom_5': [0.1]}))
    with pytest.raises(ChampionTrainingError, match='missing columns'):
        ranker.fit(pl.DataFrame({'decision_date': [date(2026, 1, 2)], 'mom_5': [0.1], 'label_return': [0.2], 'other': [1.0]}).rename({'mom_5': 'mom_x'}))
    with pytest.raises(ChampionTrainingError, match='no finite'):
        ranker.fit(pl.DataFrame({'decision_date': [date(2026, 1, 2)], 'mom_5': [None], 'label_return': [0.2]}))
    dup = pl.DataFrame({'decision_date': [date(2026, 1, 2)] * 2, 'mom_5': [0.1, 0.2], 'label_return': [0.2, 0.3]})
    with pytest.raises(ChampionTrainingError, match='column select failed'):
        ChampionTailRanker(('mom_5', 'mom_5'), objective=objective, min_data_in_leaf=100).fit(dup)


def test_champion_ranker_objective_fit_rejects_infinite_label_mismatch() -> None:
    from datetime import date
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import ChampionTailRanker, ChampionTrainingError, TailGradeObjective

    objective = TailGradeObjective((0.30,), (1.0,))
    train = pl.DataFrame({'decision_date': [date(2026, 1, 2)] * 2, 'mom_5': [0.1, 0.2], 'label_return': [0.2, float('inf')]})
    with pytest.raises(ChampionTrainingError, match='empty feature'):
        ChampionTailRanker(('mom_5',), objective=objective, min_data_in_leaf=100).fit(train)


def test_champion_ranker_legacy_fit_rejects_bad_columns_and_rows() -> None:
    from datetime import date
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import ChampionTailRanker

    good = {'decision_date': [date(2026, 1, 2)], 'mom_5': [0.1], 'label_return': [0.2], 'label_rank': [0.5]}
    with pytest.raises(ValueError, match='missing columns'):
        ChampionTailRanker(('nope',)).fit(pl.DataFrame(good))
    with pytest.raises(ValueError, match='missing label'):
        ChampionTailRanker(('mom_5',)).fit(pl.DataFrame({'decision_date': [date(2026, 1, 2)], 'mom_5': [0.1], 'label_rank': [0.5]}))
    wide = dict(good)
    for i in range(26):
        wide[f'f{i}'] = [0.1]
    with pytest.raises(ValueError, match='exceeds 25'):
        ChampionTailRanker().fit(pl.DataFrame(wide))
    nulls = {'decision_date': [date(2026, 1, 2)], 'mom_5': [None], 'label_return': [0.2], 'label_rank': [0.5]}
    with pytest.raises(ValueError, match='no finite'):
        ChampionTailRanker(('mom_5',)).fit(pl.DataFrame(nulls))


def test_champion_ranker_objective_rejects_too_many_inferred_features() -> None:
    from datetime import date
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import ChampionTailRanker, ChampionTrainingError, TailGradeObjective

    row: dict[str, object] = {'decision_date': date(2026, 1, 2), 'label_return': 0.2}
    for i in range(26):
        row[f'f{i}'] = 0.1
    frame = pl.DataFrame([row])
    with pytest.raises(ChampionTrainingError, match='exceeds 25'):
        ChampionTailRanker(objective=TailGradeObjective((0.30,), (1.0,))).fit(frame)


def test_champion_ranker_fails_closed_on_nonfinite_prediction() -> None:
    from datetime import date
    from unittest.mock import patch
    from types import SimpleNamespace
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import ChampionTailRanker, ChampionTrainingError, TailGradeObjective

    train = pl.DataFrame({'decision_date': [date(2026, 1, 2)] * 2, 'mom_5': [0.1, 0.2], 'label_return': [0.2, 0.5]})
    objective = TailGradeObjective((0.30,), (1.0,))
    booster = SimpleNamespace(predict=lambda _x: [float('nan')])
    with patch('lightgbm.train', return_value=booster):  # noqa: SIM117
        with pytest.raises(ChampionTrainingError, match='non-finite prediction'):
            ChampionTailRanker(('mom_5',), objective=objective, min_data_in_leaf=100).fit(train)


def test_is_valid_champion_artifact_field_matrix() -> None:
    from datetime import date
    from types import SimpleNamespace
    from src.alpha.champion_ranker import is_valid_champion_artifact

    class _Model:
        def predict(self, _x):
            return [0.1]

    base = {'backend': 'lightgbm', 'lgbm_version': '4.7.0', 'trained_through': date(2025, 12, 31), 'feature_config_hash': 'x', 'model_config_hash': 'y', 'panel_hash': 'z', 'model': _Model()}
    assert is_valid_champion_artifact(SimpleNamespace(**base)) is True
    assert is_valid_champion_artifact(SimpleNamespace(**{**base, 'lgbm_version': ''})) is False
    assert is_valid_champion_artifact(SimpleNamespace(**{**base, 'feature_config_hash': ''})) is False
    assert is_valid_champion_artifact(SimpleNamespace(**{**base, 'trained_through': None})) is False
    assert is_valid_champion_artifact(SimpleNamespace(**{**base, 'model': None})) is False
    assert is_valid_champion_artifact(SimpleNamespace(**{**base, 'model': {'mean': 0.5}})) is False
    assert is_valid_champion_artifact(SimpleNamespace(**{**base, 'model': object()})) is False
    assert is_valid_champion_artifact(object()) is False

    class _Boom:
        def __getattribute__(self, _name):
            raise RuntimeError('unreadable artifact')

    assert is_valid_champion_artifact(_Boom()) is False


def test_champion_ranker_score_skips_bad_rows_and_failing_models() -> None:
    from datetime import date
    from types import SimpleNamespace
    import polars as pl
    from src.alpha.champion_ranker import ChampionTailRanker

    train = pl.DataFrame({'decision_date': [date(2026, 1, 2)] * 2, 'source_ticker': ['A', 'B'], 'family_key': ['A', 'B'], 'mom_5': [0.1, 0.2], 'label_return': [0.2, 0.3], 'label_rank': [0.4, 0.6]})
    artifact = ChampionTailRanker(('mom_5',)).fit(train)
    snapshot = pl.DataFrame({'source_ticker': ['GOOD', 'BAD'], 'mom_5': [0.15, float('inf')]})
    scored = ChampionTailRanker(('mom_5',)).score(snapshot, artifact=artifact)

    assert set(scored.keys()) == {'GOOD'}

    class _Failing:
        def predict(self, _x):
            raise ValueError('predict unavailable')

    broken = SimpleNamespace(feature_columns=('mom_5',), model=_Failing())
    assert ChampionTailRanker(('mom_5',)).score(pl.DataFrame({'source_ticker': ['A'], 'mom_5': [0.1]}), artifact=broken) == {}


def test_champion_ranker_fails_closed_when_prediction_raises_unexpected() -> None:
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import patch
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import ChampionTailRanker, ChampionTrainingError, TailGradeObjective

    train = pl.DataFrame({'decision_date': [date(2026, 1, 2)] * 2, 'mom_5': [0.1, 0.2], 'label_return': [0.2, 0.5]})
    objective = TailGradeObjective((0.30,), (1.0,))
    booster = SimpleNamespace(predict=lambda _x: (_ for _ in ()).throw(RuntimeError('predict exploded')))
    with patch('lightgbm.train', return_value=booster):  # noqa: SIM117
        with pytest.raises(ChampionTrainingError, match='prediction failed'):
            ChampionTailRanker(('mom_5',), objective=objective, min_data_in_leaf=100).fit(train)


def test_oos_score_store_decision_for_reports_activation() -> None:
    from datetime import date
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import OosScoreStore

    day = date(2026, 1, 5)
    store = OosScoreStore(pl.DataFrame({'decision_date': [day], 'source_ticker': ['A'], 'score': [0.4], 'activate': [True], 'is_evaluation': [True]}))
    scores, active = store.decision_for(day, {'A'})
    assert scores == {'A': 0.4}
    assert active is True
    plain = OosScoreStore(pl.DataFrame({'decision_date': [day], 'source_ticker': ['A'], 'score': [0.4]}))
    scores2, active2 = plain.decision_for(day, {'A'})
    assert scores2 == {'A': 0.4}
    assert active2 is False
    with pytest.raises(ValueError, match='OOS'):
        plain.decision_for(date(2026, 1, 6), {'A'})
    crowded = OosScoreStore(pl.DataFrame({'decision_date': [day, day], 'source_ticker': ['A', 'B'], 'score': [0.4, 0.9], 'activate': [False, True], 'is_evaluation': [True, True]}))
    scores3, active3 = crowded.decision_for(day, {'A'})
    assert scores3 == {'A': 0.4}
    assert active3 is False
