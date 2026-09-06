from __future__ import annotations


def test_activation_calibration_rejects_tail_with_excess_ruin() -> None:
    from src.alpha.champion_ranker import TailGradeObjective
    from src.alpha.opportunity_hurdle import calibrate_activation_threshold

    objective = TailGradeObjective((0.30, 0.40, 0.50, 0.60), (0.10, 0.25, 0.45, 0.20))
    result = calibrate_activation_threshold([0.9, 0.8, 0.7, 0.6], [0.70, 0.65, -0.40, -0.35], [0.10] * 4, objective=objective, ruin_threshold=-0.25, max_ruin_probability=0.05, min_activations=4)

    assert result is None


def test_opportunity_model_delegates_to_incumbent_when_inactive() -> None:
    from datetime import date
    from types import SimpleNamespace
    import polars as pl
    from src.alpha.base import DecisionContext
    from src.alpha.champion_ranker import OosScoreStore
    from src.alpha.opportunity_hurdle import ExecutableOpportunityModel

    day = date(2026, 1, 5)
    store = OosScoreStore(pl.DataFrame({'decision_date': [day], 'source_ticker': ['DIRECT'], 'score': [0.9], 'activate': [False], 'is_evaluation': [True]}))
    incumbent = SimpleNamespace(score=lambda _snapshot, _context: {'sticky.mom60_raw': 1.0})
    model = ExecutableOpportunityModel(scores=store, incumbent=incumbent)
    context = DecisionContext(decision_date=day, regime=None, capital=1_000_000_000.0, held={}, rules=None)

    assert model.score(pl.DataFrame({'ticker': ['DIRECT', 'sticky.mom60_raw']}), context) == {'sticky.mom60_raw': 1.0}


def test_activation_calibration_finds_best_threshold_and_validates_inputs() -> None:
    from src.alpha.champion_ranker import TailGradeObjective
    from src.alpha.opportunity_hurdle import calibrate_activation_threshold

    objective = TailGradeObjective((0.30, 0.40, 0.50, 0.60), (0.10, 0.25, 0.45, 0.20))
    best = calibrate_activation_threshold([0.9, 0.8, 0.1, 0.05], [0.70, 0.65, 0.50, 0.45], [0.0] * 4, objective=objective, ruin_threshold=-0.25, max_ruin_probability=0.05, min_activations=1)

    assert best == 0.9
    # Mismatched, non-finite, or invalid control inputs cannot calibrate.
    assert calibrate_activation_threshold([0.9], [0.7, 0.6], [0.0, 0.0], objective=objective, ruin_threshold=-0.25, max_ruin_probability=0.05, min_activations=1) is None
    assert calibrate_activation_threshold([0.9, float('nan')], [0.7, 0.6], [0.0, 0.0], objective=objective, ruin_threshold=-0.25, max_ruin_probability=0.05, min_activations=1) is None
    assert calibrate_activation_threshold([0.9], [0.7], [0.0], objective=objective, ruin_threshold=float('nan'), max_ruin_probability=0.05, min_activations=1) is None
    assert calibrate_activation_threshold([0.9], [0.7], [0.0], objective=objective, ruin_threshold=-0.25, max_ruin_probability=0.05, min_activations=0) is None
    # Dominant incumbent blocks activation even with attainable margins.
    assert calibrate_activation_threshold([0.9, 0.8], [0.05, 0.04], [0.70, 0.65], objective=objective, ruin_threshold=-0.25, max_ruin_probability=0.05, min_activations=1) is None


def test_fit_executable_hurdle_trains_and_rejects_bad_inputs() -> None:
    from datetime import date
    from unittest.mock import patch
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import ChampionTrainingError, TailGradeObjective
    from src.alpha.opportunity_hurdle import fit_executable_hurdle

    objective = TailGradeObjective((0.30, 0.40), (0.5, 0.5))
    days = [date(2026, 1, 2)] * 4 + [date(2026, 1, 5)] * 4
    train = pl.DataFrame({
        'decision_date': days,
        'mom_5': [0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45],
        'label_return': [0.1, 0.2, 0.5, 0.6, 0.12, 0.22, 0.52, 0.62],
    })
    calibration = pl.DataFrame({
        'decision_date': [date(2026, 1, 6)] * 2,
        'mom_5': [0.2, 0.3],
        'label_return': [0.5, 0.1],
        'incumbent_return': [0.0, 0.0],
    })

    artifact = fit_executable_hurdle(train, calibration, feature_columns=('mom_5',), objective=objective, seed=7, min_activations=1)

    assert artifact.backend == 'lightgbm'
    assert len(artifact.threshold_models) == 2
    assert artifact.thresholds == (0.30, 0.40)
    assert artifact.trained_through == date(2026, 1, 5)
    with pytest.raises(ChampionTrainingError, match='empty feature'):
        fit_executable_hurdle(train, calibration, feature_columns=(), objective=objective, seed=7, min_activations=1)
    with pytest.raises(ChampionTrainingError, match='missing label_return'):
        fit_executable_hurdle(train.drop('label_return'), calibration, feature_columns=('mom_5',), objective=objective, seed=7, min_activations=1)
    with pytest.raises(ChampionTrainingError, match='no finite'):
        fit_executable_hurdle(train.with_columns(pl.lit(None).alias('mom_5')), calibration, feature_columns=('mom_5',), objective=objective, seed=7, min_activations=1)
    with pytest.raises(ChampionTrainingError, match='non-finite training'):
        fit_executable_hurdle(train.with_columns(pl.lit(float('inf')).alias('mom_5')), calibration, feature_columns=('mom_5',), objective=objective, seed=7, min_activations=1)
    with patch('lightgbm.train', side_effect=RuntimeError('boom')):  # noqa: SIM117
        with pytest.raises(ChampionTrainingError, match='boom'):
            fit_executable_hurdle(train, calibration, feature_columns=('mom_5',), objective=objective, seed=7, min_activations=1)


def test_fit_executable_hurdle_fails_closed_on_nonfinite_prediction_and_gates_outage() -> None:
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import patch
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import ChampionTrainingError, TailGradeObjective
    from src.alpha.opportunity_hurdle import fit_executable_hurdle

    objective = TailGradeObjective((0.30,), (1.0,))
    train = pl.DataFrame({'decision_date': [date(2026, 1, 2)] * 2, 'mom_5': [0.1, 0.2], 'label_return': [0.2, 0.5]})
    calibration = pl.DataFrame({'decision_date': [date(2026, 1, 6)], 'mom_5': [0.2], 'label_return': [0.4], 'incumbent_return': [0.0]})
    booster = SimpleNamespace(predict=lambda _x: [float('nan')])
    with patch('lightgbm.train', return_value=booster):  # noqa: SIM117
        with pytest.raises(ChampionTrainingError, match='non-finite prediction'):
            fit_executable_hurdle(train, calibration, feature_columns=('mom_5',), objective=objective, seed=7, min_activations=1)
    with patch('builtins.open', side_effect=OSError('gates outage')):
        artifact = fit_executable_hurdle(train, calibration, feature_columns=('mom_5',), objective=objective, seed=7, min_activations=1)
        assert artifact.backend == 'lightgbm'


def test_fit_executable_hurdle_fails_closed_on_corrupt_calibration() -> None:
    from datetime import date
    import polars as pl
    import pytest
    from src.alpha.champion_ranker import ChampionTrainingError, TailGradeObjective
    from src.alpha.opportunity_hurdle import fit_executable_hurdle

    objective = TailGradeObjective((0.30,), (1.0,))
    train = pl.DataFrame({'decision_date': [date(2026, 1, 2)] * 2, 'mom_5': [0.1, 0.2], 'label_return': [0.2, 0.5]})
    corrupt = pl.DataFrame({'decision_date': [date(2026, 1, 6)], 'mom_5': ['bad'], 'label_return': [0.4], 'incumbent_return': [0.0]})
    with pytest.raises(ChampionTrainingError, match='calibration failed'):
        fit_executable_hurdle(train, corrupt, feature_columns=('mom_5',), objective=objective, seed=7, min_activations=1)


def test_opportunity_model_hold_and_active_paths() -> None:
    from datetime import date
    from types import SimpleNamespace
    import polars as pl
    from src.alpha.base import DecisionContext
    from src.alpha.champion_ranker import OosScoreStore
    from src.alpha.opportunity_hurdle import ExecutableOpportunityModel
    from src.portfolio.intent import HOLD_INTENT

    day = date(2026, 1, 5)
    context = DecisionContext(decision_date=day, regime=None, capital=1_000_000_000.0, held={}, rules=None)
    incumbent = SimpleNamespace(score=lambda _snapshot, _context: {'sticky.mom60_raw': 1.0})
    store = OosScoreStore(pl.DataFrame({'decision_date': [day], 'source_ticker': ['DIRECT'], 'score': [0.9], 'activate': [False], 'is_evaluation': [True]}))
    model = ExecutableOpportunityModel(scores=store, incumbent=incumbent)

    assert model.score(pl.DataFrame({'ticker': []}), context) == HOLD_INTENT
    assert model.score(pl.DataFrame({'other': ['DIRECT']}), context) == HOLD_INTENT
    # Unscored dates and failing incumbents fail closed without activation.
    assert model.score(pl.DataFrame({'ticker': ['DIRECT']}), DecisionContext(decision_date=date(2026, 2, 1), regime=None, capital=1_000_000_000.0, held={}, rules=None)) == {'sticky.mom60_raw': 1.0}
    failing = SimpleNamespace(score=lambda _snapshot, _context: (_ for _ in ()).throw(RuntimeError('no incumbent')))
    active_store = OosScoreStore(pl.DataFrame({'decision_date': [day], 'source_ticker': ['DIRECT'], 'score': [0.9], 'activate': [True], 'is_evaluation': [True]}))
    assert ExecutableOpportunityModel(scores=active_store, incumbent=failing).score(pl.DataFrame({'ticker': ['DIRECT']}), context) == HOLD_INTENT
    assert ExecutableOpportunityModel(scores=active_store, incumbent=SimpleNamespace(score=lambda _s, _c: {})).score(pl.DataFrame({'ticker': ['DIRECT']}), context) == HOLD_INTENT
    # Active dates route direct scores once the incumbent confirms a live selection.
    assert ExecutableOpportunityModel(scores=active_store, incumbent=incumbent).score(pl.DataFrame({'ticker': ['DIRECT']}), context) == {'DIRECT': 0.9}
    # Non-finite direct scores delegate back to the incumbent.
    nan_store = OosScoreStore(pl.DataFrame({'decision_date': [day], 'source_ticker': ['DIRECT'], 'score': [float('nan')], 'activate': [True], 'is_evaluation': [True]}))
    assert ExecutableOpportunityModel(scores=nan_store, incumbent=incumbent).score(pl.DataFrame({'ticker': ['DIRECT']}), context) == {'sticky.mom60_raw': 1.0}
    # Snapshots keyed by source_ticker and unreadable snapshots also fail closed.
    assert ExecutableOpportunityModel(scores=store, incumbent=incumbent).score(pl.DataFrame({'source_ticker': ['DIRECT']}), context) == {'sticky.mom60_raw': 1.0}
    ghost = SimpleNamespace(height=1, columns=['ticker'])
    assert ExecutableOpportunityModel(scores=store, incumbent=incumbent).score(ghost, context) == HOLD_INTENT
