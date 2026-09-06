from __future__ import annotations


def test_executable_label_limits_buys_to_horizon_fraction() -> None:
    import pytest
    from src.alpha.champion_ranker import TailGradeObjective
    from src.alpha.executable_labels import ExecutableLabelConfig, simulate_executable_vehicle_path
    from src.backtest.costs import CostConfig

    objective = TailGradeObjective((0.30, 0.40, 0.50, 0.60), (0.10, 0.25, 0.45, 0.20))
    config = ExecutableLabelConfig(horizon=4, fill_budget_fraction=0.25, capital=1_000.0, participation=0.10, max_single_weight=0.80, max_effective_gross=1.60, min_cash=0.05, costs=CostConfig(commission_bps=0.0, slippage_bps=0.0))
    label = simulate_executable_vehicle_path(opens=[100.0] * 4, closes=[100.0, 100.0, 100.0, 200.0], adv=[1_000.0] * 4, leverage_multiple=1, objective=objective, config=config)

    assert label.fill_sessions == 1
    assert label.filled_weight == pytest.approx(0.10)
    assert label.net_return == pytest.approx(0.10)
    assert label.tail_grade == 0


def test_simulate_executable_rejects_invalid_inputs() -> None:
    import pytest
    from src.alpha.champion_ranker import TailGradeObjective
    from src.alpha.executable_labels import ExecutableLabelConfig, simulate_executable_vehicle_path
    from src.backtest.costs import CostConfig

    objective = TailGradeObjective((0.30,), (1.0,))
    base = {'horizon': 4, 'fill_budget_fraction': 0.25, 'capital': 1_000.0, 'participation': 0.10, 'max_single_weight': 0.80, 'max_effective_gross': 1.60, 'min_cash': 0.05, 'costs': CostConfig(commission_bps=0.0, slippage_bps=0.0)}
    config = ExecutableLabelConfig(**base)
    with pytest.raises(ValueError, match='match horizon'):
        simulate_executable_vehicle_path([100.0] * 3, [100.0] * 4, [1_000.0] * 4, leverage_multiple=1, objective=objective, config=config)
    with pytest.raises(ValueError, match='non-finite'):
        simulate_executable_vehicle_path([100.0, float('nan'), 100.0, 100.0], [100.0] * 4, [1_000.0] * 4, leverage_multiple=1, objective=objective, config=config)
    with pytest.raises(ValueError, match='non-positive'):
        simulate_executable_vehicle_path([100.0] * 4, [0.0, 100.0, 100.0, 100.0], [1_000.0] * 4, leverage_multiple=1, objective=objective, config=config)
    with pytest.raises(ValueError, match='non-zero'):
        simulate_executable_vehicle_path([100.0] * 4, [100.0] * 4, [1_000.0] * 4, leverage_multiple=0, objective=objective, config=config)
    bad = ExecutableLabelConfig(**{**base, 'participation': 0.0})
    with pytest.raises(ValueError, match='positive'):
        simulate_executable_vehicle_path([100.0] * 4, [100.0] * 4, [1_000.0] * 4, leverage_multiple=1, objective=objective, config=bad)


def test_simulate_executable_missing_adv_never_fills() -> None:
    import pytest
    from src.alpha.champion_ranker import TailGradeObjective
    from src.alpha.executable_labels import ExecutableLabelConfig, simulate_executable_vehicle_path
    from src.backtest.costs import CostConfig

    objective = TailGradeObjective((0.30,), (1.0,))
    config = ExecutableLabelConfig(horizon=4, fill_budget_fraction=0.25, capital=1_000.0, participation=0.10, max_single_weight=0.80, max_effective_gross=1.60, min_cash=0.05, costs=CostConfig(commission_bps=1.0, slippage_bps=1.0))
    label = simulate_executable_vehicle_path([100.0] * 4, [100.0] * 4, [0.0] * 4, leverage_multiple=1, objective=objective, config=config)

    assert label.fill_sessions == 0
    assert label.filled_weight == pytest.approx(0.0)
    assert label.tail_grade == 0
    # Dust-level ADV cannot accumulate a measurable fill either.
    dust = simulate_executable_vehicle_path([100.0] * 4, [100.0] * 4, [1e-15] * 4, leverage_multiple=1, objective=objective, config=config)

    assert dust.fill_sessions == 0
    assert dust.filled_weight == pytest.approx(0.0)


def test_simulate_executable_stops_when_target_filled() -> None:
    import pytest
    from src.alpha.champion_ranker import TailGradeObjective
    from src.alpha.executable_labels import ExecutableLabelConfig, simulate_executable_vehicle_path
    from src.backtest.costs import CostConfig

    objective = TailGradeObjective((0.30,), (1.0,))
    config = ExecutableLabelConfig(horizon=4, fill_budget_fraction=1.0, capital=1_000.0, participation=1.0, max_single_weight=0.80, max_effective_gross=1.60, min_cash=0.05, costs=CostConfig(commission_bps=0.0, slippage_bps=0.0))
    label = simulate_executable_vehicle_path([100.0] * 4, [100.0, 100.0, 100.0, 150.0], adv=[1_000_000.0] * 4, leverage_multiple=2, objective=objective, config=config)

    assert label.fill_sessions == 1
    assert label.filled_weight == pytest.approx(0.80)
    assert label.net_return == pytest.approx(0.40)
    assert label.tail_grade == 1


def test_build_executable_vehicle_labels_chunks_and_skips() -> None:
    from datetime import date, timedelta
    import polars as pl
    from src.alpha.champion_ranker import TailGradeObjective
    from src.alpha.executable_labels import ExecutableLabelConfig, build_executable_vehicle_labels
    from src.backtest.costs import CostConfig

    objective = TailGradeObjective((0.30,), (1.0,))
    config = ExecutableLabelConfig(horizon=4, fill_budget_fraction=0.25, capital=1_000.0, participation=0.10, max_single_weight=0.80, max_effective_gross=1.60, min_cash=0.05, costs=CostConfig(commission_bps=0.0, slippage_bps=0.0))
    start = date(2026, 1, 2)
    sessions = [start + timedelta(days=i) for i in range(6)]
    panel = pl.DataFrame([{'date': d, 'ticker': 'ONE', 'open': 100.0, 'close': 101.0, 'trading_value': 1_000_000.0} for d in sessions])
    candidates = pl.DataFrame([
        {'decision_date': sessions[0], 'source_ticker': 'ONE', 'family_key': 'F1', 'leverage_multiple': 1},
        {'decision_date': sessions[1], 'source_ticker': 'GHOST', 'family_key': 'F9', 'leverage_multiple': 'xx'},
        {'decision_date': sessions[0], 'source_ticker': 'ONE', 'family_key': 'F1', 'leverage_multiple': 1},
    ])

    result = build_executable_vehicle_labels(candidates, panel, sessions=sessions, objective=objective, config=config, chunk_decision_dates=1)

    assert result.height == 2
    assert set(result.get_column('source_ticker').to_list()) == {'ONE'}
    assert result.schema['label_return'] == pl.Float64
    short = build_executable_vehicle_labels(candidates, panel, sessions=sessions[:2], objective=objective, config=config)
    assert short.height == 0
    nocand = build_executable_vehicle_labels(candidates.head(0), panel, sessions=sessions, objective=objective, config=config)
    assert nocand.height == 0


def test_build_executable_labels_tolerates_corrupt_panel_fields() -> None:
    from datetime import date, timedelta
    import polars as pl
    from src.alpha.champion_ranker import TailGradeObjective
    from src.alpha.executable_labels import ExecutableLabelConfig, build_executable_vehicle_labels
    from src.backtest.costs import CostConfig

    objective = TailGradeObjective((0.30,), (1.0,))
    config = ExecutableLabelConfig(horizon=4, fill_budget_fraction=0.25, capital=1_000.0, participation=0.10, max_single_weight=0.80, max_effective_gross=1.60, min_cash=0.05, costs=CostConfig(commission_bps=0.0, slippage_bps=0.0))
    start = date(2026, 1, 2)
    sessions = [start + timedelta(days=i) for i in range(6)]
    candidates = pl.DataFrame([{'decision_date': sessions[0], 'source_ticker': 'ONE', 'family_key': 'F1'}])
    # Non-date panel rows are ignored, leaving no executable path.
    assert build_executable_vehicle_labels(candidates, pl.DataFrame({'date': ['bad'], 'ticker': ['ONE'], 'open': [100.0], 'close': [101.0], 'trading_value': [1e6]}), sessions=sessions, objective=objective, config=config).height == 0
    # Corrupt opens/closes invalidate the path; corrupt ADV degrades to an unfilled label.
    bad_open = pl.DataFrame([{'date': d, 'ticker': 'ONE', 'open': 'bad', 'close': 101.0, 'trading_value': 1e6} for d in sessions])
    assert build_executable_vehicle_labels(candidates, bad_open, sessions=sessions, objective=objective, config=config).height == 0
    bad_close = pl.DataFrame([{'date': d, 'ticker': 'ONE', 'open': 100.0, 'close': 'bad', 'trading_value': 1e6} for d in sessions])
    assert build_executable_vehicle_labels(candidates, bad_close, sessions=sessions, objective=objective, config=config).height == 0
    bad_adv = pl.DataFrame([{'date': d, 'ticker': 'ONE', 'open': 100.0, 'close': 101.0, 'trading_value': 'bad'} for d in sessions])
    degraded = build_executable_vehicle_labels(candidates, bad_adv, sessions=sessions, objective=objective, config=config)

    assert degraded.height == 1
    assert degraded.item(0, 'fill_sessions') == 0
