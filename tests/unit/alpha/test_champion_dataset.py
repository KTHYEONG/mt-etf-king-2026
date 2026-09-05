from __future__ import annotations


def test_family_tail_label_uses_next_open_real_one_x_prices() -> None:
    from datetime import date
    import polars as pl
    from src.alpha.champion_dataset import ChampionDatasetConfig, build_family_tail_dataset

    candidates = pl.DataFrame({'decision_date': [date(2026, 1, 2)], 'source_ticker': ['ONE_X'], 'family_key': ['SEMICONDUCTOR'], 'mom_5': [0.10]})
    prices = pl.DataFrame({'date': [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)], 'ticker': ['ONE_X', 'ONE_X', 'ONE_X', 'ONE_X'], 'open': [100.0, 110.0, 120.0, 130.0], 'close': [100.0, 115.0, 125.0, 143.0]})
    config = ChampionDatasetConfig(feature_columns=('mom_5',), label_horizon=2, entry_cost_rate=0.001, exit_cost_rate=0.001, source_multiple=1)
    result = build_family_tail_dataset(candidates, prices, sessions=[date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)], config=config)

    assert result.height == 1
    expected = (125.0 / 110.0) * (1.0 - 0.001) / (1.0 + 0.001) - 1.0
    assert abs(result.item(0, 'label_return') - expected) < 1e-12
    assert result.item(0, 'source_ticker') == 'ONE_X'


def test_family_tail_label_normalizes_mixed_numeric_feature_types() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.champion_dataset import ChampionDatasetConfig, build_family_tail_dataset

    d0, d1, d2 = date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)
    candidates = pl.DataFrame(
        {
            'decision_date': [d0, d0],
            'source_ticker': ['ONE', 'TWO'],
            'family_key': ['ONE', 'TWO'],
            'mom_5': pl.Series('mom_5', [0, 0.073575], dtype=pl.Float64),
        }
    )
    prices = pl.DataFrame(
        {
            'date': [d0, d1, d2, d0, d1, d2],
            'ticker': ['ONE', 'ONE', 'ONE', 'TWO', 'TWO', 'TWO'],
            'open': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            'close': [100.0, 110.0, 120.0, 100.0, 110.0, 120.0],
        }
    )

    result = build_family_tail_dataset(
        candidates,
        prices,
        sessions=[d0, d1, d2],
        config=ChampionDatasetConfig(
            feature_columns=('mom_5',), label_horizon=2, entry_cost_rate=0.0, exit_cost_rate=0.0,
        ),
    )

    assert result.height == 2
    assert result.schema['mom_5'] == pl.Float64
    assert result.get_column('mom_5').to_list() == [0.0, 0.073575]


def test_collect_family_candidates_is_deployment_pit_one_x_only() -> None:
    from datetime import date
    import polars as pl
    from src.alpha.champion_dataset import ChampionDatasetConfig, collect_family_candidates
    from src.universe.provider import UniverseFilters, UniverseMode
    from tests.unit.universe.conftest import build_universe

    d0, d1 = date(2026, 1, 2), date(2026, 1, 5)
    panel = pl.DataFrame([
        {'date': d0, 'ticker': 'ONE', 'name': 'KODEX 200', 'underlying_index_name': 'KOSPI 200', 'is_tradable': True, 'close': 100.0, 'open': 100.0, 'trading_value': 2e12, 'mom_5': 0.1},
        {'date': d0, 'ticker': 'TWO', 'name': 'KODEX 레버리지', 'underlying_index_name': 'KOSPI 200', 'is_tradable': True, 'close': 100.0, 'open': 100.0, 'trading_value': 2e12, 'mom_5': 0.2},
        {'date': d1, 'ticker': 'ONE', 'name': 'KODEX 200', 'underlying_index_name': 'KOSPI 200', 'is_tradable': True, 'close': 101.0, 'open': 101.0, 'trading_value': 2e12, 'mom_5': 0.1},
        {'date': d1, 'ticker': 'TWO', 'name': 'KODEX 레버리지', 'underlying_index_name': 'KOSPI 200', 'is_tradable': True, 'close': 102.0, 'open': 102.0, 'trading_value': 2e12, 'mom_5': 0.2},
        {'date': d1, 'ticker': 'LATE', 'name': 'KODEX 반도체', 'underlying_index_name': 'SEMICONDUCTOR', 'is_tradable': True, 'close': 100.0, 'open': 100.0, 'trading_value': 2e12, 'mom_5': 0.9},
    ])
    universe, master, _ = build_universe(panel, adv_window=1, brand_map={})
    config = ChampionDatasetConfig(feature_columns=('mom_5',), label_horizon=2, entry_cost_rate=0.0, exit_cost_rate=0.0, source_multiple=1)
    filters = UniverseFilters(mode=UniverseMode.STRUCTURAL, warmup_sessions=0, capital=1, max_order_to_adv=1.0)
    result = collect_family_candidates(panel, sessions=[d0], universe=universe, filters=filters, master=master, config=config)

    assert result.select('source_ticker').to_series().to_list() == ['ONE']
    assert result.select('family_key').to_series().to_list() == [result.item(0, 'family_key')]
    assert result.height == 1


def test_collect_family_candidates_source_two_labels_exact_plus_two_vehicle() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.champion_dataset import ChampionDatasetConfig, build_family_tail_dataset, collect_family_candidates
    from src.universe.provider import UniverseFilters, UniverseMode
    from tests.unit.universe.conftest import build_universe

    d0, d1, d2 = date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)
    panel = pl.DataFrame([
        {'date': d0, 'ticker': 'ONE', 'name': 'KODEX 200', 'underlying_index_name': 'KOSPI 200', 'is_tradable': True, 'open': 100.0, 'close': 100.0, 'trading_value': 2e12, 'mom_60': 0.10},
        {'date': d0, 'ticker': 'TWO', 'name': 'KODEX 레버리지', 'underlying_index_name': 'KOSPI 200', 'is_tradable': True, 'open': 100.0, 'close': 100.0, 'trading_value': 2e12, 'mom_60': 0.20},
        {'date': d1, 'ticker': 'ONE', 'name': 'KODEX 200', 'underlying_index_name': 'KOSPI 200', 'is_tradable': True, 'open': 100.0, 'close': 105.0, 'trading_value': 2e12, 'mom_60': 0.10},
        {'date': d1, 'ticker': 'TWO', 'name': 'KODEX 레버리지', 'underlying_index_name': 'KOSPI 200', 'is_tradable': True, 'open': 100.0, 'close': 110.0, 'trading_value': 2e12, 'mom_60': 0.20},
        {'date': d2, 'ticker': 'ONE', 'name': 'KODEX 200', 'underlying_index_name': 'KOSPI 200', 'is_tradable': True, 'open': 105.0, 'close': 106.0, 'trading_value': 2e12, 'mom_60': 0.10},
        {'date': d2, 'ticker': 'TWO', 'name': 'KODEX 레버리지', 'underlying_index_name': 'KOSPI 200', 'is_tradable': True, 'open': 110.0, 'close': 121.0, 'trading_value': 2e12, 'mom_60': 0.20},
    ])
    universe, master, _ = build_universe(panel, adv_window=1, brand_map={})
    config = ChampionDatasetConfig(feature_columns=('mom_60',), label_horizon=2, entry_cost_rate=0.0, exit_cost_rate=0.0, source_multiple=2)
    filters = UniverseFilters(mode=UniverseMode.STRUCTURAL, warmup_sessions=0, capital=1, max_order_to_adv=1.0)

    candidates = collect_family_candidates(panel, sessions=[d0], universe=universe, filters=filters, master=master, config=config)
    labeled = build_family_tail_dataset(candidates, panel, sessions=[d0, d1, d2], config=config)

    assert candidates.get_column('source_ticker').to_list() == ['TWO']
    assert labeled.item(0, 'source_ticker') == 'TWO'
    assert labeled.item(0, 'label_return') == 0.21


def test_family_tail_dataset_ranks_primary_tail_utility_before_raw_return() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.champion_dataset import ChampionDatasetConfig, build_family_tail_dataset

    d0, d1, d2 = date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)
    candidates = pl.DataFrame({'decision_date': [d0, d0], 'source_ticker': ['TWO_A', 'TWO_B'], 'family_key': ['A', 'B'], 'mom_5': [0.1, 0.2]})
    prices = pl.DataFrame({'date': [d0, d1, d2, d0, d1, d2], 'ticker': ['TWO_A', 'TWO_A', 'TWO_A', 'TWO_B', 'TWO_B', 'TWO_B'], 'open': [100.0] * 6, 'close': [100.0, 155.0, 155.0, 100.0, 135.0, 135.0]})
    config = ChampionDatasetConfig(feature_columns=('mom_5',), label_horizon=2, entry_cost_rate=0.0, exit_cost_rate=0.0, source_multiple=2, tail_thresholds=(0.30, 0.40, 0.50, 0.60), tail_weights=(0.10, 0.25, 0.45, 0.20))

    result = build_family_tail_dataset(candidates, prices, sessions=[d0, d1, d2], config=config).sort('source_ticker')

    assert result.get_column('label_tail_utility').to_list() == [0.8, 0.1]
    assert result.get_column('label_rank').to_list() == [1.0, 0.5]
    assert result.get_column('label_return').to_list() == [0.55, 0.35]


def test_champion_dataset_rejects_malformed_primary_tail_objective() -> None:
    from datetime import date

    import polars as pl
    import pytest

    from src.alpha.champion_dataset import ChampionDatasetConfig, build_family_tail_dataset

    d0, d1, d2 = date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)
    candidates = pl.DataFrame({'decision_date': [d0], 'source_ticker': ['TWO'], 'family_key': ['A'], 'mom_5': [0.1]})
    prices = pl.DataFrame({'date': [d0, d1, d2], 'ticker': ['TWO', 'TWO', 'TWO'], 'open': [100.0] * 3, 'close': [100.0, 140.0, 140.0]})
    config = ChampionDatasetConfig(feature_columns=('mom_5',), label_horizon=2, entry_cost_rate=0.0, exit_cost_rate=0.0, source_multiple=2, tail_thresholds=(0.30, 0.40), tail_weights=(1.0,))

    with pytest.raises(ValueError, match='tail'):
        build_family_tail_dataset(candidates, prices, sessions=[d0, d1, d2], config=config)


def test_champion_dataset_rejects_invalid_tail_objective_values() -> None:
    from datetime import date
    import polars as pl
    import pytest
    from src.alpha.champion_dataset import ChampionDatasetConfig, build_family_tail_dataset

    d0, d1, d2 = date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)
    candidates = pl.DataFrame({'decision_date': [d0], 'source_ticker': ['T'], 'family_key': ['A'], 'x': [1.0]})
    prices = pl.DataFrame({'date': [d0, d1, d2], 'ticker': ['T'] * 3, 'open': [100.0] * 3, 'close': [100.0, 120.0, 120.0]})
    base = {'feature_columns': ('x',), 'label_horizon': 2, 'entry_cost_rate': 0.0, 'exit_cost_rate': 0.0}
    cases = [((0.0,), (1.0,), 'positive'), ((0.4, 0.3), (1.0, 1.0), 'ascending'), ((0.4,), (-1.0,), 'non-negative'), ((0.4,), (0.0,), 'positive total')]
    for thresholds, weights, match in cases:
        with pytest.raises(ValueError, match=match):
            build_family_tail_dataset(candidates, prices, sessions=[d0, d1, d2], config=ChampionDatasetConfig(**base, tail_thresholds=thresholds, tail_weights=weights))
