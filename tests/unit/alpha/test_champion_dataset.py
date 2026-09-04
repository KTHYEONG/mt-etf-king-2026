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
