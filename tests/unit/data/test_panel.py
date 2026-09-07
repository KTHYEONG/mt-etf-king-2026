# ruff: noqa
from __future__ import annotations

from datetime import date

import polars as pl

from src.core.paths import DataPaths
from src.data.panel import BACKTEST_PANEL_COLUMNS, load_backtest_panel


def test_SCENARIO_DSR_03_projection_and_date_filter(tmp_path) -> None:  # noqa: N802, ANN001
    """SCENARIO_DSR_03: projection and date filter."""
    paths = DataPaths(root=tmp_path)
    # create gold parquet with full columns plus extra
    gold_path = paths.gold("etf_features")
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    # generate dates
    dates = [date(2026, 1, d) for d in range(1, 11)]  # 10 days
    rows = []
    for d in dates:
        for ticker in ["069500", "091160"]:
            row = {
                "date": d,
                "ticker": ticker,
                "close": 100.0,
                "open": 99.0,
                "trading_value": 1_000_000,
                "is_tradable": True,
                "name": "Test",
                "underlying_index_name": "IDX",
                "mom_3": 0.01,
                "mom_5": 0.02,
                "mom_10": 0.03,
                "mom_20": 0.04,
                "mom_40": 0.05,
                "mom_60": 0.06,
                "ma_20": 95.0,
                "ma_ratio_20": 1.05,
                "drawdown_20": -0.02,
                "extra_col": "should_not_be_loaded",
            }
            rows.append(row)
    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    df.write_parquet(str(gold_path), compression="zstd")
    # full load
    full = load_backtest_panel(paths)
    assert full is not None
    # requested subset
    requested = ["date", "ticker", "close", "mom_20", "ma_20"]
    start = date(2026, 1, 3)
    end = date(2026, 1, 7)
    filtered = load_backtest_panel(paths, columns=requested, start=start, end=end)
    assert filtered is not None
    assert set(filtered.columns).issubset(set(requested))
    # date filter
    assert filtered.height < full.height
    # check dates within range
    mins = filtered.select(pl.col("date").min()).item()
    maxs = filtered.select(pl.col("date").max()).item()
    assert mins >= start
    assert maxs <= end
    # also verify that extra_col not present when not requested
    assert "extra_col" not in filtered.columns
    # verify BACKTEST_PANEL_COLUMNS invariant
    for col in ["date", "ticker", "close", "open", "trading_value", "is_tradable", "name", "underlying_index_name", "mom_3", "mom_5", "mom_10", "mom_20", "mom_40", "mom_60", "ma_20", "ma_ratio_20", "drawdown_20"]:
        assert col in BACKTEST_PANEL_COLUMNS
    # ensure not claiming full gold schema: loader intersects
    assert "extra_col" not in BACKTEST_PANEL_COLUMNS


def test_panel_fallback_to_silver(tmp_path) -> None:  # noqa: ANN001
    """Fallback to silver when gold missing."""
    paths = DataPaths(root=tmp_path)
    silver_path = paths.silver("etf_daily")
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "date": [date(2026, 1, 1), date(2026, 1, 2)],
            "ticker": ["069500", "069500"],
            "close": [100.0, 101.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    df.write_parquet(str(silver_path))
    # gold missing, should fallback
    loaded = load_backtest_panel(paths, columns=["date", "ticker", "close"])
    assert loaded is not None
    assert loaded.height == 2


def test_panel_both_sources_apply_date_bounds(tmp_path) -> None:  # noqa: ANN001
    paths = DataPaths(root=tmp_path)
    paths.silver("etf_daily").parent.mkdir(parents=True, exist_ok=True)
    paths.gold("etf_features").parent.mkdir(parents=True, exist_ok=True)
    silver = pl.DataFrame({"date": [date(2026, 1, 1), date(2026, 1, 2)], "ticker": ["A", "A"], "open": [10.0, 10.0], "close": [10.0, 11.0], "is_tradable": [True, True], "trading_value": [100.0, 100.0]})
    gold = silver.with_columns(pl.lit(0.1).alias("mom_60"))
    silver.write_parquet(paths.silver("etf_daily"))
    gold.write_parquet(paths.gold("etf_features"))
    result = load_backtest_panel(paths, columns=["date", "ticker", "close"], start=date(2026, 1, 2), end=date(2026, 1, 2))
    assert result is not None and result.height == 1


def test_backtest_panel_includes_volume_expansion() -> None:
    from src.data.panel import BACKTEST_PANEL_COLUMNS

    assert "volume_expansion" in BACKTEST_PANEL_COLUMNS
    assert "drawdown_20" in BACKTEST_PANEL_COLUMNS
    assert "mom_5" in BACKTEST_PANEL_COLUMNS
    assert "extra_col" not in BACKTEST_PANEL_COLUMNS

def test_align_feature_rows_drops_only_empty_extra_rows() -> None:
    from datetime import date
    import polars as pl
    from src.data.panel import align_feature_rows
    silver = pl.DataFrame({"date":[date(2024,1,2)],"ticker":["A"],"open":[10.0],"close":[11.0],"is_tradable":[True],"trading_value":[100.0]})
    gold = silver.with_columns(pl.lit(0.1).alias("mom_60"))
    extra=pl.DataFrame({"date":[date(2024,1,3)],"ticker":["A"],"open":[None],"close":[None],"is_tradable":[False],"trading_value":[0.0],"mom_60":[None]},schema=gold.schema)
    result=align_feature_rows(silver,pl.concat([gold,extra]))
    assert result.equals(gold)

def test_align_feature_rows_rejects_stale_or_duplicate_quotes() -> None:
    from datetime import date
    import polars as pl
    from src.data.panel import align_feature_rows
    silver = pl.DataFrame({"date":[date(2024,1,2)],"ticker":["A"],"open":[10.0],"close":[11.0],"is_tradable":[True],"trading_value":[100.0]})
    gold = silver.with_columns(pl.lit(0.1).alias("mom_60"))
    import pytest
    for bad in (gold.with_columns(pl.lit(99.0).alias("close")),pl.concat([gold,gold]),gold.with_columns(pl.lit(date(2024,1,3)).alias("date")),gold.head(0)):
        with pytest.raises(ValueError):
            align_feature_rows(silver,bad)

def test_load_backtest_panel_validates_before_projection(tmp_path) -> None:
    from datetime import date
    import polars as pl
    from src.data.panel import align_feature_rows
    silver = pl.DataFrame({"date":[date(2024,1,2)],"ticker":["A"],"open":[10.0],"close":[11.0],"is_tradable":[True],"trading_value":[100.0]})
    gold = silver.with_columns(pl.lit(0.1).alias("mom_60"))
    from src.core.paths import DataPaths
    from src.data.panel import load_backtest_panel
    import pytest
    paths=DataPaths(root=tmp_path)
    paths.silver("etf_daily").parent.mkdir(parents=True)
    paths.gold("etf_features").parent.mkdir(parents=True)
    silver.write_parquet(paths.silver("etf_daily"))
    gold.with_columns(pl.lit(99.0).alias("close")).write_parquet(paths.gold("etf_features"))
    with pytest.raises(ValueError):
        load_backtest_panel(paths,columns=["date","ticker","mom_60"])
