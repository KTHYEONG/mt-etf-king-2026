from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.core.paths import DataPaths
from src.reporting.results import write_backtest_result
from src.reporting.timeseries import enrich_daily_timeseries, normalize_trades_timeseries, write_timeseries_parquet


def test_enrich_daily_timeseries_cum_return_and_drawdown() -> None:
    daily = pl.DataFrame(
        {
            "date": [date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 6)],
            "ret": [0.10, -0.05, 0.02],
            "equity": [1_000_000_000.0, 1_050_000_000.0, 997_500_000.0],
        }
    )
    out = enrich_daily_timeseries(daily)
    assert "cum_return" in out.columns
    assert "drawdown" in out.columns
    assert float(out.sort("date")["cum_return"][-1]) == pytest.approx(-0.0025, abs=1e-9)


def test_normalize_trades_timeseries_buy_sell() -> None:
    trades = pl.DataFrame(
        {
            "decision_date": [date(2026, 1, 2), date(2026, 1, 3)],
            "execution_date": [date(2026, 1, 3), date(2026, 1, 6)],
            "ticker": ["069500", "069500"],
            "side": ["BUY", "SELL"],
            "weight_before": [0.0, 1.0],
            "weight_after": [1.0, 0.0],
            "delta_weight": [1.0, -1.0],
            "weight": [1.0, 0.0],
            "price": [100.0, 101.0],
        }
    )
    out = normalize_trades_timeseries(trades)
    assert out.height == 2
    assert set(out["side"].to_list()) == {"BUY", "SELL"}


def test_write_backtest_result_includes_parquet_artifacts(tmp_path) -> None:
    paths = DataPaths(root=tmp_path)
    run_id = "test_run_artifacts"
    daily = pl.DataFrame(
        {
            "date": [date(2026, 1, 2), date(2026, 1, 3)],
            "ret": [0.0, 0.01],
            "equity": [1.0, 1.01],
        }
    )
    trades = pl.DataFrame(
        {
            "decision_date": [date(2026, 1, 2)],
            "execution_date": [date(2026, 1, 3)],
            "ticker": ["069500"],
            "side": ["BUY"],
            "weight_before": [0.0],
            "weight_after": [1.0],
            "delta_weight": [1.0],
            "weight": [1.0],
            "price": [100.0],
        }
    )
    dest = write_backtest_result(
        paths,
        run_id=run_id,
        meta={"model": "B1"},
        summary={"n_windows": 1},
        daily=daily,
        trades=trades,
    )
    assert (dest / "daily.parquet").exists()
    assert (dest / "trades.parquet").exists()
    roundtrip = pl.read_parquet(dest / "trades.parquet")
    assert roundtrip.height == 1
    assert roundtrip["side"][0] == "BUY"
    artifacts = write_timeseries_parquet(dest, daily, trades)
    assert artifacts["daily"] == "daily.parquet"
