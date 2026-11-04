from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from math import sqrt
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.arena.panel import Panel, PanelConfig, build_panel
from src.arena.signals import efficiency_ratio, flow_imbalance, rolling_z, trend_score, wide

PANEL_CFG = PanelConfig(lock_timeout_s=900.0, nav_divergence_max=0.10)
SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date(),
    "ticker": pl.String(),
    "name": pl.String(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Int64(),
    "trading_value": pl.Int64(),
    "nav": pl.Float64(),
    "shares_outstanding": pl.Int64(),
    "underlying_index_name": pl.String(),
}


def _bdays(n: int) -> list[date]:
    days: list[date] = []
    day = date(2024, 1, 1)
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _panel(
    closes: Mapping[str, Sequence[float | None]],
    shares: Mapping[str, Sequence[int]] | None = None,
    tickers: Sequence[str] | None = None,
) -> Panel:
    days = _bdays(len(next(iter(closes.values()))))
    rows: list[dict[str, Any]] = []
    for ticker, series in closes.items():
        for i, (day, close) in enumerate(zip(days, series, strict=True)):
            if close is None:
                continue
            count = shares[ticker][i] if shares is not None else 1000
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "name": f"{ticker} fund",
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000,
                    "trading_value": 10**9,
                    "nav": close,
                    "shares_outstanding": count,
                    "underlying_index_name": "IDX",
                }
            )
    frame = pl.DataFrame(rows, schema=SCHEMA)
    return build_panel(frame, list(tickers or closes), days[0], days[-1], PANEL_CFG)


def _series(values: Sequence[float | None], name: str = "x") -> pl.Series:
    return pl.Series(name, values, dtype=pl.Float64)


def _assert_same(actual: pl.Series, expected: pl.Series) -> None:
    assert actual.is_null().to_list() == expected.is_null().to_list()
    assert actual.drop_nulls().to_list() == pytest.approx(expected.drop_nulls().to_list(), rel=1e-12, abs=1e-12)


def test_wide_pivots_listed_columns_and_nulls_absent_tickers() -> None:
    panel = _panel({"AAA": [100.0, 101.0, None, 103.0], "BBB": [50.0, 51.0, 52.0, 53.0]}, tickers=["AAA", "BBB", "ZZZ"])
    table = wide(panel, "close_ref", ["AAA", "ZZZ", "AAA"])
    assert table.columns == ["date", "AAA", "ZZZ"]
    assert table.height == 4
    assert table["AAA"].to_list() == [100.0, 101.0, None, 103.0]
    assert table["ZZZ"].null_count() == 4
    assert table.schema["ZZZ"] == pl.Float64


def test_trend_score_known_values_and_clipping() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.002, 0.01, 200)
    prices = 100.0 * np.cumprod(1.0 + returns)
    ret_series = _series([None, *returns[1:].tolist()])
    horizons = (5, 10)
    vol_window = 20
    score = trend_score(_series(prices.tolist()), ret_series, horizons, vol_window).to_list()

    day = 150
    sigma = float(np.std(returns[day - vol_window + 1 : day + 1], ddof=1))
    zs = [min(1.0, max(-1.0, (prices[day] / prices[day - h] - 1.0) / (sigma * sqrt(h)))) for h in horizons]
    assert score[day] == pytest.approx(sum(zs) / len(zs), rel=1e-12)
    assert score[day] > 0.0

    jump = np.cumprod(np.r_[100.0, np.full(40, 1.01)])
    noise = np.where(np.arange(41) % 2 == 0, 0.0, 0.001)
    rising = trend_score(_series(jump.tolist()), _series([None, *noise[1:].tolist()]), (2, 4), 5).to_list()
    assert rising[-1] == pytest.approx(1.0)
    falling = trend_score(_series((jump[::-1]).tolist()), _series([None, *noise[1:].tolist()]), (2, 4), 5).to_list()
    assert falling[-1] == pytest.approx(-1.0)


def test_trend_score_null_until_history_and_on_zero_sigma() -> None:
    prices = _series([100.0 * 1.01**i for i in range(12)])
    returns = _series([None, *([0.01, -0.005] * 5 + [0.0])])
    score = trend_score(prices, returns, (3, 5), 4).to_list()
    assert all(v is None for v in score[:5])
    assert score[5] is not None

    flat_prices = _series([100.0] * 12)
    flat_returns = _series([None, *([0.0] * 11)])
    assert trend_score(flat_prices, flat_returns, (2, 3), 4).null_count() == 12


def test_efficiency_ratio_extremes_and_null_history() -> None:
    monotone = _series([100.0 + i for i in range(12)])
    ratio = efficiency_ratio(monotone, 4).to_list()
    assert ratio[:4] == [None, None, None, None]
    assert ratio[4:] == pytest.approx([1.0] * 8, rel=1e-12)

    alternating = _series([100.0 if i % 2 == 0 else 101.0 for i in range(12)])
    choppy = efficiency_ratio(alternating, 4).to_list()
    assert choppy[4:] == pytest.approx([0.0] * 8, abs=1e-12)

    flat = efficiency_ratio(_series([100.0] * 8), 3).to_list()
    assert all(v is None for v in flat)


def test_flow_imbalance_matches_hand_computation_and_nulls() -> None:
    lev_n = _series([100.0, 110.0, 120.0, 120.0, 150.0])
    lev_p = _series([10.0] * 5)
    inv_n = _series([50.0, 50.0, 40.0, 40.0, 40.0])
    inv_p = _series([20.0] * 5)
    ratio = flow_imbalance(lev_n, lev_p, inv_n, inv_p, 2).to_list()
    assert ratio[0] is None and ratio[1] is None
    assert ratio[2] == pytest.approx(400.0 / 2000.0, rel=1e-12)
    assert ratio[4] == pytest.approx(300.0 / 2300.0, rel=1e-12)

    with_gap = flow_imbalance(_series([100.0, 110.0, 120.0, None, 150.0]), lev_p, inv_n, inv_p, 2).to_list()
    assert with_gap[2] == pytest.approx(0.2, rel=1e-12)
    assert with_gap[3] is None and with_gap[4] is None

    empty_value = flow_imbalance(_series([0.0] * 3), _series([10.0] * 3), _series([0.0] * 3), _series([20.0] * 3), 2)
    assert empty_value.null_count() == 3


def test_rolling_z_known_values_and_zero_variance() -> None:
    z = rolling_z(_series([1.0, 2.0, 3.0, 4.0, 5.0]), 3).to_list()
    assert z[:2] == [None, None]
    assert z[2:] == pytest.approx([1.0, 1.0, 1.0], rel=1e-12)

    flat_then_step = rolling_z(_series([1.0, 1.0, 1.0, 2.0]), 3).to_list()
    assert flat_then_step[2] is None
    assert flat_then_step[3] == pytest.approx((2.0 - 4.0 / 3.0) / sqrt(1.0 / 3.0), rel=1e-12)


def test_signals_are_causal_on_rows_up_to_cut() -> None:
    rng = np.random.default_rng(11)
    n, cut = 90, 50
    base_returns = rng.normal(0.001, 0.02, n)
    base_prices = 100.0 * np.cumprod(1.0 + base_returns)
    shares_a = np.cumsum(rng.integers(-5, 6, n)) + 1000
    shares_b_other = np.cumsum(rng.integers(-5, 6, n)) + 1000
    price_b = base_prices.copy()
    price_b[cut + 1 :] = rng.uniform(50.0, 150.0, n - cut - 1)
    returns_b = np.r_[np.nan, np.diff(price_b) / price_b[:-1]]
    shares_b = shares_a.copy()
    shares_b[cut + 1 :] = shares_b_other[cut + 1 :]
    returns_a = np.r_[np.nan, np.diff(base_prices) / base_prices[:-1]]

    pa, pb = _series(base_prices.tolist()), _series(price_b.tolist())
    ra, rb = _series(returns_a.tolist()), _series(returns_b.tolist())
    sa, sb = _series(shares_a.astype(float).tolist()), _series(shares_b.astype(float).tolist())
    lev_p = _series(base_prices.tolist())
    inv_p = _series((base_prices * 0.5).tolist())
    inv_pb = _series((price_b * 0.5).tolist())

    cases = [
        (
            trend_score(pa, ra, (3, 7), 5),
            trend_score(pb, rb, (3, 7), 5),
        ),
        (efficiency_ratio(pa, 6), efficiency_ratio(pb, 6)),
        (
            flow_imbalance(sa, lev_p, _series((shares_a * 0 + 900.0).tolist()), inv_p, 4),
            flow_imbalance(sb, _series(price_b.tolist()), _series((shares_a * 0 + 900.0).tolist()), inv_pb, 4),
        ),
        (rolling_z(ra, 8), rolling_z(rb, 8)),
    ]
    for early, late in cases:
        assert early.len() == n
        _assert_same(early[: cut + 1], late[: cut + 1])
