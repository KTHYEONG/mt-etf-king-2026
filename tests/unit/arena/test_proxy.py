from __future__ import annotations

import copy
import math
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.arena import proxy
from src.arena.panel import PanelConfig, PanelSourceError, build_panel
from src.arena.proxy import (
    ProxyConfig,
    ProxyError,
    ProxyProduct,
    calibrate,
    load_proxy_config,
    load_stock_silver,
    proxy_rows,
    splice_proxy,
    underlying_returns,
)
from src.core.config import ConfigError, load_config
from src.core.paths import DataPaths

PANEL_CFG = PanelConfig(lock_timeout_s=900.0, nav_divergence_max=0.10)
PRODUCT = ProxyProduct(etf="0193T0", underlying="000660", multiple=2.0)
INVERSE = ProxyProduct(etf="0197X0", underlying="000660", multiple=-2.0)
UNDER = "000660"
FILLER = "069500"
DRAG_BPS = 5.0
CFG = ProxyConfig(
    tag="test",
    execution="close",
    drag_bps_per_session=DRAG_BPS,
    max_abs_session_return=0.30,
    min_overlap_sessions=60,
    min_close_corr=0.99,
    max_close_te_bps=150.0,
    products=(PRODUCT,),
)
N = 130
LISTING = 40
ANCHOR = 1000.0
DAYS = [date(2020, 1, 1) + timedelta(days=i) for i in range(N)]
STOCK_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "ticker": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "change_abs": pl.Float64,
    "volume": pl.Int64,
    "trading_value": pl.Int64,
}
ETF_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "ticker": pl.String,
    "name": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "trading_value": pl.Int64,
    "nav": pl.Float64,
    "shares_outstanding": pl.Int64,
    "underlying_index_name": pl.String,
}


def _returns(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    r_cc = rng.normal(0.0, 0.02, N)
    r_cc[0] = 0.0
    r_on = rng.normal(0.0, 0.01, N)
    return r_cc, r_on


def _prices(r_cc: np.ndarray, r_on: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    closes = 100.0 * np.cumprod(1.0 + r_cc)
    prev = np.concatenate([closes[:1], closes[:-1]])
    return closes, prev * (1.0 + r_on)


def _stock(closes: np.ndarray, opens: np.ndarray, *, volume: int = 1000, zero_volume: tuple[int, ...] = ()) -> pl.DataFrame:
    volumes = np.full(N, volume, dtype=np.int64)
    volumes[np.array(zero_volume, dtype=np.int64)] = 0
    prev = np.concatenate([closes[:1], closes[:-1]])
    return pl.DataFrame(
        {
            "date": DAYS,
            "ticker": [UNDER] * N,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.01,
            "low": np.minimum(opens, closes) * 0.99,
            "close": closes,
            "change_abs": closes - prev,
            "volume": volumes,
            "trading_value": np.rint(closes * volumes).astype(np.int64),
        },
        schema=STOCK_SCHEMA,
    )


def _raw(
    closes: list[float],
    change: list[float],
    *,
    volume: list[int] | None = None,
    trading_value: list[int | None] | None = None,
    ticker: str = UNDER,
) -> pl.DataFrame:
    n = len(closes)
    vols = [1000] * n if volume is None else volume
    tvs: list[int | None] = [round(c * v) for c, v in zip(closes, vols, strict=True)] if trading_value is None else trading_value
    return pl.DataFrame(
        {
            "date": DAYS[:n],
            "ticker": [ticker] * n,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "change_abs": change,
            "volume": vols,
            "trading_value": tvs,
        },
        schema=STOCK_SCHEMA,
    )


def _table(
    product: ProxyProduct,
    r_cc: np.ndarray,
    r_on: np.ndarray,
    *,
    follow: bool = True,
    listing: int = LISTING,
    seed: int = 11,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    if follow:
        ret = product.multiple * r_cc - DRAG_BPS / 1e4 + rng.normal(0.0, 0.0005, N)
        ret_on = product.multiple * r_on + rng.normal(0.0, 0.002, N)
    else:
        ret = rng.normal(0.0, 0.04, N)
        ret_on = rng.normal(0.0, 0.04, N)
    closes = np.full(N, np.nan)
    opens = np.full(N, np.nan)
    closes[listing] = ANCHOR
    opens[listing] = ANCHOR
    for t in range(listing + 1, N):
        closes[t] = closes[t - 1] * (1.0 + ret[t])
        opens[t] = closes[t - 1] * (1.0 + ret_on[t])
    listed = slice(listing, N)
    n_listed = N - listing
    etf = pl.DataFrame(
        {
            "date": DAYS[listing:],
            "ticker": [product.etf] * n_listed,
            "name": ["ETF"] * n_listed,
            "open": opens[listed],
            "high": np.maximum(opens[listed], closes[listed]) * 1.01,
            "low": np.minimum(opens[listed], closes[listed]) * 0.99,
            "close": closes[listed],
            "volume": np.full(n_listed, 1000, dtype=np.int64),
            "trading_value": np.rint(closes[listed] * 1000).astype(np.int64),
            "nav": closes[listed],
            "shares_outstanding": np.full(n_listed, 1000, dtype=np.int64),
            "underlying_index_name": ["IDX"] * n_listed,
        },
        schema=ETF_SCHEMA,
    )
    filler = pl.DataFrame(
        {
            "date": DAYS,
            "ticker": [FILLER] * N,
            "name": ["KODEX 200"] * N,
            "open": [100.0] * N,
            "high": [101.0] * N,
            "low": [99.0] * N,
            "close": [100.0] * N,
            "volume": [1000] * N,
            "trading_value": [100_000] * N,
            "nav": [100.0] * N,
            "shares_outstanding": [1000] * N,
            "underlying_index_name": ["KOSPI200"] * N,
        },
        schema=ETF_SCHEMA,
    )
    return pl.concat([etf, filler])


def test_config_loads_and_rejects_bad_values(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = load_proxy_config()
    assert [product.etf for product in loaded.products] == ["0193L0", "0193T0", "0193W0", "0197X0"]
    assert loaded.execution == "close"

    def _bad(mutate: object) -> None:
        data = copy.deepcopy(dict(load_config("arena")))
        mutate(data["proxy"])  # type: ignore[operator]
        monkeypatch.setattr(proxy, "load_config", lambda name: data)
        with pytest.raises(ConfigError):
            load_proxy_config()

    def _product(key: str, **changes: object) -> object:
        return lambda section: section["products"][key].update(changes)

    def _set(**changes: object) -> object:
        return lambda section: section.update(changes)

    bad: list[object] = [
        lambda section: section.update(extra=1),
        _product("0193T0", extra=1),
        _product("0193T0", multiple=0.0),
        _product("0193T0", multiple=True),
        _product("0193T0", multiple="2"),
        _product("0193T0", multiple=float("nan")),
        _product("0193T0", underlying="0006"),
        _product("0193T0", underlying=6),
        _set(products={"abc": {"underlying": UNDER, "multiple": 2.0}}),
        _set(products={"0193T0": 1.0}),
        _set(products=[]),
        _set(products={}),
        _set(min_close_corr=1.5),
        _set(min_close_corr=True),
        _set(execution="vwap"),
        _set(tag=""),
        _set(drag_bps_per_session=-1.0),
        _set(max_abs_session_return=1.0),
        _set(min_overlap_sessions=True),
        _set(min_overlap_sessions=1),
        _set(max_close_te_bps=0.0),
    ]
    for mutate in bad:
        _bad(mutate)

    monkeypatch.setattr(proxy, "load_config", lambda name: {})
    with pytest.raises(ConfigError):
        load_proxy_config()


def test_base_price_handles_split() -> None:
    # Pre-split close 250000 on 50:1; today's close 5100 against an adjusted base of 5000.
    split = _raw([250_000.0, 5_100.0], [0.0, 100.0])
    out = underlying_returns(split, UNDER, CFG)
    assert out["traded"].to_list() == [True, True]
    assert out["r_cc"][1] == pytest.approx(5_100.0 / 5_000.0 - 1.0, rel=1e-12)
    assert out["r_cc"][1] != pytest.approx(5_100.0 / 250_000.0 - 1.0)
    assert out.columns == ["date", "traded", "r_cc", "r_on", "r_hi", "r_lo", "trading_value"]


def test_price_limit_breach_fails_closed() -> None:
    breach = _raw([100.0, 135.0], [0.0, 35.0])
    with pytest.raises(ProxyError, match="price limit"):
        underlying_returns(breach, UNDER, CFG)

    untraded_move = _raw([100.0, 200.0], [0.0, 100.0], volume=[1000, 0])
    out = underlying_returns(untraded_move, UNDER, CFG)
    assert out["traded"].to_list() == [True, False]
    assert out["r_cc"][1] is None

    duplicated = pl.concat([_raw([100.0], [0.0]), _raw([100.0], [0.0])])
    with pytest.raises(ProxyError, match="duplicate"):
        underlying_returns(duplicated, UNDER, CFG)
    with pytest.raises(ProxyError, match="no stock rows"):
        underlying_returns(_raw([100.0], [0.0]), "005930", CFG)
    with pytest.raises(ProxyError, match="null trading_value"):
        underlying_returns(_raw([100.0], [0.0], trading_value=[None]), UNDER, CFG)


def test_untraded_underlying_day_is_untraded_proxy_row() -> None:
    r_cc, r_on = _returns()
    closes, opens = _prices(r_cc, r_on)
    stock = _stock(closes, opens, zero_volume=(20,))
    rows = proxy_rows(_table(PRODUCT, r_cc, r_on), stock, PRODUCT, CFG)
    by_date = {row["date"]: row for row in rows.iter_rows(named=True)}
    gap = by_date[DAYS[20]]
    assert gap["open"] is None
    assert gap["high"] is None
    assert gap["low"] is None
    assert gap["volume"] == 0
    assert gap["trading_value"] == 0
    assert gap["close"] == pytest.approx(by_date[DAYS[19]]["close"], rel=1e-12)


def test_proxy_returns_follow_multiple_minus_drag() -> None:
    r_cc, r_on = _returns()
    closes, opens = _prices(r_cc, r_on)
    stock = _stock(closes, opens)
    prev = np.concatenate([closes[:1], closes[:-1]])
    r_hi = np.maximum(opens, closes) * 1.01 / prev - 1.0
    r_lo = np.minimum(opens, closes) * 0.99 / prev - 1.0
    for product in (PRODUCT, INVERSE):
        rows = proxy_rows(_table(product, r_cc, r_on), stock, product, CFG)
        m = product.multiple
        assert rows.height == LISTING
        close = rows["close"].to_numpy()
        high = rows["high"].to_numpy()
        low = rows["low"].to_numpy()
        opened = rows["open"].to_numpy()
        np.testing.assert_allclose(close[1:] / close[:-1], 1.0 + m * r_cc[1:LISTING] - DRAG_BPS / 1e4, rtol=1e-12)
        np.testing.assert_allclose(opened[1:] / close[:-1], 1.0 + m * r_on[1:LISTING], rtol=1e-9)
        assert np.all(high >= low)
        swapped_hi, swapped_lo = (r_hi, r_lo) if m > 0 else (r_lo, r_hi)
        np.testing.assert_allclose(high[1:] / close[:-1], 1.0 + m * swapped_hi[1:LISTING], rtol=1e-9)
        np.testing.assert_allclose(low[1:] / close[:-1], 1.0 + m * swapped_lo[1:LISTING], rtol=1e-9)


def test_anchor_only_scales_levels() -> None:
    r_cc, r_on = _returns()
    closes, opens = _prices(r_cc, r_on)
    stock = _stock(closes, opens)
    table = _table(PRODUCT, r_cc, r_on)
    factor = 1.7
    scaled_table = table.with_columns(
        close=pl.when((pl.col("ticker") == PRODUCT.etf) & (pl.col("date") == DAYS[LISTING]))
        .then(pl.col("close") * factor)
        .otherwise(pl.col("close"))
    )
    base = proxy_rows(table, stock, PRODUCT, CFG)
    scaled = proxy_rows(scaled_table, stock, PRODUCT, CFG)
    np.testing.assert_allclose(scaled["close"].to_numpy(), factor * base["close"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(scaled["open"].to_numpy(), factor * base["open"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(scaled["high"].to_numpy(), factor * base["high"].to_numpy(), rtol=1e-12)
    base_close = base["close"].to_numpy()
    scaled_close = scaled["close"].to_numpy()
    np.testing.assert_allclose(scaled_close[1:] / scaled_close[:-1], base_close[1:] / base_close[:-1], rtol=1e-12)


def test_listing_day_panel_return_is_proxy_return() -> None:
    r_cc, r_on = _returns()
    closes, opens = _prices(r_cc, r_on)
    stock = _stock(closes, opens)
    table = _table(PRODUCT, r_cc, r_on)
    spliced, _ = splice_proxy(table, stock, CFG)
    panel = (
        build_panel(spliced, [PRODUCT.etf], DAYS[0], DAYS[-1], PANEL_CFG)
        .frame.filter(pl.col("ticker") == PRODUCT.etf)
        .sort("date")
    )
    panel_r = panel["r_cc"].to_numpy()
    listing_return = PRODUCT.multiple * r_cc[LISTING] - DRAG_BPS / 1e4
    assert panel_r[LISTING] == pytest.approx(listing_return, rel=1e-9)
    etf_close = table.filter(pl.col("ticker") == PRODUCT.etf).sort("date")["close"].to_numpy()
    np.testing.assert_allclose(panel_r[LISTING + 1 :], etf_close[1:] / etf_close[:-1] - 1.0, rtol=1e-9)


def test_capacity_is_underlying_value_over_multiple() -> None:
    r_cc, r_on = _returns()
    for volume in (1000, 1):
        closes, opens = _prices(r_cc, r_on)
        stock = _stock(closes, opens, volume=volume)
        tv_by_date = stock.select("date", "trading_value")
        for product in (PRODUCT, INVERSE):
            rows = proxy_rows(_table(product, r_cc, r_on), stock, product, CFG).join(tv_by_date, on="date", suffix="_u")
            expected_tv = np.rint(rows["trading_value_u"].to_numpy() / abs(product.multiple))
            np.testing.assert_array_equal(rows["trading_value"].to_numpy(), expected_tv)
            close = rows["close"].to_numpy()
            expected_volume = np.maximum(1.0, np.rint(expected_tv / close))
            assert np.all(rows["volume"].to_numpy() >= 1)
            np.testing.assert_array_equal(rows["volume"].to_numpy(), expected_volume)


def test_calibration_fails_closed() -> None:
    r_cc, r_on = _returns()
    closes, opens = _prices(r_cc, r_on)
    stock = _stock(closes, opens)
    table = _table(PRODUCT, r_cc, r_on)
    fit = calibrate(table, stock, PRODUCT, CFG)
    stats = (fit.close_corr, fit.close_te_bps, fit.close_mean_bps, fit.open_corr, fit.open_te_bps, fit.open_mean_bps)
    assert all(math.isfinite(value) for value in stats)
    assert fit.close_corr >= 0.99
    assert fit.overlap_sessions == N - LISTING - 1
    assert fit.listed_first == DAYS[LISTING]
    assert fit.proxy_first is None and fit.proxy_sessions == 0

    uncorrelated = _table(PRODUCT, r_cc, r_on, follow=False)
    with pytest.raises(ProxyError, match="close_corr"):
        calibrate(uncorrelated, stock, PRODUCT, CFG)
    with pytest.raises(ProxyError, match="overlap"):
        calibrate(table, stock, PRODUCT, replace(CFG, min_overlap_sessions=N))
    with pytest.raises(ProxyError, match="close_te_bps"):
        calibrate(table, stock, PRODUCT, replace(CFG, max_close_te_bps=0.001))

    flat = table.with_columns(
        close=pl.when(pl.col("ticker") == PRODUCT.etf).then(1000.0).otherwise(pl.col("close")),
        open=pl.when(pl.col("ticker") == PRODUCT.etf).then(1000.0).otherwise(pl.col("open")),
    )
    with pytest.raises(ProxyError, match="not finite"):
        calibrate(flat, stock, PRODUCT, CFG)
    duplicated = pl.concat([table, table.filter(pl.col("ticker") == PRODUCT.etf).head(1)])
    with pytest.raises(ProxyError, match="duplicate ETF rows"):
        calibrate(duplicated, stock, PRODUCT, CFG)


def test_underlying_gap_fails_closed() -> None:
    r_cc, r_on = _returns()
    closes, opens = _prices(r_cc, r_on)
    table = _table(PRODUCT, r_cc, r_on)

    gap = _stock(closes, opens).filter(pl.col("date") != DAYS[20])
    with pytest.raises(ProxyError, match="no underlying row"):
        proxy_rows(table, gap, PRODUCT, CFG)

    untraded_listing = _stock(closes, opens, zero_volume=(LISTING,))
    with pytest.raises(ProxyError, match="not traded"):
        proxy_rows(table, untraded_listing, PRODUCT, CFG)

    crash_r = r_cc.copy()
    crash_r[20] = -0.25
    crash_closes, crash_opens = _prices(crash_r, r_on)
    with pytest.raises(ProxyError, match="growth"):
        proxy_rows(table, _stock(crash_closes, crash_opens), replace(PRODUCT, multiple=4.0), CFG)

    gap_on = r_on.copy()
    gap_on[20] = -0.26
    gap_closes, gap_opens = _prices(r_cc, gap_on)
    with pytest.raises(ProxyError, match="price must be"):
        proxy_rows(table, _stock(gap_closes, gap_opens), replace(PRODUCT, multiple=4.0), CFG)

    zero_listing = table.with_columns(
        close=pl.when((pl.col("ticker") == PRODUCT.etf) & (pl.col("date") == DAYS[LISTING]))
        .then(0.0)
        .otherwise(pl.col("close"))
    )
    with pytest.raises(ProxyError, match="listing close"):
        proxy_rows(zero_listing, _stock(closes, opens), PRODUCT, CFG)


def test_splice_keeps_listed_rows_and_other_tickers() -> None:
    r_cc, r_on = _returns()
    stock = _stock(*_prices(r_cc, r_on))
    table = _table(PRODUCT, r_cc, r_on)
    spliced, calibrations = splice_proxy(table, stock, CFG)

    listed = spliced.join(table, on=["ticker", "date"], how="semi").sort(["ticker", "date"])
    assert listed.equals(table.sort(["ticker", "date"]))
    proxy_only = spliced.join(table, on=["ticker", "date"], how="anti")
    assert proxy_only.height == LISTING
    assert set(proxy_only["date"].to_list()) <= set(table["date"].to_list())
    assert not spliced.select("date", "ticker").is_duplicated().any()
    assert calibrations[0].proxy_first == DAYS[0]
    assert calibrations[0].proxy_last == DAYS[LISTING - 1]
    assert calibrations[0].proxy_sessions == LISTING

    listed_first = _table(PRODUCT, r_cc, r_on, listing=0)
    _, no_history = splice_proxy(listed_first, stock, CFG)
    assert no_history[0].proxy_first is None
    assert no_history[0].proxy_last is None
    assert no_history[0].proxy_sessions == 0

    with pytest.raises(ProxyError, match="no ETF rows"):
        splice_proxy(table.filter(pl.col("ticker") != PRODUCT.etf), stock, CFG)
    with pytest.raises(ProxyError, match="duplicate"):
        splice_proxy(table, stock, replace(CFG, products=(PRODUCT, PRODUCT)))


def test_missing_silver_raises_panel_source_error(tmp_path: Path) -> None:
    paths = DataPaths(root=tmp_path)
    with pytest.raises(PanelSourceError):
        load_stock_silver(paths, PANEL_CFG, [UNDER])

    source = paths.silver("stock_daily")
    source.parent.mkdir(parents=True)
    _raw([100.0], [0.0]).drop("change_abs").write_parquet(source)
    with pytest.raises(PanelSourceError, match="change_abs"):
        load_stock_silver(paths, PANEL_CFG, [UNDER])

    stored = pl.concat([_raw([100.0, 101.0], [0.0, 1.0]), _raw([50.0], [0.0], ticker="005930")])
    stored.write_parquet(source)
    loaded = load_stock_silver(paths, PANEL_CFG, [UNDER, UNDER])
    assert loaded.columns == list(STOCK_SCHEMA)
    assert loaded["ticker"].unique().to_list() == [UNDER]
    assert loaded.height == 2
