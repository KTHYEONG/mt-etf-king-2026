"""Invariant guards for the contest vehicle panel."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.contest.panel import PanelGapError, VehiclePanel, build_vehicle_panel, neutralize_drift

START = date(2026, 8, 3)
N = 80

K200 = "코스피 200"
KOSPI = "코스피"
IT = "코스피 200 정보기술"

VEHICLES: dict[str, Any] = {
    "K2": {"ticker": "T_K2", "returns": "listed", "tradable": True, "crowd_only": False},
    "SEMI2": {
        "ticker": "T_SEMI",
        "returns": "listed",
        "synthetic": {"underlying": IT, "multiple": 2.0},
        "tradable": True,
        "crowd_only": False,
    },
    "HY2": {
        "ticker": "T_HY",
        "returns": "listed",
        "synthetic": {"underlying": "HYNIX", "multiple": 2.0},
        "tradable": True,
        "crowd_only": False,
    },
    "K2I": {
        "ticker": "T_K2I",
        "returns": "synthetic",
        "synthetic": {"underlying": K200, "multiple": -2.0},
        "tradable": True,
        "crowd_only": False,
    },
    "KOSPI": {
        "returns": "synthetic",
        "synthetic": {"underlying": KOSPI, "multiple": 1.0},
        "tradable": False,
        "crowd_only": False,
    },
}


def _grid(n: int = N) -> list[date]:
    return [START + timedelta(days=i) for i in range(n)]


def _drift_frame(dates: list[date], key: str, val: str, base: float, step: float) -> pl.DataFrame:
    opens = [base + step * i for i in range(len(dates))]
    closes = [o * (1.0 + 0.001 * (1 if i % 2 == 0 else -1)) for i, o in enumerate(opens)]
    return pl.DataFrame(
        {"date": dates, key: [val] * len(dates), "open": opens, "close": closes},
        schema={"date": pl.Date, key: pl.String, "open": pl.Float64, "close": pl.Float64},
    )


def _toy_inputs(listed_from: int = 5) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    dates = _grid()
    index_daily = pl.concat(
        [
            _drift_frame(dates, "index_name", K200, 1000.0, 1.0),
            _drift_frame(dates, "index_name", KOSPI, 7000.0, 5.0),
            _drift_frame(dates, "index_name", IT, 500.0, 0.8),
        ]
    )
    etf_rows = [_drift_frame(dates, "ticker", "T_K2", 100.0, 0.2)]
    etf_rows.append(_drift_frame(dates[listed_from:], "ticker", "T_SEMI", 50.0, 0.15))
    etf_rows.append(_drift_frame(dates[listed_from:], "ticker", "T_HY", 30.0, 0.1))
    etf_rows.append(_drift_frame(dates, "ticker", "T_K2I", 67.0, 0.0))
    etf_daily = pl.concat(etf_rows)
    ref_dates = dates[:listed_from]
    reference = pl.concat(
        [
            _drift_frame(ref_dates, "symbol", "HYNIX", 200.0, 0.5),
            _drift_frame(ref_dates, "symbol", "SAMSUNG", 80.0, 0.1),
        ]
    )
    return etf_daily, index_daily, reference


def _toy_panel(**kwargs: Any) -> VehiclePanel:
    etf_daily, index_daily, reference = _toy_inputs(**kwargs)
    return build_vehicle_panel(etf_daily, index_daily, reference, VEHICLES, 0.01)


def test_build_listed_legs_override_synthetic_after_listing() -> None:
    """Rows before listing equal 2x underlying minus fee; rows after equal listed legs."""
    etf_daily, index_daily, reference = _toy_inputs(listed_from=5)
    panel = build_vehicle_panel(etf_daily, index_daily, reference, VEHICLES, 0.01)
    j = panel.index("SEMI2")
    it = index_daily.filter(pl.col("index_name") == IT).sort("date")
    o = it.get_column("open").to_numpy().astype(float)
    c = it.get_column("close").to_numpy().astype(float)
    fee = 0.01 / 252.0
    for r in range(1, 5):
        assert panel.gap[r, j] == pytest.approx(2.0 * (o[r] / c[r - 1] - 1.0))
        assert panel.intraday[r, j] == pytest.approx(2.0 * (c[r] / o[r] - 1.0) - fee)
    listed = etf_daily.filter(pl.col("ticker") == "T_SEMI").sort("date")
    lo = listed.get_column("open").to_numpy().astype(float)
    lc = listed.get_column("close").to_numpy().astype(float)
    for k in range(1, len(lo)):
        r = 5 + k
        assert panel.gap[r, j] == pytest.approx(lo[k] / lc[k - 1] - 1.0)
        assert panel.intraday[r, j] == pytest.approx(lc[k] / lo[k] - 1.0)


def test_build_contest_period_gap_fails_closed() -> None:
    """A tradable vehicle missing one session after contest start raises PanelGapError."""
    etf_daily, index_daily, reference = _toy_inputs()
    missing_day = date(2026, 9, 22)
    etf_daily = etf_daily.filter(~((pl.col("ticker") == "T_K2") & (pl.col("date") == missing_day)))
    with pytest.raises(PanelGapError):
        build_vehicle_panel(etf_daily, index_daily, reference, VEHICLES, 0.01)


def test_build_k2i_always_synthetic() -> None:
    """K2I legs equal -2x KOSPI200 legs even with tick-quantized listed prices."""
    etf_daily, index_daily, reference = _toy_inputs()
    panel = build_vehicle_panel(etf_daily, index_daily, reference, VEHICLES, 0.01)
    j = panel.index("K2I")
    k200 = index_daily.filter(pl.col("index_name") == K200).sort("date")
    o = k200.get_column("open").to_numpy().astype(float)
    c = k200.get_column("close").to_numpy().astype(float)
    fee = 0.01 / 252.0
    for r in range(1, N):
        assert panel.gap[r, j] == pytest.approx(-2.0 * (o[r] / c[r - 1] - 1.0))
        assert panel.intraday[r, j] == pytest.approx(-2.0 * (c[r] / o[r] - 1.0) - fee)
    assert panel.gap[0, j] == 0.0


def test_neutralize_drift_keeps_realized_sessions() -> None:
    """Arithmetic mean close-to-close return over fit rows is ~0 while kept rows stay bitwise unchanged."""
    panel = _toy_panel()
    start = START + timedelta(days=10)
    end = START + timedelta(days=69)
    keeps = [end + timedelta(days=1), end + timedelta(days=2)]
    out = neutralize_drift(panel, start, end, keeps)
    rows = [panel.row(d) for d in panel.dates if start <= d <= end]
    g = out.gap[rows].astype(np.float64)
    o = out.intraday[rows].astype(np.float64)
    mean_cc = ((1 + g) * (1 + o) - 1.0).mean(axis=0)
    assert np.all(np.abs(mean_cc) < 1e-6)
    for s in keeps:
        r = panel.row(s)
        assert np.array_equal(out.intraday[r], panel.intraday[r])
        assert np.array_equal(out.gap[r], panel.gap[r])
    assert np.array_equal(out.log_nav, panel.log_nav)
    assert out.names == panel.names and out.dates == panel.dates


def test_neutralize_drift_long_inverse_pair_stays_coherent() -> None:
    """A long/inverse pair with exact-negative close-to-close returns stays exact-negative after neutralizing."""
    dates = _grid(40)
    n = len(dates)
    rng = np.random.default_rng(7)
    r = rng.uniform(-0.03, 0.04, n).astype(np.float64)
    r[0] = 0.0
    gap_l = np.zeros(n, dtype=np.float32)
    intra_l = np.zeros(n, dtype=np.float32)
    gap_i = np.zeros(n, dtype=np.float32)
    intra_i = np.zeros(n, dtype=np.float32)
    gap_l[1:] = 0.0
    intra_l[1:] = (r[1:] / 1.0).astype(np.float32)
    intra_i[1:] = (-r[1:]).astype(np.float32)
    gap = np.column_stack([gap_l, gap_i])
    intra = np.column_stack([intra_l, intra_i])
    cc = (1 + gap.astype(np.float64)) * (1 + intra.astype(np.float64)) - 1.0
    panel = VehiclePanel(
        dates=tuple(dates), names=("L", "I"), gap=gap, intraday=intra,
        log_nav=np.cumsum(np.log1p(np.clip(cc, -0.999999, None)), axis=0),
    )
    start, end = dates[1], dates[-1]
    out = neutralize_drift(panel, start, end, [])
    rows = [panel.row(d) for d in dates if start <= d <= end]
    g = out.gap[rows].astype(np.float64)
    o = out.intraday[rows].astype(np.float64)
    adj = (1 + g) * (1 + o) - 1.0
    assert np.all(np.abs(adj[:, 0] + adj[:, 1]) < 1e-6)
    assert np.all(np.log1p(np.clip(adj, -0.999999, None)).mean(axis=0) < 0)


def test_panel_row_and_index_lookup_errors() -> None:
    """Unknown sessions raise ValueError; unknown vehicles raise KeyError."""
    panel = _toy_panel()
    with pytest.raises(ValueError, match="session not in panel"):
        panel.row(date(2000, 1, 1))
    with pytest.raises(KeyError):
        panel.index("NOPE")
    assert panel.row(START) == 0
    assert panel.index("K2") == 0


def test_build_empty_grid_fails_closed() -> None:
    """An index table without KOSPI200 rows fails closed."""
    etf_daily, _, reference = _toy_inputs()
    index_daily = pl.DataFrame(
        {"date": [], "index_name": [], "open": [], "close": []},
        schema={"date": pl.Date, "index_name": pl.String, "open": pl.Float64, "close": pl.Float64},
    )
    with pytest.raises(ValueError, match=r"empty session grid"):
        build_vehicle_panel(etf_daily, index_daily, reference, VEHICLES, 0.01)


def test_build_unknown_underlying_fails_closed() -> None:
    """A synthetic vehicle with an unknown underlying fails closed."""
    etf_daily, index_daily, reference = _toy_inputs()
    bad = dict(VEHICLES)
    bad["MYST"] = {
        "returns": "synthetic",
        "synthetic": {"underlying": "no-such-index", "multiple": 2.0},
        "tradable": True,
        "crowd_only": False,
    }
    with pytest.raises(PanelGapError):
        build_vehicle_panel(etf_daily, index_daily, reference, bad, 0.01)


def test_build_vehicle_without_any_legs_fails_closed() -> None:
    """A vehicle with no ticker rows and no synthetic config fails closed."""
    etf_daily, index_daily, reference = _toy_inputs()
    bad = dict(VEHICLES)
    bad["GHOST"] = {"returns": "synthetic", "tradable": True, "crowd_only": False}
    with pytest.raises(PanelGapError):
        build_vehicle_panel(etf_daily, index_daily, reference, bad, 0.01)


def test_neutralize_empty_or_fully_kept_window_fails_closed() -> None:
    """Empty windows and fully-kept windows fail closed."""
    panel = _toy_panel()
    with pytest.raises(ValueError, match=r"empty neutralize window"):
        neutralize_drift(panel, date(2000, 1, 1), date(2000, 1, 2), [])
    start = START + timedelta(days=10)
    end = START + timedelta(days=12)
    keeps = [START + timedelta(days=i) for i in range(10, 13)]
    with pytest.raises(ValueError, match=r"fully covered"):
        neutralize_drift(panel, start, end, keeps)
