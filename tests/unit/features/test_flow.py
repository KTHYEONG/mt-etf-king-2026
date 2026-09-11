from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.features.flow import add_flow, decompose_aum_change
from tests.unit.features.conftest import session_dates


def test_add_flow_phantom_session_does_not_null_propagate_flow_and_volume_expansion() -> None:
    """A market-wide phantom session must not null out flow_{w}d / volume_expansion for the
    next w real trading sessions (rolling_sum/rolling_mean with min_samples=w counts physical
    rows, same defect class as add_trend/add_volatility). `creation_flow_krw` is supplied
    directly (not derived via decompose_aum_change) so the assertion isolates the rolling-window
    fix from decompose_aum_change's own separate shift(1)-across-the-phantom-row nullity."""
    sessions = session_dates(date(2026, 1, 2), 15)
    rows: list[dict[str, object]] = []
    for i, d in enumerate(sessions):
        phantom = i == 10
        rows.append(
            {
                "date": d,
                "ticker": "A",
                "close": None if phantom else 100.0,
                "shares_outstanding": 1_000_000,
                "creation_flow_krw": None if phantom else 1_000.0,
                "trading_value": None if phantom else 1_000_000.0,
                "net_assets": 100_000_000,
            }
        )
    frame = pl.DataFrame(rows)

    out = add_flow(frame, [3, 5], date(2026, 12, 31), key="ticker")

    flow3_next = out.filter(pl.col("date") == sessions[11]).select("flow_3d").item()
    assert flow3_next == pytest.approx(3_000.0)
    vol_exp_next = out.filter(pl.col("date") == sessions[11]).select("volume_expansion").item()
    assert vol_exp_next == pytest.approx(1.0)


def test_scenario_05_04_decompose_aum_change_identity() -> None:
    """SCENARIO-05-04"""
    d0, d1 = session_dates(date(2026, 8, 1), 2)
    shares0, shares1 = 12_600_000, 13_000_000
    nav0, nav1 = 35_057.03, 35_500.00
    frame = pl.DataFrame(
        {
            "date": [d0, d1],
            "ticker": ["A", "A"],
            "shares_outstanding": [shares0, shares1],
            "nav": [nav0, nav1],
            "net_assets": [shares0 * nav0, shares1 * nav1],
        }
    )
    out = decompose_aum_change(frame, date(2026, 12, 31), key="ticker")
    row = out.filter(pl.col("date") == d1)
    creation = row.select("creation_flow_krw").item()
    perf = row.select("performance_effect").item()
    expected_creation = (shares1 - shares0) * nav1
    expected_perf = shares0 * (nav1 - nav0)
    assert creation == pytest.approx(expected_creation)
    assert perf == pytest.approx(expected_perf)
    net_delta = shares1 * nav1 - shares0 * nav0
    assert creation + perf == pytest.approx(net_delta, rel=1e-9)


def test_scenario_05_05_add_flow_guards() -> None:
    """SCENARIO-05-05"""
    sessions = session_dates(date(2026, 1, 2), 25)
    rows: list[dict[str, object]] = []
    for i, d in enumerate(sessions):
        rows.append(
            {
                "date": d,
                "ticker": "A",
                "close": 100.0,
                "nav": 100.0 if i != 5 else None,
                "shares_outstanding": 1_000_000,
                "net_assets": 0 if i == 10 else 100_000_000,
                "trading_value": 0,
            }
        )
    frame = pl.DataFrame(rows)
    out = add_flow(frame, [5, 20], date(2026, 12, 31), key="ticker")
    normal = out.filter(pl.col("date") == sessions[24])
    assert normal.select("disparity").item() == pytest.approx(0.0)
    null_nav = out.filter(pl.col("date") == sessions[5]).select("disparity").item()
    assert null_nav is None
    zero_nav = out.with_columns(pl.when(pl.col("date") == sessions[6]).then(0.0).otherwise(pl.col("nav")).alias("nav"))
    zero_nav = add_flow(zero_nav, [5, 20], date(2026, 12, 31), key="ticker")
    assert zero_nav.filter(pl.col("date") == sessions[6]).select("disparity").item() is None
    assert out.filter(pl.col("date") == sessions[10]).select("turnover").item() is None
    assert out.filter(pl.col("date") == sessions[24]).select("volume_expansion").item() is None


def test_add_flow_no_optional_columns_returns_frame_unchanged() -> None:
    sessions = session_dates(date(2026, 1, 2), 5)
    frame = pl.DataFrame({"date": sessions, "ticker": ["A"] * 5, "close": [100.0] * 5})
    out = add_flow(frame, [3], date(2026, 12, 31), key="ticker")
    assert out.equals(frame.sort(["ticker", "date"]))
