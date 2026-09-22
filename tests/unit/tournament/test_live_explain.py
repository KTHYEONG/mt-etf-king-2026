"""Invariant guards for live decision explanation."""

from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

from src.tournament.live_explain import (
    AnchorMonitorRow,
    LiveAction,
    LiveReason,
    build_anchor_monitor,
    derive_live_action,
    explain_live_decision,
    explanation_to_payload,
    next_trading_session,
)


def test_derive_live_action_covers_all_transitions() -> None:
    """Action mapping covers the 5 (prior, new) shapes."""
    assert derive_live_action(None, "122630") is LiveAction.BUY
    assert derive_live_action("122630", None) is LiveAction.SELL_ALL
    assert derive_live_action("122630", "233740") is LiveAction.SWITCH
    assert derive_live_action("122630", "122630") is LiveAction.HOLD
    assert derive_live_action(None, None) is LiveAction.STAY_CASH


def _anchor_panel() -> pl.DataFrame:
    d = date(2026, 9, 21)
    return pl.DataFrame(
        {
            "date": [d, d],
            "ticker": ["122630", "233740"],
            "name": ["KODEX 레버리지", "KODEX 코스닥150레버리지"],
            "close": [114060.0, 10000.0],
            "mom_20": [0.05, 0.10],
            "drawdown_20": [-0.0132, -0.01],
        },
        schema={
            "date": pl.Date,
            "ticker": pl.String,
            "name": pl.String,
            "close": pl.Float64,
            "mom_20": pl.Float64,
            "drawdown_20": pl.Float64,
        },
    )


def test_build_anchor_monitor_derives_stop_close() -> None:
    """Given or-vps 9/21 values; roll_max_20 ~= 115580, stop_close ~= 98243."""
    rows = build_anchor_monitor(
        _anchor_panel(),
        decision_date=date(2026, 9, 21),
        anchor_tickers=("122630",),
        stop_drawdown=0.15,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.roll_max_20 == pytest.approx(115580, rel=1e-3)
    assert row.stop_close == pytest.approx(98243, rel=1e-3)
    assert row.stopped is False


def test_build_anchor_monitor_inclusive_stop_boundary() -> None:
    """drawdown_20 == -0.15 is stopped (inclusive boundary)."""
    d = date(2026, 9, 21)
    panel = pl.DataFrame(
        {
            "date": [d],
            "ticker": ["122630"],
            "close": [100.0],
            "mom_20": [0.01],
            "drawdown_20": [-0.15],
        },
        schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64, "mom_20": pl.Float64, "drawdown_20": pl.Float64},
    )
    (row,) = build_anchor_monitor(panel, decision_date=d, anchor_tickers=("122630",), stop_drawdown=0.15)
    assert row.stopped is True


def test_build_anchor_monitor_omits_non_finite_rows() -> None:
    """NaN dd20 anchor is absent, the other is present."""
    d = date(2026, 9, 21)
    panel = pl.DataFrame(
        {
            "date": [d, d],
            "ticker": ["122630", "233740"],
            "close": [100.0, 100.0],
            "mom_20": [0.01, 0.02],
            "drawdown_20": [float("nan"), -0.01],
        },
        schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64, "mom_20": pl.Float64, "drawdown_20": pl.Float64},
    )
    rows = build_anchor_monitor(panel, decision_date=d, anchor_tickers=("122630", "233740"), stop_drawdown=0.15)
    assert [r.ticker for r in rows] == ["233740"]


@pytest.mark.parametrize("bad", [0, 1, True])
def test_build_anchor_monitor_rejects_bad_stop(bad: object) -> None:
    """stop 0, 1, True raise ValueError."""
    with pytest.raises(ValueError, match="stop_drawdown must be in"):
        build_anchor_monitor(
            _anchor_panel(),
            decision_date=date(2026, 9, 21),
            anchor_tickers=("122630",),
            stop_drawdown=bad,  # type: ignore[arg-type]
        )


def _row(ticker: str, *, stopped: bool, mom_20: float = 0.05, drawdown_20: float = -0.01) -> AnchorMonitorRow:
    return AnchorMonitorRow(
        ticker=ticker,
        name=ticker,
        close=100.0,
        mom_20=mom_20,
        drawdown_20=-0.20 if stopped else drawdown_20,
        roll_max_20=110.0,
        stop_close=93.5,
        stopped=stopped,
    )


def test_explain_latch_hold() -> None:
    """Prior 122630, new 122630, INACTIVE -> HOLD / ANCHOR_LATCH_HOLD."""
    exp = explain_live_decision(
        decision_date=date(2026, 9, 21),
        execution_date=date(2026, 9, 28),
        sleeve="INACTIVE",
        prior_held="122630",
        prior_weight=0.95,
        new_held="122630",
        new_weight=0.95,
        anchors=(_row("122630", stopped=False),),
        anchor_tickers=("122630", "233740"),
    )
    assert exp.action is LiveAction.HOLD
    assert exp.reason is LiveReason.ANCHOR_LATCH_HOLD


def test_explain_stop_switch() -> None:
    """Prior 122630 stopped, new 233740 -> SWITCH / ANCHOR_STOP_SWITCH."""
    exp = explain_live_decision(
        decision_date=date(2026, 9, 21),
        execution_date=None,
        sleeve="CRASH_REBOUND",
        prior_held="122630",
        prior_weight=0.95,
        new_held="233740",
        new_weight=0.95,
        anchors=(_row("122630", stopped=True), _row("233740", stopped=False)),
        anchor_tickers=("122630", "233740"),
    )
    assert exp.action is LiveAction.SWITCH
    assert exp.reason is LiveReason.ANCHOR_STOP_SWITCH


def test_explain_stop_cash() -> None:
    """Prior 122630 stopped, new None -> SELL_ALL / ANCHOR_STOP_CASH."""
    exp = explain_live_decision(
        decision_date=date(2026, 9, 21),
        execution_date=None,
        sleeve="CRASH_REBOUND",
        prior_held="122630",
        prior_weight=0.95,
        new_held=None,
        new_weight=0.0,
        anchors=(_row("122630", stopped=True),),
        anchor_tickers=("122630", "233740"),
    )
    assert exp.action is LiveAction.SELL_ALL
    assert exp.reason is LiveReason.ANCHOR_STOP_CASH


def test_explain_fresh_pick() -> None:
    """Prior None, new 233740, CRASH_REBOUND -> BUY / ANCHOR_PICK."""
    exp = explain_live_decision(
        decision_date=date(2026, 9, 21),
        execution_date=None,
        sleeve="CRASH_REBOUND",
        prior_held=None,
        prior_weight=0.0,
        new_held="233740",
        new_weight=0.95,
        anchors=(_row("233740", stopped=False),),
        anchor_tickers=("122630", "233740"),
    )
    assert exp.action is LiveAction.BUY
    assert exp.reason is LiveReason.ANCHOR_PICK


def test_explain_stay_cash() -> None:
    """Prior None, new None -> STAY_CASH / NO_ANCHOR_CASH."""
    exp = explain_live_decision(
        decision_date=date(2026, 9, 21),
        execution_date=None,
        sleeve="UNCERTAIN",
        prior_held=None,
        prior_weight=0.0,
        new_held=None,
        new_weight=0.0,
        anchors=(),
        anchor_tickers=("122630", "233740"),
    )
    assert exp.action is LiveAction.STAY_CASH
    assert exp.reason is LiveReason.NO_ANCHOR_CASH


def test_next_session_skips_holidays() -> None:
    """Calendar stub sessions after 9/23 start at 9/28 -> 9/28; empty horizon -> None."""

    class _Stub:
        def sessions(self, start: date, end: date) -> list[date]:
            all_sessions = [date(2026, 9, 28), date(2026, 9, 29)]
            return [s for s in all_sessions if start <= s <= end]

    assert next_trading_session(date(2026, 9, 23), _Stub()) == date(2026, 9, 28)  # type: ignore[arg-type]

    class _Empty:
        def sessions(self, start: date, end: date) -> list[date]:
            return []

    assert next_trading_session(date(2026, 9, 23), _Empty()) is None  # type: ignore[arg-type]


def test_explanation_payload_is_json_safe() -> None:
    """Payload serializes with ISO dates."""
    exp = explain_live_decision(
        decision_date=date(2026, 9, 21),
        execution_date=date(2026, 9, 28),
        sleeve="CRASH_REBOUND",
        prior_held="122630",
        prior_weight=0.95,
        new_held="122630",
        new_weight=0.95,
        anchors=(_row("122630", stopped=False),),
        anchor_tickers=("122630", "233740"),
    )
    payload = explanation_to_payload(exp)
    text = json.dumps(payload, ensure_ascii=False)
    assert isinstance(text, str)
    assert payload["execution_date"] == "2026-09-28"
    assert isinstance(payload["reason_ko"], str) and payload["reason_ko"]


def test_build_anchor_monitor_empty_tickers_returns_empty() -> None:
    """Empty anchor_tickers yields no rows without touching the panel."""
    assert (
        build_anchor_monitor(
            _anchor_panel(),
            decision_date=date(2026, 9, 21),
            anchor_tickers=(),
            stop_drawdown=0.15,
        )
        == ()
    )


def test_build_anchor_monitor_ignores_unusable_panels() -> None:
    """Non-frame, empty, or key-column-less panels yield no rows."""
    assert (
        build_anchor_monitor(
            pl.DataFrame({"a": [1.0]}),
            decision_date=date(2026, 9, 21),
            anchor_tickers=("122630",),
            stop_drawdown=0.15,
        )
        == ()
    )
    assert (
        build_anchor_monitor(
            pl.DataFrame(
                {"date": [], "ticker": [], "close": [], "mom_20": [], "drawdown_20": []},
                schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64, "mom_20": pl.Float64, "drawdown_20": pl.Float64},
            ),
            decision_date=date(2026, 9, 21),
            anchor_tickers=("122630",),
            stop_drawdown=0.15,
        )
        == ()
    )
    assert (
        build_anchor_monitor(
            pl.DataFrame({"close": [1.0], "mom_20": [0.1], "drawdown_20": [-0.01]}),
            decision_date=date(2026, 9, 21),
            anchor_tickers=("122630",),
            stop_drawdown=0.15,
        )
        == ()
    )
    assert (
        build_anchor_monitor(
            pl.DataFrame(
                {"date": [date(2026, 9, 20)], "ticker": ["122630"], "close": [1.0]},
                schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64},
            ),
            decision_date=date(2026, 9, 21),
            anchor_tickers=("122630",),
            stop_drawdown=0.15,
        )
        == ()
    )
    assert (
        build_anchor_monitor(
            _anchor_panel(),
            decision_date=date(2026, 9, 20),
            anchor_tickers=("122630",),
            stop_drawdown=0.15,
        )
        == ()
    )


def test_build_anchor_monitor_skips_absent_and_broken_anchors() -> None:
    """Tickers missing that day, bool/non-numeric fields, or 1+dd<=0 are omitted."""
    d = date(2026, 9, 21)
    panel = pl.DataFrame(
        {
            "date": [d, d, d],
            "ticker": ["233740", "069500", "114800"],
            "close": [100.0, 100.0, 100.0],
            "mom_20": [0.02, 0.02, 0.02],
            "drawdown_20": [-0.01, -0.01, -1.0],
        },
        schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64, "mom_20": pl.Float64, "drawdown_20": pl.Float64},
    )
    rows = build_anchor_monitor(panel, decision_date=d, anchor_tickers=("122630", "233740", "069500", "114800"), stop_drawdown=0.15)
    assert [r.ticker for r in rows] == ["233740", "069500"]
    bool_frame = pl.DataFrame(
        {
            "date": [d],
            "ticker": ["069500"],
            "close": [100.0],
            "mom_20": [True],
            "drawdown_20": [-0.01],
        },
        strict=False,
    ).with_columns(pl.col("date").cast(pl.Date))
    assert build_anchor_monitor(bool_frame, decision_date=d, anchor_tickers=("069500",), stop_drawdown=0.15) == ()
    str_frame = pl.DataFrame(
        {
            "date": [d],
            "ticker": ["069500"],
            "close": ["100"],
            "mom_20": [0.02],
            "drawdown_20": [-0.01],
        },
        strict=False,
    ).with_columns(pl.col("date").cast(pl.Date))
    assert build_anchor_monitor(str_frame, decision_date=d, anchor_tickers=("069500",), stop_drawdown=0.15) == ()


def test_explain_unknown_tickers_fall_through() -> None:
    """Tickers outside the anchor set follow the generic branches."""
    mom = explain_live_decision(
        decision_date=date(2026, 9, 21),
        execution_date=None,
        sleeve="LOTTERY_ON",
        prior_held=None,
        prior_weight=0.0,
        new_held="069500",
        new_weight=0.5,
        anchors=(),
        anchor_tickers=("122630",),
    )
    assert mom.action is LiveAction.BUY
    assert mom.reason is LiveReason.MOM60_LEADER
    other = explain_live_decision(
        decision_date=date(2026, 9, 21),
        execution_date=None,
        sleeve="CRASH_REBOUND",
        prior_held=None,
        prior_weight=0.0,
        new_held="069500",
        new_weight=0.5,
        anchors=(),
        anchor_tickers=("122630",),
    )
    assert other.reason is LiveReason.OTHER
    cash = explain_live_decision(
        decision_date=date(2026, 9, 21),
        execution_date=None,
        sleeve="UNCERTAIN",
        prior_held="122630",
        prior_weight=0.95,
        new_held=None,
        new_weight=0.0,
        anchors=(),
        anchor_tickers=("122630",),
    )
    assert cash.reason is LiveReason.NO_ANCHOR_CASH
