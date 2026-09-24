from __future__ import annotations

import random
from datetime import date

import polars as pl

from src.core.calendar import TradingCalendar
from src.data.validation import PanelValidator, Severity, classify_session, mark_tradability


def test_scenario_03_03_classify_session() -> None:
    """SCENARIO-03-03"""
    rows_holiday = [{"TDD_CLSPRC": ""} for _ in range(1163)]
    assert classify_session(rows_holiday, "TDD_CLSPRC") is False
    rows_trading = [{"TDD_CLSPRC": "100"} for _ in range(1163)]
    assert classify_session(rows_trading, "TDD_CLSPRC") is True
    rows_mix = [{"TDD_CLSPRC": "100"} for _ in range(582)] + [{"TDD_CLSPRC": ""} for _ in range(581)]
    assert classify_session(rows_mix, "TDD_CLSPRC", 0.5) is True
    assert classify_session(rows_mix, "TDD_CLSPRC", 0.51) is False


def test_scenario_03_04_validator_critical() -> None:
    """SCENARIO-03-04"""
    cal = TradingCalendar()
    validator = PanelValidator(cal)
    # clean two-session panel
    df = pl.DataFrame(
        {
            "date": [date(2026, 8, 13), date(2026, 8, 13), date(2026, 8, 14), date(2026, 8, 14)],
            "ticker": ["451060", "069500", "451060", "069500"],
            "close": [100.0, 200.0, 101.0, 201.0],
            "open": [99, 199, 100, 200],
            "high": [101, 201, 102, 202],
            "low": [98, 198, 99, 199],
            "trading_value": [1000000, 1000000, 1000000, 1000000],
            "nav": [100.0, 200.0, 101.0, 201.0],
            "market_cap": [1000, 2000, 1010, 2010],
            "shares_outstanding": [10, 10, 10, 10],
            "net_assets": [1000, 2000, 1010, 2010],
        }
    )
    report = validator.validate("etf_daily", df)
    assert not report.is_fatal()
    # non-session date
    df_nonsess = pl.DataFrame({"date": [date(2026, 8, 15)], "ticker": ["451060"], "close": [100.0], "market_cap": [1000], "shares_outstanding": [10]})
    report2 = validator.validate("etf_daily", df_nonsess)
    assert report2.is_fatal()
    assert any(i.severity == Severity.CRITICAL for i in report2.issues)
    # duplicate
    df_dup = pl.concat([df, df.slice(0, 1)])
    report3 = validator.validate("etf_daily", df_dup)
    assert report3.is_fatal()
    # market cap violation >0.1%
    rows = []
    for i in range(1000):
        close = 100.0
        shares = 1000
        mcap = close * shares if i >= 2 else close * shares * 2
        rows.append({"date": date(2026, 8, 13), "ticker": f"{i:06d}", "close": close, "market_cap": mcap, "shares_outstanding": shares, "nav": 100.0, "net_assets": 100000, "trading_value": 1000, "high": 101, "low": 99, "open": 100})
    df3 = pl.DataFrame(rows)
    report4 = validator.validate("etf_daily", df3)
    assert report4.is_fatal()


def test_scenario_03_05_mark_tradability_outlier() -> None:
    """SCENARIO-03-05"""
    random.seed(0)
    rows = []
    for _ in range(100):
        disp = random.gauss(0.003, 0.0005)
        close = 100 * (1 + disp)
        rows.append({"close": close, "nav": 100.0, "trading_value": 1000000, "high": close + 1, "low": close - 1, "open": close})
    rows.append({"close": 8535.0, "nav": 48.38, "trading_value": 0, "high": 8536.0, "low": 8534.0, "open": 8535.0})
    df = pl.DataFrame(rows)
    out = mark_tradability(df)
    assert out.height == 101
    false_count = out.filter(pl.col("is_tradable") == False).height  # noqa: E712
    assert false_count == 1


def test_scenario_03_06_mark_tradability_ohlc() -> None:
    """SCENARIO-03-06"""
    df = pl.DataFrame(
        [
            {"close": None, "nav": 100.0, "trading_value": 1000, "high": 101.0, "low": 99.0, "open": 100.0},
            {"close": 100.0, "nav": 100.0, "trading_value": 0, "high": 101.0, "low": 99.0, "open": 100.0},
            {"close": 100.0, "nav": 100.0, "trading_value": 1000, "high": 90.0, "low": 99.0, "open": 100.0},
            {"close": 200.0, "nav": 100.0, "trading_value": 1000, "high": 150.0, "low": 90.0, "open": 100.0},
        ]
    )
    out = mark_tradability(df)
    assert out.height == 4
    assert out.filter(pl.col("is_tradable") == False).height >= 3  # noqa: E712


def test_SCENARIO_03A_02_no_future_dates_gate() -> None:  # noqa: N802
    """SCENARIO-03A-02"""
    cal = TradingCalendar()
    validator = PanelValidator(cal, today=lambda: date(2026, 8, 28))
    df_ok = pl.DataFrame({"date": [date(2026, 8, 28)], "ticker": ["451060"], "close": [100.0]})
    report_ok = validator.validate("etf_daily", df_ok)
    assert not report_ok.is_fatal()
    df_future = pl.DataFrame({"date": [date(2026, 8, 29)], "ticker": ["451060"], "close": [100.0]})
    report_future = validator.validate("etf_daily", df_future)
    assert report_future.is_fatal()
    assert any(i.gate == "V10_no_future_dates" and i.severity == Severity.CRITICAL for i in report_future.issues)


globals()["test SCENARIO-03A-02"] = test_SCENARIO_03A_02_no_future_dates_gate  # noqa: E402, F401


def test_etf_index_stale_gap_is_detected() -> None:
    """An ETF date missing behind the index max is a fatal stale gap."""
    from src.data.validation import etf_index_session_gaps

    report = etf_index_session_gaps(
        {date(2026, 8, 27), date(2026, 8, 31)},
        {date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31)},
    )
    assert report.stale == (date(2026, 8, 28),)
    assert report.pending == ()
    assert report.is_fatal is True


def test_etf_index_latest_gap_is_pending() -> None:
    """A gap only at the index max may still publish, so it is not fatal."""
    from src.data.validation import etf_index_session_gaps

    report = etf_index_session_gaps({date(2026, 9, 22)}, {date(2026, 9, 22), date(2026, 9, 23)})
    assert report.pending == (date(2026, 9, 23),)
    assert report.stale == ()
    assert report.is_fatal is False


def test_etf_ahead_of_index_is_not_a_gap() -> None:
    """ETF dates the index lacks are never reported as gaps."""
    from src.data.validation import etf_index_session_gaps

    report = etf_index_session_gaps({date(2026, 9, 22), date(2026, 9, 23)}, {date(2026, 9, 22)})
    assert report.stale == ()
    assert report.pending == ()
    assert report.is_fatal is False


def test_empty_index_yields_no_gaps() -> None:
    """No index witness means no gaps at all."""
    from src.data.validation import etf_index_session_gaps

    report = etf_index_session_gaps({date(2026, 9, 22)}, set())
    assert report.stale == ()
    assert report.pending == ()


def test_missing_calendar_sessions_are_sorted() -> None:
    """Absent calendar sessions come back in ascending order."""
    from src.data.validation import missing_calendar_sessions

    assert missing_calendar_sessions(
        {date(2026, 8, 27), date(2026, 8, 31)},
        [date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31)],
    ) == (date(2026, 8, 28),)


def test_validator_ignores_declared_closures() -> None:
    """A panel covering every effective session reports no V6 missing sessions."""
    cal = TradingCalendar()
    sessions = cal.sessions(date(2026, 6, 1), date(2026, 7, 31))
    assert date(2026, 6, 3) not in sessions
    assert date(2026, 7, 17) not in sessions
    frame = pl.DataFrame(
        {
            "date": list(sessions),
            "ticker": ["451060"] * len(sessions),
        }
    )
    report = PanelValidator(cal).validate("etf_daily", frame)
    assert not any(issue.gate == "V6_missing_sessions" for issue in report.issues)


def test_validator_rejects_rows_on_declared_closure() -> None:
    """Panel rows on a declared closure are a fatal V2 session mismatch."""
    cal = TradingCalendar()
    frame = pl.DataFrame({"date": [date(2026, 6, 3)], "ticker": ["451060"]})
    report = PanelValidator(cal).validate("etf_daily", frame)
    assert report.is_fatal()
    mismatches = [issue for issue in report.issues if issue.gate == "V2_session_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0].severity == Severity.CRITICAL
    assert "2026-06-03" in mismatches[0].detail
