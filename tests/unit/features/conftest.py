from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from src.core.calendar import get_calendar


def session_dates(start: date, count: int) -> list[date]:
    cal = get_calendar()
    end = start + timedelta(days=max(count * 3, 30))
    sessions = cal.sessions(start, end)
    if len(sessions) < count:
        raise ValueError(f"need {count} sessions from {start}, got {len(sessions)}")
    return sessions[:count]


def make_etf_panel(
    sessions: list[date],
    tickers: list[str],
    *,
    base_price: float = 10_000.0,
    growth: float = 0.001,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        price = base_price
        for d in sessions:
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "close": price,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "nav": price * 0.999,
                    "shares_outstanding": 1_000_000,
                    "net_assets": int(price * 1_000_000),
                    "trading_value": 1_000_000_000,
                }
            )
            price *= 1.0 + growth
    return pl.DataFrame(rows)
