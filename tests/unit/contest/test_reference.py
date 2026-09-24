"""Guards for the single-stock reference download (daily granularity is a hard requirement)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import pytest

from src.contest.reference import ReferenceFetchError, _download_symbol


def _payload(days: list[date]) -> dict[str, Any]:
    ts = [int(datetime(d.year, d.month, d.day, 0, 0, tzinfo=UTC).timestamp()) for d in days]
    return {"chart": {"result": [{"timestamp": ts, "indicators": {"quote": [{
        "open": [100.0 + i for i in range(len(ts))], "close": [101.0 + i for i in range(len(ts))]}]}}]}}


def _client(days: list[date], seen: list[dict[str, str]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=_payload(days))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_requests_explicit_daily_period_and_keeps_rows_up_to_cutoff() -> None:
    """The request pins interval=1d with an explicit period (never range=max) and honors the cutoff."""
    days = [date(2026, 5, 1) + timedelta(days=i) for i in range(40)]
    seen: list[dict[str, str]] = []
    frame = _download_symbol(_client(days, seen), "HYNIX", "000660.KS", date(2026, 5, 26))
    assert seen[0]["interval"] == "1d" and "period1" in seen[0] and "range" not in seen[0]
    assert frame.get_column("date").max() == date(2026, 5, 26)
    assert frame.height == 26


def test_download_rejects_monthly_bars() -> None:
    """Monthly-granularity payloads (Yahoo's range=max degradation) fail closed instead of seeding sparse history."""
    monthly = [date(2020 + i // 12, i % 12 + 1, 1) for i in range(60)]
    with pytest.raises(ReferenceFetchError, match="non-daily bars"):
        _download_symbol(_client(monthly, []), "HYNIX", "000660.KS", date(2026, 5, 26))
