"""One-time seed of single-stock OHLC used to synthesize pre-listing 2x history.

Single-stock 2x ETFs (listed 2026-05-27) need synthetic history before listing:
2x daily of the underlying close. This module downloads daily OHLC from Yahoo
Finance once; post-listing history comes from the listed ETFs in the KRX panel,
so the file never needs refreshing.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Mapping
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import httpx
import polars as pl

logger = logging.getLogger(__name__)

_YAHOO_CHART_URL: Final[str] = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_USER_AGENT: Final[str] = "mt-etf-king-2026 contest-seed-reference/1.0"
_KST: Final[ZoneInfo] = ZoneInfo("Asia/Seoul")
_HISTORY_START_EPOCH: Final[int] = 1483228800  # 2017-01-01; before the contest panel grid (2018-01-02) starts
_MAX_MEDIAN_GAP_DAYS: Final[int] = 4


class ReferenceFetchError(RuntimeError):
    """Raised when the single-stock reference download or parse fails (nothing written)."""


def _download_symbol(client: httpx.Client, key: str, yahoo_symbol: str, cutoff: date) -> pl.DataFrame:
    try:
        # range=max silently degrades to monthly bars; an explicit period keeps the requested daily granularity
        params: dict[str, str | int] = {"interval": "1d", "period1": _HISTORY_START_EPOCH, "period2": int(datetime.now(tz=UTC).timestamp()) + 86400}
        response = client.get(_YAHOO_CHART_URL.format(symbol=yahoo_symbol), params=params)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ReferenceFetchError(f"download failed symbol={key} error={exc!r}") from exc
    try:
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        opens = quote["open"]
        closes = quote["close"]
    except (KeyError, TypeError, IndexError) as exc:
        raise ReferenceFetchError(f"unexpected chart shape symbol={key}") from exc
    rows: list[tuple[date, float, float]] = []
    for ts, o, c in zip(timestamps, opens, closes, strict=False):
        try:
            day = datetime.fromtimestamp(float(ts), tz=UTC).astimezone(_KST).date()
            o_f, c_f = float(o), float(c)
        except (TypeError, ValueError, OverflowError, OSError):
            continue
        if day > cutoff or not (o_f > 0 and c_f > 0):
            continue
        rows.append((day, o_f, c_f))
    if not rows:
        raise ReferenceFetchError(f"no usable rows symbol={key} cutoff={cutoff.isoformat()}")
    day_list = sorted({r[0] for r in rows})
    gaps = sorted((b - a).days for a, b in pairwise(day_list))
    median_gap = gaps[len(gaps) // 2] if gaps else None
    if median_gap is None or median_gap > _MAX_MEDIAN_GAP_DAYS:
        raise ReferenceFetchError(f"non-daily bars symbol={key} median_gap_days={median_gap}")
    frame = pl.DataFrame(
        {"date": [r[0] for r in rows], "symbol": [key] * len(rows), "open": [r[1] for r in rows], "close": [r[2] for r in rows]},
        schema={"date": pl.Date, "symbol": pl.String, "open": pl.Float64, "close": pl.Float64},
    )
    return frame.unique("date", keep="last").sort("date")


def seed_single_stock_reference(symbols: Mapping[str, str], cutoff: date, out_path: Path, timeout_s: float) -> int:
    """One-time download of daily OHLC for the single-stock underlyings used to synthesize pre-listing history of
    single-stock 2x ETFs. Only rows with KST trade date <= cutoff are kept (post-listing history comes from the
    listed ETFs in the KRX panel, so this file never needs refreshing).

    Returns:
        Number of rows written (columns: date, symbol, open, close).

    Raises:
        ReferenceFetchError: download or parse failure (nothing written).
    """
    if not symbols:
        raise ReferenceFetchError("no symbols configured")
    try:
        with httpx.Client(timeout=timeout_s, headers={"User-Agent": _USER_AGENT}) as client:
            frames = [_download_symbol(client, key, ticker, cutoff) for key, ticker in symbols.items()]
    except ReferenceFetchError:
        raise
    except Exception as exc:
        raise ReferenceFetchError(f"download failed error={exc!r}") from exc
    frame = pl.concat(frames).sort("symbol", "date")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(out_path.parent), suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        frame.write_parquet(tmp_path)
        tmp_path.rename(out_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise ReferenceFetchError(f"write failed path={out_path} error={exc!r}") from exc
    n_rows = frame.height
    logger.info(f"[DATA] contest_seed_reference rows={n_rows} symbols={len(frames)} cutoff={cutoff.isoformat()}")
    return n_rows
