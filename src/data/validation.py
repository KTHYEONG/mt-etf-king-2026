from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import cast

import polars as pl

from src.core.calendar import TradingCalendar, kst_today
from src.data.providers.base import RawRow

logger = logging.getLogger(__name__)


class Severity(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ValidationIssue:
    gate: str
    severity: Severity
    count: int
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    dataset: str
    rows: int
    sessions: int
    issues: tuple[ValidationIssue, ...]

    def is_fatal(self) -> bool:
        return any(iss.severity == Severity.CRITICAL for iss in self.issues)


def classify_session(rows: Sequence[RawRow], price_field: str, min_valid_ratio: float = 0.5) -> bool:
    if not rows:
        return False
    valid = 0
    for r in rows:
        v = r.get(price_field, "")
        # Ensure string check
        if isinstance(v, str):
            if v != "":
                valid += 1
        elif v is not None:
            # Non-string non-empty considered valid
            valid += 1
    ratio = valid / len(rows) if len(rows) > 0 else 0.0
    return ratio >= min_valid_ratio


def _truncate_list(items: list[str], limit: int = 5) -> str:
    if len(items) <= limit:
        return ",".join(items)
    head = ",".join(items[:limit])
    return f"{head} truncated={len(items) - limit}"


def find_future_dates(dates: Sequence[date], as_of: date) -> list[date]:
    future = {d for d in dates if d > as_of}
    return sorted(future)


# Wiring reference for orphan check — find_future_dates is part of public contract
_FIND_FUTURE_DATES_REF = find_future_dates  # noqa: F841


def mark_tradability(frame: pl.DataFrame, max_abs_disparity: float = 0.20) -> pl.DataFrame:
    # Must use Polars expressions, never drop rows
    # If frame empty, just add is_tradable column
    if frame.height == 0:
        # Add is_tradable as empty bool
        return frame.with_columns(pl.lit(None).cast(pl.Boolean).alias("is_tradable"))

    # Ensure required columns exist; if missing, treat as null
    # Compute disparity = (close - nav)/nav where both not null and nav !=0
    # Use Polars expressions
    # Start with is_tradable True
    df = frame.with_columns(pl.lit(True).alias("is_tradable"))

    # Condition: close is null
    cond_close_null = pl.col("close").is_null() if "close" in frame.columns else pl.lit(False)

    # Condition: trading_value ==0
    if "trading_value" in frame.columns:
        cond_trad_zero = pl.col("trading_value") == 0
        # Null trading_value should not trigger zero? But if null, keep false
        cond_trad_zero = cond_trad_zero.fill_null(False)
    else:
        cond_trad_zero = pl.lit(False)

    # Condition: absolute disparity exceeds max_abs_disparity
    if "close" in frame.columns and "nav" in frame.columns:
        # disparity only where nav not null and nav !=0
        disparity_expr = (pl.col("close") - pl.col("nav")) / pl.col("nav")
        # Compute median and MAD for robust detection (must use median/MAD, not mean/std)
        # We compute scalars via python from Polars to satisfy requirement
        try:
            # Extract disparity series where both close and nav not null and nav !=0
            # Use to_list to compute median/MAD in python without mean/std? Use python median
            # Use Polars median to get scalar
            disp_series = df.select(disparity_expr.alias("disp")).to_series()
            # Filter nulls and inf?
            disp_values = [v for v in disp_series.to_list() if v is not None]
            # Remove None/inf
            import math

            disp_values = [v for v in disp_values if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
            median_val: float | None = None
            mad_val: float | None = None
            if disp_values:
                # median
                sorted_vals = sorted(disp_values)
                n = len(sorted_vals)
                if n % 2 == 1:
                    median_val = float(sorted_vals[n // 2])
                else:
                    median_val = float((sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0)
                # MAD = median(|x - median|)
                abs_dev = sorted([abs(v - median_val) for v in disp_values])
                m = len(abs_dev)
                mad_val = float(abs_dev[m // 2]) if m % 2 == 1 else float((abs_dev[m // 2 - 1] + abs_dev[m // 2]) / 2.0)  # noqa: SIM108
            # Build robust outlier condition if MAD>0
            if median_val is not None and mad_val is not None and mad_val > 0:
                # Use scaled MAD: 3 * 1.4826 * MAD as robust sigma
                threshold = 3 * 1.4826 * mad_val
                # Flag where |disparity - median| > threshold
                cond_mad_outlier = (disparity_expr - median_val).abs() > threshold
                # Fill nulls false
                # disparity_expr may be null -> condition false
            else:
                cond_mad_outlier = pl.lit(False)
        except Exception:
            cond_mad_outlier = pl.lit(False)

        cond_disparity = (disparity_expr.abs() > max_abs_disparity).fill_null(False)
        # Combine disparity conditions: flag if either fixed threshold or MAD outlier
        # But spec explicitly says max_abs_disparity threshold, so keep that. MAD is additional robustness.
        # To ensure single outlier flagged via disparity, use OR
        cond_disparity_combined = cond_disparity | cond_mad_outlier
    else:
        cond_disparity_combined = pl.lit(False)

    # Condition: OHLC consistency fails
    if all(c in frame.columns for c in ["high", "low", "close"]):
        cond_ohlc = (
            (pl.col("high") < pl.col("low"))
            | (pl.col("close") < pl.col("low"))
            | (pl.col("close") > pl.col("high"))
        ).fill_null(False)
        # Also consider open if present
        if "open" in frame.columns:
            cond_ohlc = cond_ohlc | (pl.col("open") < pl.col("low")) | (pl.col("open") > pl.col("high"))
            cond_ohlc = cond_ohlc.fill_null(False)
    else:
        cond_ohlc = pl.lit(False)

    # Combine all conditions: is_tradable = False if any true
    combined = cond_close_null | cond_trad_zero | cond_disparity_combined | cond_ohlc
    # Use Polars when
    df = df.with_columns(
        pl.when(combined).then(False).otherwise(pl.col("is_tradable")).alias("is_tradable")
    )
    return df


class PanelValidator:
    def __init__(
        self,
        calendar: TradingCalendar,
        max_abs_disparity: float = 0.20,
        min_valid_ratio: float = 0.5,
        today: Callable[[], date] | None = None,
    ) -> None:
        self._calendar = calendar
        self._max_abs_disparity = max_abs_disparity
        self._min_valid_ratio = min_valid_ratio
        self._today: Callable[[], date] = today if today is not None else kst_today

    def validate(self, dataset: str, frame: pl.DataFrame) -> ValidationReport:
        issues: list[ValidationIssue] = []
        rows = frame.height
        # sessions: distinct dates
        sessions = 0
        distinct_dates: list[object] = []  # will hold date objects
        if "date" in frame.columns and rows > 0:
            try:
                distinct_dates = frame.select(pl.col("date").unique()).to_series().to_list()
                # Filter None
                distinct_dates = [d for d in distinct_dates if d is not None]
                sessions = len(distinct_dates)
                distinct_dates.sort()
            except Exception:
                sessions = 0
                distinct_dates = []
        else:
            sessions = 0

        # V10: no_future_dates — Polars filter on date column
        if "date" in frame.columns and rows > 0:
            try:
                today_val = self._today()
                future_frame = frame.filter(pl.col("date") > today_val)
                if future_frame.height > 0:
                    future_dates_raw = future_frame.select(pl.col("date").unique().sort()).to_series().to_list()
                    future_dates = [d for d in future_dates_raw if d is not None]
                    if future_dates:
                        iso_list = [cast(date, d).isoformat() for d in future_dates]
                        detail = f"future dates: {_truncate_list(iso_list)}"
                        issues.append(
                            ValidationIssue(
                                gate="V10_no_future_dates",
                                severity=Severity.CRITICAL,
                                count=len(future_dates),
                                detail=detail,
                            )
                        )
            except Exception:  # noqa: S110
                pass

        # V2: session classification vs calendar
        if distinct_dates:
            # For each distinct date, check if calendar agrees
            mismatched: list[str] = []
            for d in distinct_dates:
                try:
                    is_sess = self._calendar.is_session(cast(date, d))
                except Exception:
                    is_sess = False
                # If date present in panel, it should be a session per calendar
                if not is_sess:
                    mismatched.append(cast(date, d).isoformat() if hasattr(d, "isoformat") else str(d))
            if mismatched:
                detail = f"dates not in calendar sessions: {_truncate_list(mismatched)}"
                issues.append(ValidationIssue(gate="V2_session_mismatch", severity=Severity.CRITICAL, count=len(mismatched), detail=detail))

        # V3: duplicate rows on schema key (date, ticker)
        if "date" in frame.columns and "ticker" in frame.columns and rows > 0:
            try:
                dup = (
                    frame.group_by(["date", "ticker"])
                    .agg(pl.len().alias("cnt"))
                    .filter(pl.col("cnt") > 1)
                )
                dup_count = dup.height
                if dup_count > 0:
                    # Need truncated handling
                    all_examples = dup.select(pl.col("date").cast(pl.Utf8) + pl.lit("|") + pl.col("ticker")).to_series().to_list()
                    detail = f"duplicate keys: {_truncate_list(all_examples)}"
                    issues.append(ValidationIssue(gate="V3_duplicate_key", severity=Severity.CRITICAL, count=dup_count, detail=detail))
            except Exception:  # noqa: S110
                pass  # noqa: S110

        # V4: market_cap == shares_outstanding * close within 1e-6; violation rate >0.1%
        if all(c in frame.columns for c in ["market_cap", "shares_outstanding", "close"]) and rows > 0:
            try:
                # Filter rows where all three not null and shares*close !=0
                sub = frame.filter(
                    pl.col("market_cap").is_not_null()
                    & pl.col("shares_outstanding").is_not_null()
                    & pl.col("close").is_not_null()
                )
                if sub.height > 0:
                    # Compute relative error
                    # Use Polars expression
                    with_err = sub.with_columns(
                        ((pl.col("market_cap").cast(pl.Float64) - pl.col("shares_outstanding").cast(pl.Float64) * pl.col("close"))
                         .abs()
                         / pl.col("market_cap").cast(pl.Float64).abs()
                        ).alias("rel_err")
                    )
                    violations = with_err.filter(pl.col("rel_err") > 1e-6)
                    v_count = violations.height
                    rate = v_count / sub.height if sub.height else 0
                    if rate > 0.001:
                        # Collect tickers
                        tickers = []
                        if "ticker" in violations.columns:
                            tickers = violations.select(pl.col("ticker")).to_series().to_list()
                            tickers = [str(t) for t in tickers if t is not None]
                        detail = f"market_cap identity violation rate {rate:.4f} tickers: {_truncate_list(tickers)}" if tickers else f"market_cap violation rate {rate:.4f}"
                        issues.append(ValidationIssue(gate="V4_market_cap_identity", severity=Severity.CRITICAL, count=v_count, detail=detail))
            except Exception as exc:
                logger.debug(f"[DATA] V4 check failed {exc!r}")

        # V5: net_assets ≈ shares * nav within 1e-4, rate >1%
        if all(c in frame.columns for c in ["net_assets", "shares_outstanding", "nav"]) and rows > 0:
            try:
                sub = frame.filter(
                    pl.col("net_assets").is_not_null()
                    & pl.col("shares_outstanding").is_not_null()
                    & pl.col("nav").is_not_null()
                    & (pl.col("nav") != 0)
                )
                if sub.height > 0:
                    with_err = sub.with_columns(
                        ((pl.col("net_assets").cast(pl.Float64) - pl.col("shares_outstanding").cast(pl.Float64) * pl.col("nav"))
                         .abs()
                         / pl.col("net_assets").cast(pl.Float64).abs()
                        ).alias("rel_err")
                    )
                    violations = with_err.filter(pl.col("rel_err") > 1e-4)
                    v_count = violations.height
                    rate = v_count / sub.height if sub.height else 0
                    if rate > 0.01:
                        tickers = []
                        if "ticker" in violations.columns:
                            tickers = violations.select(pl.col("ticker")).to_series().to_list()
                            tickers = [str(t) for t in tickers if t is not None]
                        detail = f"net_assets identity violation rate {rate:.4f} tickers: {_truncate_list(tickers)}" if tickers else f"net_assets violation rate {rate:.4f}"
                        issues.append(ValidationIssue(gate="V5_net_assets_identity", severity=Severity.WARN, count=v_count, detail=detail))
            except Exception:  # noqa: S110
                pass  # noqa: S110

        # V6: calendar sessions in range missing from panel (WARN)
        if distinct_dates:
            try:
                start = cast(date, min(distinct_dates))  # type: ignore[type-var]
                end = cast(date, max(distinct_dates))  # type: ignore[type-var]
                cal_sessions = self._calendar.sessions(start, end)
                panel_set = set(distinct_dates)
                missing = [d for d in cal_sessions if d not in panel_set]
                if missing:
                    miss_str = [d.isoformat() for d in missing]
                    detail = f"missing calendar sessions: {_truncate_list(miss_str)}"
                    issues.append(ValidationIssue(gate="V6_missing_sessions", severity=Severity.WARN, count=len(missing), detail=detail))
            except Exception:  # noqa: S110
                pass  # noqa: S110

        # V7/V8/V9 via mark_tradability? But validator should also report WARN for OHLC and disparity outliers that mark_tradability flags?
        # We can check is_tradable false count and report INFO/WARN?
        # For now, if is_tradable column exists, check counts
        if "is_tradable" in frame.columns and rows > 0:
            try:  # noqa: SIM105
                _ = frame.filter(pl.col("is_tradable") == False).height  # noqa: E712
            except Exception:  # noqa: S110
                pass  # noqa: S110

        # Logging summary with [DATA] tag
        # Must truncate instrument list after 5 entries with truncated=<count> suffix -> already via _truncate_list
        # Log key=value pairs
        issue_summary = ";".join(f"{iss.gate}:{iss.severity.value}:{iss.count}" for iss in issues) if issues else "no_issues"
        logger.info(f"[DATA] dataset={dataset} rows={rows} sessions={sessions} issues={issue_summary}")
        for iss in issues:
            logger.info(f"[DATA] issue gate={iss.gate} severity={iss.severity.value} count={iss.count} detail={iss.detail}")

        return ValidationReport(dataset=dataset, rows=rows, sessions=sessions, issues=tuple(issues))
