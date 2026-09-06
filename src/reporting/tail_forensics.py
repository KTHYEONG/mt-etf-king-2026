# mypy: ignore-errors
# ruff: noqa
"""Backward-compatible facade over split tail_forensics submodules."""

from __future__ import annotations

from src.reporting.tail_attribution import (
    _median,
    _quantile,
    _trimmed_mean,
    attribute_window,
    summarise_tail_attribution,
    write_tail_attribution_report,
)
from src.reporting.tail_miss import TailMissReport, summarise_tail_miss_windows, write_tail_miss_report
from src.reporting.tail_windows import (
    TailAttributionSummary,
    WindowAttribution,
    _actual_family_from_trades,
    _close_on,
    _family_of,
    _finite_or_none,
    _next_session_after,
    _next_session_on_or_after,
    _open_on,
    _plus2_tickers,
    _zero_attribution,
    compound_close_return,
    next_open_path_return,
    oracle_peak_path_return,
    pit_plus2_tickers,
    select_attribution_windows,
)

__all__ = [
    "TailMissReport",
    "WindowAttribution",
    "TailAttributionSummary",
    "_close_on",
    "compound_close_return",
    "_open_on",
    "_next_session_on_or_after",
    "_next_session_after",
    "next_open_path_return",
    "oracle_peak_path_return",
    "pit_plus2_tickers",
    "_plus2_tickers",
    "_family_of",
    "select_attribution_windows",
    "_zero_attribution",
    "_actual_family_from_trades",
    "_finite_or_none",
    "attribute_window",
    "_median",
    "_trimmed_mean",
    "_quantile",
    "summarise_tail_attribution",
    "write_tail_attribution_report",
    "summarise_tail_miss_windows",
    "write_tail_miss_report",
]
