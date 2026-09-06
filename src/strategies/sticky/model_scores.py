# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import polars as pl

from src.alpha.base import DecisionContext
from src.strategies.sticky.capacity import apply_capacity_filter, cached_filtered_scores
from src.universe.instruments import resolve_leverage

from src.strategies.sticky.model_config import StickyLeaderConfig, name_excluded


def collapse_plus2_by_family(scores: Mapping[str, float], snapshot: pl.DataFrame, adv_col: str = "trading_value") -> dict[str, float]:
    if not scores:
        return {}
    if snapshot is None or not isinstance(snapshot, pl.DataFrame):
        return {}
    try:
        if snapshot.height == 0 or snapshot.width == 0:
            return {}
    except Exception:
        return {}
    if "ticker" not in snapshot.columns:
        # without ticker column, treat each ticker as own family
        return dict(scores)
    # Build row lookup and family groups
    row_by_ticker: dict[str, dict] = {}
    try:
        for row in snapshot.iter_rows(named=True):
            try:
                t = str(row.get("ticker"))
            except Exception:
                continue
            if t in scores:
                row_by_ticker[t] = row
    except Exception:
        return dict(scores)
    has_underlying = "underlying_index_name" in snapshot.columns
    has_adv = adv_col in snapshot.columns
    # Group tickers by family
    family_groups: dict[str, list[str]] = {}
    for ticker in scores.keys():
        t_str = str(ticker)
        family: str
        row = row_by_ticker.get(t_str)
        if has_underlying and row is not None:
            try:
                val = row.get("underlying_index_name")
                if val is not None:
                    # handle polars null or nan
                    s = str(val).strip()
                    if s and s.lower() != "none" and s.lower() != "nan":
                        # need to check if original was None, but str(None) == "None" filtered above
                        # also check if val is float nan
                        try:
                            if isinstance(val, float) and not math.isfinite(val):
                                family = t_str
                            else:
                                family = s
                        except Exception:
                            family = s
                    else:
                        family = t_str
                else:
                    family = t_str
            except Exception:
                family = t_str
        else:
            family = t_str
        family_groups.setdefault(family, []).append(t_str)
    out: dict[str, float] = {}
    for family, tickers in family_groups.items():
        if len(tickers) == 1:
            t = tickers[0]
            try:
                out[t] = float(scores[t])
            except Exception:
                continue
            continue
        # Multiple tickers in same family -> pick vehicle
        # If adv column missing, pick max score then ticker id
        if not has_adv:
            # pick max score, tie ticker id
            best = sorted(tickers, key=lambda tk: (-float(scores.get(tk, float("-inf"))), str(tk)))[0]
            try:
                out[best] = float(scores[best])
            except Exception:
                continue
            continue
        # adv column exists: consider finite adv values
        finite_cands: list[tuple[str, float, float]] = []
        for tk in tickers:
            row = row_by_ticker.get(tk)
            adv_val: float | None = None
            if row is not None:
                raw = row.get(adv_col)
                if raw is not None:
                    try:
                        fv = float(raw)
                        if math.isfinite(fv):
                            adv_val = float(fv)
                    except Exception:
                        adv_val = None
            if adv_val is not None:
                try:
                    sc = float(scores[tk])
                except Exception:
                    sc = float("-inf")
                if math.isfinite(sc):
                    finite_cands.append((tk, float(adv_val), float(sc)))
                else:
                    finite_cands.append((tk, float(adv_val), float("-inf")))
        if finite_cands:
            # vehicle = max finite adv (tie: max score, then ticker id)
            # sort by (-adv, -score, ticker)
            finite_cands_sorted = sorted(finite_cands, key=lambda x: (-x[1], -x[2], x[0]))
            winner = finite_cands_sorted[0][0]
            try:
                out[winner] = float(scores[winner])
            except Exception:
                continue
        else:
            # all adv NaN/non-finite -> pick max score then ticker id
            best = sorted(tickers, key=lambda tk: (-float(scores.get(tk, float("-inf"))), str(tk)))[0]
            try:
                out[best] = float(scores[best])
            except Exception:
                continue
    return out


def filter_plus2_scores(snapshot: pl.DataFrame, config: StickyLeaderConfig) -> dict[str, float]:
    if snapshot is None or not isinstance(snapshot, pl.DataFrame):
        return {}
    try:
        if snapshot.height == 0 or snapshot.width == 0:
            return {}
    except Exception:
        return {}
    if config.mom_col not in snapshot.columns:
        # mom_col missing -> empty per fail-closed iterate would skip all
        # Instead return {} directly
        return {}
    if "ticker" not in snapshot.columns:
        return {}
    out: dict[str, float] = {}
    # Check if name column exists; if not, treat name as empty -> skip all
    has_name = "name" in snapshot.columns
    for row in snapshot.iter_rows(named=True):
        try:
            ticker = str(row.get("ticker"))
        except Exception:
            continue
        v = row.get(config.mom_col)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        name = ""
        if has_name:
            try:
                nv = row.get("name")
                if nv is not None:
                    name = str(nv)
            except Exception:
                name = ""
        if not name:
            continue
        try:
            _tokens = tuple(getattr(config, "exclude_name_tokens", ()) or ())
        except Exception:
            _tokens = ()
        if _tokens and name_excluded(name, _tokens):
            continue
        try:
            _excl_synth = bool(getattr(config, "exclude_synthetic", False))
        except Exception:
            _excl_synth = False
        if _excl_synth and "(합성" in str(name):
            continue
        try:
            lev, _conf = resolve_leverage(name)
        except Exception:
            continue
        if config.only_plus_2 and lev != 2:
            continue
        if config.no_inverse and lev < 0:
            continue
        out[ticker] = float(fv)
    return out


def apply_sticky_leader(
    scores: Mapping[str, float], held: str | None, config: StickyLeaderConfig, hold_len: int
) -> dict[str, float]:
    if not scores:
        return {}
    # normalize config
    try:
        mg = float(config.min_gap)
        if not math.isfinite(mg) or mg < 0:
            mg = 0.0
    except Exception:
        mg = 0.0
    try:
        mh = int(config.min_hold)
        if mh < 0:
            mh = 0
        # check finiteness via float
        if not math.isfinite(float(mh)):
            mh = 0
    except Exception:
        mh = 0
    # copy
    out = dict(scores)
    # determine top
    try:
        sorted_items = sorted(out.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
        top_ticker = str(sorted_items[0][0])
        top_score = float(sorted_items[0][1])
    except Exception:
        return out
    if held is None or held not in out:
        return out
    try:
        hl = int(hold_len)
    except Exception:
        hl = 0
    # stay condition
    try:
        held_score = float(out[held])
    except Exception:
        return out
    stay = False
    if hl < mh:
        stay = True
    elif held_score + mg >= top_score - 1e-12:
        stay = True
    if stay:
        max_sc = max(held_score, top_score)
        out[held] = float(max_sc) + 1e-6
    return out
