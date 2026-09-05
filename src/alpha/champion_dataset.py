# mypy: ignore-errors
# ruff: noqa: S101
"""PIT family-tail dataset: 1x representatives with next-open real-vehicle labels."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True)
class ChampionDatasetConfig:
    feature_columns: tuple[str, ...]
    label_horizon: int
    entry_cost_rate: float
    exit_cost_rate: float
    source_multiple: int = 1
    tail_thresholds: tuple[float, ...] = ()
    tail_weights: tuple[float, ...] = ()


def _validate_primary_tail_objective(
    thresholds: tuple[float, ...], weights: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    thr = tuple(float(v) for v in (thresholds or ()))
    wts = tuple(float(v) for v in (weights or ()))
    if not thr and not wts:
        return None
    if len(thr) == 0 or len(wts) == 0:  # pragma: no cover - defensive malformed config
        raise ValueError("tail objective vectors must be non-empty and equal length")
    if len(thr) != len(wts):
        raise ValueError("tail thresholds/weights length mismatch")
    for v in thr:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("tail thresholds must be positive finite")
    for i in range(1, len(thr)):
        if not thr[i] > thr[i - 1]:
            raise ValueError("tail thresholds must be strictly ascending")
    for v in wts:
        if not math.isfinite(v) or v < 0:
            raise ValueError("tail weights must be finite non-negative")
    if sum(wts) <= 0:
        raise ValueError("tail weights must have positive total")
    return thr, wts


def collect_family_candidates(
    panel: pl.DataFrame,
    *,
    sessions: Sequence[date],
    universe,  # PointInTimeUniverse (duck-typed to avoid import cycle)
    filters,  # UniverseFilters
    master,  # InstrumentMaster
    config: ChampionDatasetConfig,
) -> pl.DataFrame:
    """Point-in-time deployment-only 1x family representatives (vectorized per date)."""
    from src.universe.instruments import Confidence

    rows: list[dict[str, object]] = []
    session_list = list(sessions)
    if not session_list or panel.height == 0:
        return pl.DataFrame(
            {
                "decision_date": [],
                "source_ticker": [],
                "family_key": [],
                **{c: [] for c in config.feature_columns},
            }
        )
    # Index panel rows by date once (no per-window full scan beyond filter).
    for d in session_list:
        snap = universe.get(d, filters)
        eligible = set(snap.tickers)
        if not eligible:
            continue
        day = panel.filter(pl.col("date") == d)
        if day.height == 0:
            continue
        day = day.filter(pl.col("ticker").is_in(sorted(eligible)))
        if day.height == 0:
            continue
        # Group candidates by family: keep one source_multiple representative.
        best: dict[str, tuple[float, str, dict[str, object]]] = {}
        for row in day.iter_rows(named=True):
            ticker = str(row.get("ticker"))
            attr = master.attributes.get(ticker)
            if attr is None:
                continue
            if bool(getattr(attr, "is_synthetic", False)):
                continue
            conf = getattr(attr, "confidence", None)
            if conf == Confidence.LOW or str(conf) == "LOW":
                # Reject LOW-confidence sources (fail-closed for leveraged inference).
                continue
            mult = int(getattr(attr, "leverage_multiple", 1))
            if mult != int(config.source_multiple):
                continue
            if mult < 0:
                continue
            family = str(getattr(attr, "leverage_family_key", ticker))
            try:
                adv_val = universe.adv(ticker, d)
                adv = float(adv_val) if adv_val is not None else float(row.get("trading_value") or 0.0)
            except Exception:
                adv = float(row.get("trading_value") or 0.0)
            if not math.isfinite(adv):
                adv = 0.0
            key = family
            cand = (adv, ticker, row)
            if key not in best or (adv, ticker) > (best[key][0], best[key][1]):
                # Deterministic tie-break: higher ADV wins; ticker ascending wins ties.
                # Store with negated tie logic via explicit compare below.
                if key not in best:
                    best[key] = cand
                else:
                    prev_adv, prev_ticker, _ = best[key]
                    if adv > prev_adv or (adv == prev_adv and ticker < prev_ticker):
                        best[key] = cand
        for family, (_, ticker, row) in sorted(best.items()):
            out: dict[str, object] = {"decision_date": d, "source_ticker": ticker, "family_key": family}
            for col in config.feature_columns:
                out[col] = row.get(col)
            rows.append(out)
    if not rows:
        return pl.DataFrame(
            {
                "decision_date": [],
                "source_ticker": [],
                "family_key": [],
                **{c: [] for c in config.feature_columns},
            }
        )
    return pl.DataFrame(
        rows,
        schema_overrides=dict.fromkeys(config.feature_columns, pl.Float64),
        strict=False,
    )


def build_family_tail_dataset(
    candidates: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    sessions: Sequence[date],
    config: ChampionDatasetConfig,
) -> pl.DataFrame:
    """Real unlevered path label: enter open(i+1), exit close(i+horizon), both costs."""
    tail = _validate_primary_tail_objective(config.tail_thresholds, config.tail_weights)
    if candidates.height == 0:
        return pl.DataFrame(
            {
                "decision_date": [],
                "source_ticker": [],
                "family_key": [],
                "label_return": [],
                "label_tail_utility": [],
                "label_rank": [],
            }
        )
    ordered = list(sessions)
    pos = {d: i for i, d in enumerate(ordered)}
    # Vectorized lookup: single pass over prices into dict.
    lookup: dict[tuple[str, date], tuple[float | None, float | None]] = {}
    for row in prices.iter_rows(named=True):
        t = str(row.get("ticker"))
        d = row.get("date")
        if not isinstance(d, date):
            continue
        try:
            o = float(row["open"]) if row.get("open") is not None else None
        except Exception:
            o = None
        try:
            c = float(row["close"]) if row.get("close") is not None else None
        except Exception:
            c = None
        lookup[(t, d)] = (o, c)
    enriched: list[dict[str, object]] = []
    for row in candidates.iter_rows(named=True):
        d = row.get("decision_date")
        ticker = str(row.get("source_ticker"))
        if not isinstance(d, date) or d not in pos:
            continue
        i = pos[d]
        entry_idx = i + 1
        exit_idx = i + int(config.label_horizon)
        if entry_idx >= len(ordered) or exit_idx >= len(ordered):
            continue
        entry_date = ordered[entry_idx]
        exit_date = ordered[exit_idx]
        entry = lookup.get((ticker, entry_date))
        exitp = lookup.get((ticker, exit_date))
        if entry is None or exitp is None:
            continue
        entry_open, _ = entry
        _, exit_close = exitp
        if entry_open is None or exit_close is None or entry_open <= 0:
            continue
        gross = float(exit_close) / float(entry_open)
        net = gross * (1.0 - float(config.exit_cost_rate)) / (1.0 + float(config.entry_cost_rate)) - 1.0
        if not math.isfinite(net):
            continue
        net = round(float(net), 12)
        out: dict[str, object] = {
            "decision_date": d,
            "source_ticker": ticker,
            "family_key": row.get("family_key"),
            "label_return": net,
        }
        if tail is not None:
            thr, wts = tail
            util = sum(float(w) for t, w in zip(thr, wts, strict=True) if float(net) > float(t))
            out["label_tail_utility"] = round(float(util), 12)
        else:
            out["label_tail_utility"] = 0.0
        for col in config.feature_columns:
            out[col] = row.get(col)
        enriched.append(out)
    if not enriched:
        return pl.DataFrame(
            {
                "decision_date": [],
                "source_ticker": [],
                "family_key": [],
                "label_return": [],
                "label_tail_utility": [],
                "label_rank": [],
            }
        )
    frame = pl.DataFrame(
        enriched,
        schema_overrides={
            "label_return": pl.Float64,
            "label_tail_utility": pl.Float64,
            **dict.fromkeys(config.feature_columns, pl.Float64),
        },
        strict=False,
    )
    if tail is not None:
        # Primary-tail cross-section: sort by (utility, net return, ticker) with
        # deterministic equal-utility tie-break; smallest ticker ranks highest.
        ranks: dict[int, float] = {}
        by_date_u: dict[date, list[tuple[int, float, float, str]]] = {}
        rets = frame.select("label_return").to_series().to_list()
        utils = frame.select("label_tail_utility").to_series().to_list()
        tickers = frame.select("source_ticker").to_series().to_list()
        dts = frame.select("decision_date").to_series().to_list()
        for idx, (dd, u, r, t) in enumerate(zip(dts, utils, rets, tickers, strict=True)):
            by_date_u.setdefault(dd, []).append((idx, float(u), float(r), str(t)))
        for items in by_date_u.values():  # noqa: PERF102
            pre = sorted(items, key=lambda kv: kv[3], reverse=True)
            ordered_items = sorted(pre, key=lambda kv: (kv[1], kv[2]))
            n = len(ordered_items)
            for rank, (idx, _u, _r, _t) in enumerate(ordered_items, start=1):
                ranks[idx] = float(rank) / float(n) if n else 0.0
        frame = frame.with_columns(pl.Series("label_rank", [ranks[i] for i in range(frame.height)]))
        return frame
    # Cross-sectional within-date rank of net return (no leverage multiplier).
    ranks = {}
    by_date: dict[date, list[tuple[int, float]]] = {}
    rets = frame.select("label_return").to_series().to_list()
    dts = frame.select("decision_date").to_series().to_list()
    for idx, (dd, r) in enumerate(zip(dts, rets, strict=True)):
        by_date.setdefault(dd, []).append((idx, float(r)))
    for items in by_date.values():  # noqa: PERF102
        ordered_items = sorted(items, key=lambda kv: kv[1])
        n = len(ordered_items)
        for rank, (idx, _r) in enumerate(ordered_items, start=1):
            ranks[idx] = float(rank) / float(n) if n else 0.0
    frame = frame.with_columns(pl.Series("label_rank", [ranks[i] for i in range(frame.height)]))
    return frame


__all__ = ["ChampionDatasetConfig", "build_family_tail_dataset", "collect_family_candidates"]
