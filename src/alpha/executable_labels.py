# mypy: ignore-errors
"""Executable staged-execution vehicle labels (P37)."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.backtest.costs import CostConfig


@dataclass(frozen=True)
class ExecutableLabelConfig:
    horizon: int
    fill_budget_fraction: float
    capital: float
    participation: float
    max_single_weight: float
    max_effective_gross: float
    min_cash: float
    costs: CostConfig


@dataclass(frozen=True)
class ExecutableVehicleLabel:
    net_return: float
    filled_weight: float
    fill_sessions: int
    tail_grade: int


def _target_weight(multiple: int, config: ExecutableLabelConfig) -> float:
    cap_single = float(config.max_single_weight)
    cap_gross = float(config.max_effective_gross) / float(abs(int(multiple)))
    cap_cash = 1.0 - float(config.min_cash)
    return float(min(cap_single, cap_gross, cap_cash, 0.95))


def simulate_executable_vehicle_path(
    opens: Sequence[float],
    closes: Sequence[float],
    adv: Sequence[float],
    *,
    leverage_multiple: int,
    objective,
    config: ExecutableLabelConfig,
) -> ExecutableVehicleLabel:
    import numpy as np

    horizon = int(config.horizon)
    o = np.asarray([float(v) for v in opens], dtype=np.float64)
    c = np.asarray([float(v) for v in closes], dtype=np.float64)
    a = np.asarray([float(v) for v in adv], dtype=np.float64)
    if o.shape[0] != horizon or c.shape[0] != horizon or a.shape[0] != horizon:
        raise ValueError("opens/closes/adv must match horizon")
    if not np.all(np.isfinite(o)) or not np.all(np.isfinite(c)):
        raise ValueError("non-finite prices")
    if np.any(o <= 0) or np.any(c <= 0):
        raise ValueError("non-positive prices")
    mult = int(leverage_multiple)
    if mult == 0:
        raise ValueError("leverage_multiple must be non-zero")
    target = _target_weight(mult, config)
    buy_budget = math.floor(float(horizon) * float(config.fill_budget_fraction))
    buy_budget = max(0, min(buy_budget, horizon))
    capital = float(config.capital)
    participation = float(config.participation)
    if capital <= 0 or participation <= 0:
        raise ValueError("capital/participation must be positive")
    bps = float(config.costs.commission_bps or 0.0) + float(config.costs.slippage_bps or 0.0) + float(config.costs.spread_bps or 0.0) + float(config.costs.tax_bps or 0.0)
    # 36-step NumPy state loop: buys only in first floor(horizon*budget) sessions, ADV-capped.
    shares = 0.0
    cash = float(capital)
    filled_notional = 0.0
    fill_sessions = 0
    target_notional = float(target) * float(capital)
    for t in range(horizon):
        if t >= buy_budget:
            break
        remaining = target_notional - filled_notional
        if remaining <= 1e-12:
            break
        adv_t = float(a[t])
        if not math.isfinite(adv_t) or adv_t <= 0:
            continue  # Missing ADV means no fill, never uncapped fill.
        cap_notional = adv_t * participation
        buy_notional = float(min(remaining, cap_notional))
        if buy_notional <= 1e-12:
            continue
        cost = buy_notional * bps / 10000.0
        px = float(o[t])
        shares += buy_notional / px
        cash -= buy_notional + cost
        filled_notional += buy_notional
        fill_sessions += 1
    filled_weight = float(filled_notional / capital)
    exit_px = float(c[t - 1] if False else c[horizon - 1])
    gross_value = float(shares * exit_px)
    exit_cost = gross_value * bps / 10000.0
    equity = float(cash + gross_value - exit_cost)
    net_return = float(equity / capital - 1.0)
    grade_arr = objective.grade([float(net_return)])
    tail_grade = int(next(iter(grade_arr)))
    return ExecutableVehicleLabel(net_return=float(net_return), filled_weight=float(filled_weight), fill_sessions=int(fill_sessions), tail_grade=int(tail_grade))


def build_executable_vehicle_labels(
    candidates: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    sessions: Sequence[date],
    objective,
    config: ExecutableLabelConfig,
    chunk_decision_dates: int = 64,
) -> pl.DataFrame:
    ordered = list(sessions)
    pos = {d: i for i, d in enumerate(ordered)}
    horizon = int(config.horizon)
    # Build date/ticker indices once; read panel once.
    opens_idx: dict[tuple[str, date], float] = {}
    closes_idx: dict[tuple[str, date], float] = {}
    adv_idx: dict[tuple[str, date], float] = {}
    if panel.height > 0:
        for row in panel.iter_rows(named=True):
            d = row.get("date")
            t = str(row.get("ticker"))
            if not isinstance(d, date):
                continue
            try:
                o = row.get("open")
                opens_idx[(t, d)] = float(o) if o is not None else float("nan")
            except Exception:
                opens_idx[(t, d)] = float("nan")
            try:
                cl = row.get("close")
                closes_idx[(t, d)] = float(cl) if cl is not None else float("nan")
            except Exception:
                closes_idx[(t, d)] = float("nan")
            try:
                tv = row.get("trading_value")
                adv_idx[(t, d)] = float(tv) if tv is not None else float("nan")
            except Exception:
                adv_idx[(t, d)] = float("nan")
    cand_by_date: dict[date, list[dict[str, object]]] = {}
    if candidates.height > 0:
        for row in candidates.iter_rows(named=True):
            d = row.get("decision_date")
            if isinstance(d, date):
                cand_by_date.setdefault(d, []).append(row)
    decision_dates = sorted(cand_by_date.keys())
    rows: list[dict[str, object]] = []
    chunk = max(1, int(chunk_decision_dates))
    for start in range(0, len(decision_dates), chunk):
        for d in decision_dates[start : start + chunk]:
            i = pos[d]
            # Decision is formed at close(t); the first fill is open(t+1).
            window = ordered[i + 1 : i + 1 + horizon]
            if len(window) != horizon:
                continue
            for crow in cand_by_date[d]:
                ticker = str(crow.get("source_ticker"))
                mult = 1
                try:
                    mult = int(crow.get("leverage_multiple", 1) or 1)
                except Exception:
                    mult = 1
                o: list[float] = []
                cl: list[float] = []
                av: list[float] = []
                ok = True
                for dd in window:
                    oo = opens_idx.get((ticker, dd), float("nan"))
                    cc = closes_idx.get((ticker, dd), float("nan"))
                    aa = adv_idx.get((ticker, dd), float("nan"))
                    if not math.isfinite(oo) or not math.isfinite(cc) or oo <= 0 or cc <= 0:
                        ok = False
                        break
                    o.append(float(oo))
                    cl.append(float(cc))
                    av.append(float(aa) if math.isfinite(aa) else 0.0)
                if not ok:
                    continue
                label = simulate_executable_vehicle_path(o, cl, av, leverage_multiple=mult, objective=objective, config=config)
                out: dict[str, object] = {
                    "decision_date": d,
                    "source_ticker": ticker,
                    "family_key": crow.get("family_key"),
                    "label_return": float(label.net_return),
                    "filled_weight": float(label.filled_weight),
                    "fill_sessions": int(label.fill_sessions),
                    "label_tail_grade": int(label.tail_grade),
                }
                for key, val in crow.items():
                    if key not in out:
                        out[key] = val
                rows.append(out)
    if not rows:
        return pl.DataFrame(
            {"decision_date": [], "source_ticker": [], "family_key": [], "label_return": [], "filled_weight": [], "fill_sessions": [], "label_tail_grade": []}
        )
    frame = pl.DataFrame(
        rows,
        schema_overrides={"label_return": pl.Float64, "filled_weight": pl.Float64, "fill_sessions": pl.Int64, "label_tail_grade": pl.Int32},
        strict=False,
    )
    # Stream chunks to project-local Parquet is handled by callers; keep frame in Float64.
    return frame


__all__ = [
    "ExecutableLabelConfig",
    "ExecutableVehicleLabel",
    "build_executable_vehicle_labels",
    "simulate_executable_vehicle_path",
]
