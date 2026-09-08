# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping
from typing import Final

import polars as pl

TOURNAMENT_INITIAL_CAPITAL_DEFAULT: Final[float] = 1_000_000_000.0


def resolve_capacity_capital(context: object, *, default: float = TOURNAMENT_INITIAL_CAPITAL_DEFAULT) -> float:
    if isinstance(default, (int, float)) and math.isfinite(float(default)) and float(default) > 0:
        fallback = float(default)
    else:
        fallback = TOURNAMENT_INITIAL_CAPITAL_DEFAULT
    rules = getattr(context, "rules", None)
    raw = getattr(rules, "initial_capital", None)
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(value) or value <= 0:
        return fallback
    return value


def cached_filtered_scores(
    cache: dict[Hashable, dict[str, float]],
    key: Hashable,
    snapshot: pl.DataFrame,
    scorer: Callable[[pl.DataFrame], dict[str, float]],
) -> dict[str, float]:
    """Return memoized scorer(snapshot), keyed by caller-supplied key.

    key must be a stable, content-derived identifier for the snapshot (e.g. the decision date)
    -- never id(snapshot), which is a memory address CPython may reuse for an unrelated later
    object once the original snapshot is garbage-collected, causing silent cross-session score
    corruption.
    """
    if key is None:
        raise ValueError("cached_filtered_scores key must not be None")
    if key not in cache:
        cache[key] = scorer(snapshot)
    return dict(cache[key])


def apply_capacity_filter(
    scores: Mapping[str, float],
    snapshot: pl.DataFrame,
    *,
    capital: float,
    max_order_to_adv: float,
    min_fill_ratio: float,
    sleeve_weight: float = 0.95,
    adv_col: str = "trading_value",
) -> dict[str, float]:
    try:
        base = dict(scores)
    except Exception:
        return {}
    if not base:
        return {}
    try:
        mfr = float(min_fill_ratio)
    except Exception:
        return dict(base)
    if not math.isfinite(mfr) or mfr <= 0:
        return dict(base)
    try:
        phi = float(max_order_to_adv)
    except Exception:
        return dict(base)
    if not math.isfinite(phi) or phi <= 0:
        return dict(base)
    try:
        cap = float(capital)
    except Exception:
        return dict(base)
    if not math.isfinite(cap):
        return dict(base)
    try:
        sleeve = float(sleeve_weight)
    except Exception:
        return dict(base)
    if not math.isfinite(sleeve) or sleeve <= 0:
        return dict(base)
    try:
        required_adv = cap * sleeve * mfr / phi
    except Exception:
        return dict(base)
    if not math.isfinite(required_adv):
        return dict(base)
    col: str | None = None
    try:
        cols = list(snapshot.columns) if isinstance(snapshot, pl.DataFrame) else []
    except Exception:
        return dict(base)
    if "adv" in cols:
        col = "adv"
    elif adv_col in cols:
        col = adv_col
    else:
        return dict(base)
    if "ticker" not in cols:
        return dict(base)
    adv_by_ticker: dict[str, float | None] = {}
    try:
        for row in snapshot.iter_rows(named=True):
            try:
                t = str(row.get("ticker"))
            except Exception:
                continue
            raw = row.get(col)
            if raw is None:
                adv_by_ticker[t] = None
                continue
            try:
                fv = float(raw)
            except Exception:
                adv_by_ticker[t] = None
                continue
            adv_by_ticker[t] = float(fv) if math.isfinite(fv) else None
    except Exception:
        return dict(base)
    out: dict[str, float] = {}
    for k, v in base.items():
        adv = adv_by_ticker.get(str(k))
        if adv is None:
            out[str(k)] = float(v) if isinstance(v, (int, float)) else v  # type: ignore[assignment]
            continue
        try:
            if float(adv) * phi < cap * sleeve * mfr:
                continue
        except Exception:
            out[str(k)] = v  # type: ignore[assignment]
            continue
        out[str(k)] = v  # type: ignore[assignment]
    return out
