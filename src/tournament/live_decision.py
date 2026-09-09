"""Live decision pipeline shared with backtest (P27 fix)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.alpha.base import DecisionContext
from src.portfolio.intent import PortfolioIntent, resolve_portfolio_intent
from src.portfolio.sizing import SizingScheme, weights_from_scores
from src.tournament.championship_regime import (
    ChampionshipSleeve,
    classify_championship_sleeve_from_maps,
    kospi_sleeve_feature_maps,
)


@dataclass(frozen=True, slots=True)
class LiveOrderEstimate:
    ticker: str
    weight: float
    price_basis_date: date
    price: float
    est_shares: int
    est_krw: float


def resolve_live_championship_sleeve(index_daily: pl.DataFrame, decision_date: date) -> str:
    if not isinstance(index_daily, pl.DataFrame):
        return ChampionshipSleeve.UNCERTAIN.value
    if index_daily.height == 0:
        return ChampionshipSleeve.UNCERTAIN.value
    mom60_by_date, mom20_by_date, rv20_daily_by_date = kospi_sleeve_feature_maps(index_daily)
    mom60_trunc = {d: v for d, v in mom60_by_date.items() if d <= decision_date}
    mom20_trunc = {d: v for d, v in mom20_by_date.items() if d <= decision_date}
    rv_trunc = {d: v for d, v in rv20_daily_by_date.items() if d <= decision_date}
    if not mom60_trunc and not mom20_trunc and not rv_trunc:
        return ChampionshipSleeve.UNCERTAIN.value
    snap = classify_championship_sleeve_from_maps(
        decision_date=decision_date,
        mom60_by_date=mom60_trunc,
        mom20_by_date=mom20_trunc,
        rv20_daily_by_date=rv_trunc,
    )
    return snap.sleeve.value


def build_live_eligible_snapshot(panel: pl.DataFrame, *, decision_date: date) -> pl.DataFrame:
    from src.tournament.attainability import market_candidates_by_session

    candidates = market_candidates_by_session(sessions=[decision_date], panel=panel)
    eligible = candidates.get(decision_date)
    if not eligible:
        return panel.head(0)
    tickers = list(eligible)
    out = panel.filter(pl.col("ticker").is_in(tickers))
    if "date" in panel.columns:
        out = out.filter(pl.col("date") == decision_date)
    return out


def compute_live_target_weights(
    model: object,
    snapshot: pl.DataFrame,
    *,
    decision_date: date,
    held: Mapping[str, float],
    capital: float,
    rules: object,
    championship_sleeve: str,
    scheme: SizingScheme = SizingScheme.TOP1,
    k: int = 1,
) -> PortfolioIntent:
    ctx = DecisionContext(
        decision_date=decision_date,
        regime=None,
        capital=float(capital),
        held=dict(held),
        rules=rules,  # type: ignore[arg-type]
        championship_sleeve=championship_sleeve,
    )
    score_result = model.score(snapshot, ctx)  # type: ignore[attr-defined]
    if score_result is None:
        return resolve_portfolio_intent(None, current_weights=held, score_failed=True)
    if isinstance(score_result, PortfolioIntent):
        return score_result
    allocate_fn = getattr(model, "allocate", None)
    if callable(allocate_fn):
        try:
            alloc_result = allocate_fn(
                score_result,
                regime=None,
                leverage_allowed=None,
                inverse_allowed=None,
            )
        except TypeError:
            alloc_result = allocate_fn(score_result)
        return resolve_portfolio_intent(alloc_result, current_weights=held, score_failed=False)
    alloc_weights = weights_from_scores(score_result, scheme, k)
    return resolve_portfolio_intent(alloc_weights, current_weights=held, score_failed=False)


def estimate_live_order_quantities(
    weights: Mapping[str, float],
    panel: pl.DataFrame,
    *,
    decision_date: date,
    capital: float,
) -> dict[str, LiveOrderEstimate]:
    if len(weights) == 0:
        return {}
    try:
        cap = float(capital)
    except (TypeError, ValueError):
        raise ValueError(f"capital must be finite strictly positive, got {capital!r}") from None
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError(f"capital must be finite strictly positive, got {capital!r}")
    out: dict[str, LiveOrderEstimate] = {}
    for ticker, w in weights.items():
        wf = float(w)
        if wf == 0.0:
            continue
        price: float | None = None
        if isinstance(panel, pl.DataFrame) and "ticker" in panel.columns and "close" in panel.columns:
            frame = panel.filter(pl.col("ticker") == ticker)
            if "date" in panel.columns:
                frame = frame.filter(pl.col("date") == decision_date)
            if frame.height > 0:
                for raw in frame.get_column("close").to_list():
                    try:
                        fv = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(fv) and fv > 0:
                        price = fv
                        break
        if price is None:
            raise ValueError(f"missing close price for ticker {ticker} on {decision_date}")
        shares = math.floor(cap * wf / price)
        out[ticker] = LiveOrderEstimate(
            ticker=ticker,
            weight=wf,
            price_basis_date=decision_date,
            price=price,
            est_shares=shares,
            est_krw=float(shares * price),
        )
    return out
