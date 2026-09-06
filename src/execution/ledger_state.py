# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl

import math

from src.backtest.costs import CostModel
from src.backtest.execution import NextOpenExecution, is_open_fillable
from src.backtest.liquidity import apply_intent_liquidity_constraints, cap_target_weights_by_adv, constrain_target_weights_sell_first
from src.portfolio.intent import CASH_INTENT, HOLD_INTENT, PortfolioIntent, resolve_portfolio_intent




@dataclass(frozen=True)
class PortfolioLedgerState:
    cash: float
    shares: dict[str, float]

    def equity_at_prices(self, prices: Mapping[str, float]) -> float:
        equity = float(self.cash)
        for tkr, sh in self.shares.items():
            p = prices.get(tkr) if isinstance(prices, Mapping) else None
            if p is None:
                continue
            try:
                pf = float(p)
                shf = float(sh)
            except Exception:
                continue
            if not pf == pf or not shf == shf:
                continue
            equity += shf * pf
        return float(equity)

    def weights_at_prices(self, prices: Mapping[str, float]) -> dict[str, float]:
        eq = self.equity_at_prices(prices)
        if eq == 0 or eq != eq:
            return dict.fromkeys(self.shares.keys(), 0.0)
        out: dict[str, float] = {}
        for tkr, sh in self.shares.items():
            p = prices.get(tkr) if isinstance(prices, Mapping) else None
            if p is None:
                out[tkr] = 0.0
                continue
            try:
                pf = float(p)
                shf = float(sh)
            except Exception:
                out[tkr] = 0.0
                continue
            out[tkr] = float(shf * pf / eq) if eq != 0 else 0.0
        return {k: float(v) for k, v in out.items()}


@dataclass(frozen=True)
class SessionTransitionDiagnostics:
    turnover_weight: float
    transaction_cost: float
    fill_count: int
    unfilled_count: int
    target_gross: float
    post_fill_gross: float
    close_realized_gross: float
    effective_gross: float
    gross_violation: bool
    cash_session: bool
    execution_gross_violation: bool = False
    carry_gross_drift: bool = False
    delever_required_next_session: bool = False
    gross_after_sell: float = 0.0
    residual_gross_budget: float = 0.0


@dataclass(frozen=True)
class PortfolioTransitionResult:
    state: PortfolioLedgerState
    equity_close: float
    session_return: float
    weights_after_close: dict[str, float]
    diagnostics: SessionTransitionDiagnostics
    fills: tuple[Fill, ...]
    unfilled: tuple[str, ...]


def ledger_state_from_weights(*, equity: float, weights: Mapping[str, float], mark_prices: Mapping[str, float]) -> PortfolioLedgerState:
    eq = float(equity)
    shares: dict[str, float] = {}
    total_w = 0.0
    for tkr, w in weights.items():
        try:
            wf = float(w)
        except Exception:
            continue
        total_w += wf
        price = mark_prices.get(str(tkr)) if isinstance(mark_prices, Mapping) else None
        if price is None:
            continue
        try:
            pf = float(price)
        except Exception:
            continue
        if pf == 0 or pf != pf:
            continue
        sh = wf * eq / pf
        shares[str(tkr)] = float(sh)
    cash = eq * (1.0 - total_w)
    return PortfolioLedgerState(cash=float(cash), shares=dict(shares))


def _gross_exposure(weights: Mapping[str, float], multiples: Mapping[str, int]) -> float:
    total = 0.0
    for tkr, w in weights.items():
        try:
            wf = float(w)
        except Exception:
            continue
        m = multiples.get(tkr, 1) if isinstance(multiples, Mapping) else 1
        try:
            mf = int(m)
        except Exception:
            mf = 1
        total += abs(wf * float(mf))
    return float(total)


def is_priceless_session(shares: Mapping[str, float], opens: Mapping[str, float]) -> bool:
    try:
        items = list(dict(shares).items())
    except Exception:
        return False
    held = False
    try:
        om = dict(opens) if isinstance(opens, Mapping) else {}
    except Exception:
        om = {}
    for tkr, sh in items:
        try:
            shf = float(sh)
        except Exception:
            continue
        if not math.isfinite(shf) or abs(shf) <= 1e-12:
            continue
        held = True
        try:
            raw = om.get(str(tkr))
            pf = float(raw) if raw is not None else float("nan")
        except Exception:
            continue
        if math.isfinite(pf) and pf > 0:
            return False
    return bool(held)


def carry_forward_marks(shares: Mapping[str, float], *price_maps: Mapping[str, float]) -> dict[str, float]:
    try:
        tickers = [str(k) for k in dict(shares).keys()]
    except Exception:
        return {}
    maps: list[Mapping[str, float]] = [m for m in price_maps if isinstance(m, Mapping)]
    out: dict[str, float] = {}
    for tkr in tickers:
        for mp in maps:
            try:
                raw = mp.get(tkr)
            except Exception:
                continue
            if raw is None:
                continue
            try:
                pf = float(raw)
            except Exception:
                continue
            if math.isfinite(pf) and pf > 0:
                out[tkr] = float(pf)
                break
    return out
