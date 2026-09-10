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


from src.execution.ledger_state import (
    PortfolioLedgerState,
    PortfolioTransitionResult,
    SessionTransitionDiagnostics,
    _gross_exposure,
    carry_forward_marks,
    is_priceless_session,
)
from src.execution.cash_accounting import cash_order_transition


def transition_portfolio_state(
    *,
    prior_state: PortfolioLedgerState,
    intent: PortfolioIntent,
    decision_date: date,
    prev_closes: Mapping[str, float],
    opens: Mapping[str, float],
    closes: Mapping[str, float],
    cost_model: CostModel,
    adv_by_ticker: Mapping[str, float],
    max_order_to_adv: float,
    exposure_limits: tuple[float, float, float] | None,
    leverage_multiples: Mapping[str, int],
    execution: NextOpenExecution | None,
    panel: pl.DataFrame | None,
    lot_size: int | None = None,
) -> PortfolioTransitionResult:
    if is_priceless_session(prior_state.shares, opens):
        try:
            _eq_prev = prior_state.equity_at_prices(prev_closes) if prev_closes else float(prior_state.cash)
        except Exception:
            _eq_prev = float(prior_state.cash)
        _marks = carry_forward_marks(prior_state.shares, closes, opens, prev_closes)
        _need = [k for k, v in dict(prior_state.shares).items() if abs(float(v)) > 1e-12] if prior_state.shares else []
        if _need and all(k in _marks for k in [str(x) for x in _need]):
            _eq = float(prior_state.cash) + sum(float(prior_state.shares[k]) * float(_marks[str(k)]) for k in [str(x) for x in _need])
        else:
            _eq = float(_eq_prev)
        _wac: dict[str, float] = {}
        if _eq != 0:
            for k in [str(x) for x in _need]:
                if k in _marks:
                    try:
                        _wac[k] = float(float(prior_state.shares[k]) * float(_marks[k]) / float(_eq))
                    except Exception:
                        continue
        _diag = SessionTransitionDiagnostics(
            turnover_weight=0.0, transaction_cost=0.0, fill_count=0, unfilled_count=0,
            target_gross=0.0, post_fill_gross=0.0, close_realized_gross=0.0, effective_gross=0.0,
            gross_violation=False, cash_session=False,
        )
        return PortfolioTransitionResult(
            state=PortfolioLedgerState(cash=float(prior_state.cash), shares=dict(prior_state.shares)),
            equity_close=float(_eq), session_return=0.0, weights_after_close=dict(_wac),
            diagnostics=_diag, fills=(), unfilled=(),
        )
    return cash_order_transition(
        prior_state=prior_state,
        intent=intent,
        decision_date=decision_date,
        prev_closes=prev_closes,
        opens=opens,
        closes=closes,
        cost_model=cost_model,
        adv_by_ticker=adv_by_ticker,
        max_order_to_adv=max_order_to_adv,
        exposure_limits=exposure_limits,
        leverage_multiples=leverage_multiples,
        execution=execution,
        panel=panel,
        lot_size=lot_size,
    )


def resolve_session_intent(
    *,
    score_result: object,
    alloc_result: object | None,
    current_weights: Mapping[str, float],
    score_failed: bool,
) -> PortfolioIntent:
    if score_failed:
        return HOLD_INTENT
    try:
        from src.portfolio.intent import PortfolioIntent as _PI

        if isinstance(score_result, _PI):
            return score_result
    except Exception:
        pass
    if alloc_result is not None:
        return resolve_portfolio_intent(alloc_result, current_weights=current_weights, score_failed=False)
    if isinstance(score_result, Mapping):
        return resolve_portfolio_intent(score_result, current_weights=current_weights, score_failed=False)
    return HOLD_INTENT


def aggregate_session_diagnostics(sessions: Sequence[SessionTransitionDiagnostics], *, gross_limit: float) -> object:
    try:
        from src.tournament.simulator import RollingDiagnostics
    except Exception:
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class RollingDiagnostics:
            gross_violation_count: int | None
            effective_gross_max: float | None
            turnover_mean: float | None
            fill_count: int | None
            unfilled_count: int | None

    if not sessions:
        try:
            return RollingDiagnostics(
                gross_violation_count=None,
                effective_gross_max=None,
                turnover_mean=0.0,
                fill_count=0,
                unfilled_count=0,
                carry_gross_drift_count=0,
                delever_required_count=0,
            )
        except TypeError:
            return RollingDiagnostics(
                gross_violation_count=None,
                effective_gross_max=None,
                turnover_mean=0.0,
                fill_count=0,
                unfilled_count=0,
            )
    gross_violation_count = 0
    effective_gross_max = 0.0
    turnover_sum = 0.0
    fill_sum = 0
    unfilled_sum = 0
    carry_count = 0
    delever_count = 0
    for s in sessions:
        try:
            # INV-SF-8: count execution_gross_violation only (carry drift excluded)
            if hasattr(s, "execution_gross_violation"):
                if bool(getattr(s, "execution_gross_violation", False)):
                    gross_violation_count += 1
            elif bool(getattr(s, "gross_violation", False)):
                gross_violation_count += 1
            if bool(getattr(s, "carry_gross_drift", False)):
                carry_count += 1
            if bool(getattr(s, "delever_required_next_session", False)):
                delever_count += 1
            eg = float(getattr(s, "close_realized_gross", getattr(s, "effective_gross", 0.0)) or 0.0)
            # effective_gross_max should be max of close_realized_gross (alias effective_gross)
            if eg > effective_gross_max:
                effective_gross_max = float(eg)
            # also track post_fill max? spec says effective_gross_max, use close
            turnover_sum += float(getattr(s, "turnover_weight", 0.0) or 0.0)
            fill_sum += int(getattr(s, "fill_count", 0) or 0)
            unfilled_sum += int(getattr(s, "unfilled_count", 0) or 0)
        except Exception:
            continue
    turnover_mean = float(turnover_sum / len(sessions)) if sessions else 0.0
    try:
        return RollingDiagnostics(
            gross_violation_count=int(gross_violation_count),
            effective_gross_max=float(effective_gross_max),
            turnover_mean=float(turnover_mean),
            fill_count=int(fill_sum),
            unfilled_count=int(unfilled_sum),
            carry_gross_drift_count=int(carry_count),
            delever_required_count=int(delever_count),
        )
    except TypeError:
        return RollingDiagnostics(
            gross_violation_count=int(gross_violation_count),
            effective_gross_max=float(effective_gross_max),
            turnover_mean=float(turnover_mean),
            fill_count=int(fill_sum),
            unfilled_count=int(unfilled_sum),
        )
