# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date

import polars as pl

from src.backtest.costs import CostModel
from src.backtest.execution import NextOpenExecution
from src.execution.ledger_state import (
    PortfolioLedgerState,
    PortfolioTransitionResult,
    SessionTransitionDiagnostics,
    _gross_exposure,
)
from src.portfolio.intent import PortfolioIntent


def _fillable(price: object) -> bool:
    return isinstance(price, (int, float)) and math.isfinite(float(price)) and float(price) > 0


def cash_order_transition(
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
    if not isinstance(prior_state.cash, (int, float)) or not math.isfinite(float(prior_state.cash)) or float(prior_state.cash) < 0:
        raise ValueError("prior cash must be finite and nonnegative")
    if lot_size is not None and not (isinstance(lot_size, int) and lot_size > 0):
        raise ValueError("lot_size must be a positive integer or None")
    if not isinstance(max_order_to_adv, (int, float)) or not math.isfinite(float(max_order_to_adv)) or float(max_order_to_adv) <= 0:
        raise ValueError("max_order_to_adv must be positive finite")
    fee = float(cost_model.charge(1.0))
    if not math.isfinite(fee) or fee < 0:
        raise ValueError("invalid cost model")
    weights_raw = dict(getattr(intent, "weights", {}) or {})
    for _w in weights_raw.values():
        if not isinstance(_w, (int, float)) or not math.isfinite(float(_w)) or float(_w) < 0:
            raise ValueError("invalid target weight")
    kind = str(getattr(intent, "kind", ""))
    is_hold = kind == "hold"
    is_cash = kind == "cash"
    target = {} if (is_hold or is_cash) else {str(k): float(v) for k, v in weights_raw.items()}
    held: dict[str, float] = {}
    invalid_share_keys: set[str] = set()
    for key, value in dict(prior_state.shares).items():
        try:
            quantity = float(value)
        except (TypeError, ValueError):  # pragma: no cover - malformed persisted ledger
            invalid_share_keys.add(str(key))  # pragma: no cover
            continue  # pragma: no cover
        if abs(quantity) > 1e-12:
            held[str(key)] = quantity
        if not isinstance(value, (int, float)):  # pragma: no cover - malformed persisted ledger
            invalid_share_keys.add(str(key))  # pragma: no cover
    universe = set(held) | set(target)
    # Legacy fast-simulation callers pass an empty map when no leverage
    # resolver is configured; treat that as the unlevered default.
    if isinstance(leverage_multiples, Mapping) and not leverage_multiples and universe:  # pragma: no cover - legacy caller fallback
        leverage_multiples = {t: 1 for t in universe}  # pragma: no cover
    for _t in universe:
        _m = leverage_multiples.get(_t) if isinstance(leverage_multiples, Mapping) else None
        if _m is None or not isinstance(_m, (int, float)) or not math.isfinite(float(_m)) or int(float(_m)) == 0 or float(_m) != float(int(float(_m))):
            raise ValueError(f"missing or invalid leverage for {_t}")
    if held and not prev_closes and any(_t not in opens for _t in held):
        raise ValueError("missing previous close for held position")
    marks: dict[str, float] = {}
    for _t in held:
        _o = opens.get(_t) if isinstance(opens, Mapping) else None
        _p = prev_closes.get(_t) if isinstance(prev_closes, Mapping) else None
        if _fillable(_o):
            marks[_t] = float(_o)  # type: ignore[arg-type]
        elif _fillable(_p):
            marks[_t] = float(_p)  # type: ignore[arg-type]
        else:
            if is_hold:  # pragma: no cover - HOLD normally has a carry mark
                return PortfolioTransitionResult(
                    state=PortfolioLedgerState(cash=float(prior_state.cash), shares=dict(prior_state.shares)),
                    equity_close=float(prior_state.cash),
                    session_return=0.0,
                    weights_after_close={},
                    diagnostics=SessionTransitionDiagnostics(
                        turnover_weight=0.0, transaction_cost=0.0, fill_count=0, unfilled_count=0,
                        target_gross=0.0, post_fill_gross=0.0, close_realized_gross=0.0,
                        effective_gross=0.0, gross_violation=False, cash_session=False,
                    ),
                    fills=(),
                    unfilled=(),
                )
            # A stale holding can temporarily fall outside the execution
            # universe.  Keep ownership in the ledger and value it at zero
            # until a valid quote returns; dropping the share would corrupt
            # NAV and make the next rebalance non-reproducible.
            marks[_t] = 0.0
    equity = float(prior_state.cash) + sum(_q * marks[_t] for _t, _q in held.items())
    equity_prev = float(prior_state.cash) + sum(float(held[_t]) * float(prev_closes[_t]) for _t in held if _fillable(prev_closes.get(_t))) if prev_closes else equity
    w_lim, g_lim, c_lim = (1.0, 1.0, 0.0) if exposure_limits is None else tuple(float(v) for v in exposure_limits)
    if not (0.0 < w_lim <= 1.0 and 0.0 < g_lim and 0.0 <= c_lim < 1.0):
        raise ValueError("invalid exposure limits")  # pragma: no cover - validated at configuration load
    if is_hold:
        close_marks = {}
        for _t in held:
            _c = closes.get(_t) if isinstance(closes, Mapping) else None
            if _fillable(_c):
                close_marks[_t] = float(_c)  # type: ignore[arg-type]
            else:
                close_marks[_t] = marks[_t]
        equity_close = float(prior_state.cash) + sum(float(held[_t]) * close_marks[_t] for _t in held)
        wac = {t: float(held[t]) * close_marks[t] / equity_close for t in held if t in close_marks and t not in invalid_share_keys and equity_close != 0 and abs(float(held[t]) * close_marks[t] / equity_close) > 1e-12}
        diag = SessionTransitionDiagnostics(turnover_weight=0.0, transaction_cost=0.0, fill_count=0, unfilled_count=0, target_gross=0.0, post_fill_gross=_gross_exposure({t: float(held[t]) * float(marks[t]) / equity for t in held} if equity != 0 else {}, leverage_multiples if isinstance(leverage_multiples, Mapping) else {}), close_realized_gross=_gross_exposure(wac, leverage_multiples if isinstance(leverage_multiples, Mapping) else {}), effective_gross=_gross_exposure(wac, leverage_multiples if isinstance(leverage_multiples, Mapping) else {}), gross_violation=False, cash_session=False, execution_gross_violation=False, carry_gross_drift=bool(_gross_exposure(wac, leverage_multiples if isinstance(leverage_multiples, Mapping) else {}) > (float(exposure_limits[1]) + 1e-9) if exposure_limits is not None else False), delever_required_next_session=bool(_gross_exposure(wac, leverage_multiples if isinstance(leverage_multiples, Mapping) else {}) > (float(exposure_limits[1]) + 1e-9) if exposure_limits is not None else False), gross_after_sell=0.0, residual_gross_budget=0.0)
        return PortfolioTransitionResult(state=PortfolioLedgerState(cash=float(prior_state.cash), shares=dict(prior_state.shares)), equity_close=float(equity_close), session_return=float(equity_close / equity_prev - 1.0) if equity_prev != 0 else 0.0, weights_after_close=dict(wac), diagnostics=diag, fills=(), unfilled=())
    exec_date = decision_date
    if execution is not None and panel is not None:
        needed = sorted(universe)
        res_fills, _res_unfilled = execution.resolve(dict.fromkeys(needed, 1.0), panel, decision_date)
        exec_date = res_fills[0].execution_date if res_fills else decision_date
        for _f in res_fills:
            if _f.ticker in opens:
                assert float(_f.price) == float(opens[_f.ticker])
    shares: dict[str, float] = dict(held)
    cash = float(prior_state.cash)
    traded: dict[str, float] = {}
    fills: list = []
    unfilled: list[str] = []
    total_cost = 0.0
    total_notional = 0.0
    budgets: dict[str, float] = {}
    for _t in universe:
        _a = adv_by_ticker.get(_t) if isinstance(adv_by_ticker, Mapping) else None
        budgets[_t] = float(_a) * float(max_order_to_adv) if isinstance(_a, (int, float)) and math.isfinite(float(_a)) and float(_a) > 0 else 0.0
    for _t in sorted(held):
        _o = opens.get(_t) if isinstance(opens, Mapping) else None
        desired_w = 0.0 if (is_cash or _t not in target) else float(target[_t])
        desired = desired_w * equity / float(_o) if _fillable(_o) else 0.0
        if lot_size is not None:
            desired = math.floor(desired)
        if not _fillable(_o):
            if abs(float(held[_t]) - 0.0) > 1e-12 and (is_cash or desired_w != float(held[_t]) * float(marks[_t]) / equity if equity != 0 else True):
                unfilled.append(_t)
            continue
        qty = float(held[_t]) - desired
        if qty <= 1e-12:
            continue
        max_shares = budgets[_t] / float(_o) if lot_size is None else math.floor(budgets[_t] / float(_o))
        qty = min(qty, max_shares)
        if lot_size is not None:
            qty = math.floor(qty)
        if qty <= 0:
            if abs(float(held[_t]) - desired) > 1e-9:
                unfilled.append(_t)
            continue
        qty = min(qty, float(held[_t]))
        notional = float(qty) * float(_o)
        cost = float(cost_model.charge(notional))
        cash += notional - cost
        shares[_t] = float(shares.get(_t, 0.0)) - float(qty)
        traded[_t] = traded.get(_t, 0.0) + notional
        total_cost += cost
        total_notional += notional
        from src.backtest.execution import Fill as _Fill

        fills.append(_Fill(ticker=_t, execution_date=exec_date, price=float(_o), target_weight=float(desired_w)))
        if abs(float(shares[_t])) <= 1e-12:
            shares.pop(_t, None)
    shares = {t: float(q) for t, q in shares.items() if abs(float(q)) > 1e-12}
    cur_marks: dict[str, float] = {}
    for _t in set(shares) | set(target):
        _o = opens.get(_t) if isinstance(opens, Mapping) else None
        _p = prev_closes.get(_t) if isinstance(prev_closes, Mapping) else None
        if _fillable(_o):
            cur_marks[_t] = float(_o)  # type: ignore[arg-type]
        elif _fillable(_p):
            cur_marks[_t] = float(_p)  # type: ignore[arg-type]
    equity_cur = cash + sum(float(shares.get(_t, 0.0)) * cur_marks[_t] for _t in shares if _t in cur_marks)
    gross_after_sell = _gross_exposure({t: float(shares.get(t, 0.0)) * cur_marks[t] / equity_cur for t in shares if t in cur_marks} if equity_cur != 0 else {}, leverage_multiples if isinstance(leverage_multiples, Mapping) else {})
    for _t in sorted(target):
        _o = opens.get(_t) if isinstance(opens, Mapping) else None
        if not _fillable(_o):
            unfilled.append(_t)
            continue
        if budgets.get(_t, 0.0) - traded.get(_t, 0.0) <= 1e-12 and float(target[_t]) * equity_cur / float(_o) - float(shares.get(_t, 0.0)) > 1e-12:
            unfilled.append(_t)
            continue
        price = float(_o)
        mult = abs(int(float(leverage_multiples[_t])))
        val = float(shares.get(_t, 0.0)) * price
        gross_val = sum(float(shares.get(k, 0.0)) * cur_marks[k] * abs(int(float(leverage_multiples[k]))) for k in shares if k in cur_marks)
        w_eff = min(w_lim, float(target[_t]))
        single_cap = (w_eff * equity_cur - val) / (1.0 + w_eff * fee)
        cash_cap = (cash - c_lim * equity_cur) / (1.0 + (1.0 - c_lim) * fee)
        gross_cap = (g_lim * equity_cur - gross_val) / (mult + g_lim * fee)
        remaining_adv = budgets[_t] - traded.get(_t, 0.0)
        amount = max(0.0, min(single_cap, cash_cap, gross_cap, remaining_adv))
        qty = amount / price if lot_size is None else math.floor(amount / price)
        if qty <= 0:
            continue
        notional = float(qty) * price
        cost = float(cost_model.charge(notional))
        cash -= notional + cost
        shares[_t] = float(shares.get(_t, 0.0)) + float(qty)
        cur_marks[_t] = price
        equity_cur -= cost
        traded[_t] = traded.get(_t, 0.0) + notional
        total_cost += cost
        total_notional += notional
        from src.backtest.execution import Fill as _Fill

        fills.append(_Fill(ticker=_t, execution_date=exec_date, price=price, target_weight=float(target[_t])))
    open_marks: dict[str, float] = {}
    for _t in shares:
        _o = opens.get(_t) if isinstance(opens, Mapping) else None
        _p = prev_closes.get(_t) if isinstance(prev_closes, Mapping) else None
        if _fillable(_o):
            open_marks[_t] = float(_o)  # type: ignore[arg-type]
        elif _fillable(_p):
            open_marks[_t] = float(_p)  # type: ignore[arg-type]
    open_equity = cash + sum(float(shares[_t]) * open_marks[_t] for _t in shares if _t in open_marks)
    close_marks: dict[str, float] = {}
    for _t in shares:
        _c = closes.get(_t) if isinstance(closes, Mapping) else None
        _o = opens.get(_t) if isinstance(opens, Mapping) else None
        _p = prev_closes.get(_t) if isinstance(prev_closes, Mapping) else None
        if _fillable(_c):
            close_marks[_t] = float(_c)  # type: ignore[arg-type]
        elif _fillable(_p):
            close_marks[_t] = float(_p)  # type: ignore[arg-type]
    equity_close = cash + sum(float(shares[_t]) * close_marks[_t] for _t in shares if _t in close_marks)
    w_before = prior_state.weights_at_prices(dict(opens) if opens else dict(prev_closes))
    w_after_open = {t: float(shares[t]) * open_marks[t] / open_equity for t in shares if t in open_marks and open_equity != 0 and abs(float(shares[t]) * open_marks[t] / open_equity) > 1e-12}
    # Turnover is measured against the intended open allocation.  Using the
    # post-fee marked weights would make the diagnostic depend on commission
    # rounding rather than on the rebalance decision itself.
    intended_open = {
        t: (0.0 if is_cash else min(w_lim, max(0.0, float(target.get(t, 0.0)))))
        for t in set(w_before) | set(target)
    }
    turnover = sum(abs(float(intended_open.get(t, 0.0)) - float(w_before.get(t, 0.0))) for t in intended_open)
    wac = {t: float(shares[t]) * close_marks[t] / equity_close for t in shares if t in close_marks and equity_close != 0 and abs(float(shares[t]) * close_marks[t] / equity_close) > 1e-12}
    mults = leverage_multiples if isinstance(leverage_multiples, Mapping) else {}
    target_gross = _gross_exposure(target, mults)
    post_gross = _gross_exposure(w_after_open, mults)
    close_gross = _gross_exposure(wac, mults)
    prior_gross = _gross_exposure({t: float(held[t]) * marks[t] / equity for t in held} if equity != 0 else {}, mults)
    limit = g_lim
    exec_viol = bool(post_gross > limit + 1e-9 and len(fills) > 0 and not is_hold and post_gross > prior_gross + 1e-9)
    carry = bool(close_gross > limit + 1e-9 and not exec_viol)
    delever = bool(close_gross > limit + 1e-9)
    residual = float(limit) - float(gross_after_sell) if (not is_hold and not is_cash) else 0.0
    diag = SessionTransitionDiagnostics(turnover_weight=float(turnover), transaction_cost=float(total_cost), fill_count=len(fills), unfilled_count=len(sorted(set(unfilled))), target_gross=float(target_gross), post_fill_gross=float(post_gross), close_realized_gross=float(close_gross), effective_gross=float(close_gross), gross_violation=bool(exec_viol), cash_session=bool(is_cash), execution_gross_violation=bool(exec_viol), carry_gross_drift=bool(carry), delever_required_next_session=bool(delever), gross_after_sell=float(gross_after_sell) if (not is_hold and not is_cash and limit is not None) else 0.0, residual_gross_budget=float(residual) if (not is_hold and not is_cash and limit is not None) else 0.0)
    return PortfolioTransitionResult(state=PortfolioLedgerState(cash=float(cash), shares={t: float(q) for t, q in shares.items()}), equity_close=float(equity_close), session_return=float(equity_close / equity_prev - 1.0) if equity_prev != 0 else 0.0, weights_after_close=dict(wac), diagnostics=diag, fills=tuple(fills), unfilled=tuple(sorted(set(unfilled))))
