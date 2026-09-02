# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.backtest.costs import CostModel
from src.backtest.execution import NextOpenExecution, is_open_fillable
from src.backtest.liquidity import cap_target_weights_by_adv
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
) -> PortfolioTransitionResult:
    try:
        equity_prev = prior_state.equity_at_prices(prev_closes) if prev_closes else prior_state.equity_at_prices(opens)
    except Exception:
        equity_prev = float(prior_state.cash)
    try:
        equity_open = prior_state.equity_at_prices(opens) if opens else equity_prev
    except Exception:
        equity_open = equity_prev
    if equity_open == 0 or equity_open != equity_open:
        equity_open = equity_prev
    # INV-WEIGHT-1 split
    try:
        weights_before_close = prior_state.weights_at_prices(prev_closes) if prev_closes else prior_state.weights_at_prices(opens)
    except Exception:
        weights_before_close = {}
    try:
        weights_before_open = prior_state.weights_at_prices(opens) if opens else dict(weights_before_close)
    except Exception:
        weights_before_open = dict(weights_before_close)

    is_hold = bool(getattr(intent, "kind", "") == "hold")
    is_cash = bool(getattr(intent, "kind", "") == "cash")

    target: dict[str, float] = {}
    if is_hold:
        target = {}
    elif is_cash:
        target = {str(k): 0.0 for k in prior_state.shares.keys()}
        try:
            w_int = dict(getattr(intent, "weights", {}))
            for k, v in w_int.items():
                target[str(k)] = 0.0
        except Exception:
            pass
    else:
        raw = {}
        try:
            raw = dict(getattr(intent, "weights", {})) if getattr(intent, "weights", None) is not None else {}
        except Exception:
            raw = {}
        try:
            from src.portfolio.constraints import normalize_weights

            if raw:
                target = normalize_weights(raw, max_weight=1.0)
            else:
                target = {}
        except Exception:
            target = {str(k): float(v) for k, v in raw.items()} if raw else {}
        if exposure_limits is not None and target:
            try:
                max_single, max_gross, min_cash = exposure_limits
                from src.portfolio.constraints import apply_portfolio_exposure_limits

                mults = leverage_multiples if isinstance(leverage_multiples, Mapping) else {str(k): 1 for k in target.keys()}
                target = apply_portfolio_exposure_limits(
                    target,
                    mults,
                    max_single_weight=float(max_single),
                    max_gross_exposure=float(max_gross),
                    min_cash=float(min_cash),
                )
            except Exception:
                pass

    # INV-GROSS-4 target_gross before execution
    target_gross = _gross_exposure(target, leverage_multiples if isinstance(leverage_multiples, Mapping) else {})

    # INV-WEIGHT-2 ADV cap uses weights_before_open
    if not is_hold and not is_cash and target is not None and adv_by_ticker is not None and max_order_to_adv is not None:
        if target:
            try:
                adv_map = {str(k): float(v) for k, v in dict(adv_by_ticker).items() if v is not None}
                target = cap_target_weights_by_adv(target, weights_before_open, float(equity_open), adv_map, float(max_order_to_adv))
            except Exception:
                pass

    fills: list[Fill] = []
    unfilled: tuple[str, ...] = ()
    if is_hold:
        fills = []
        unfilled = ()
        new_weights: dict[str, float] = {}
    else:
        if target is not None and len(target) > 0:
            if execution is not None and panel is not None and hasattr(execution, "resolve"):
                try:
                    fl, uf = execution.resolve(target, panel, decision_date)
                    fills = list(fl)
                    if isinstance(uf, (list, tuple)):
                        unfilled = tuple(str(x) for x in uf)
                    else:
                        unfilled = tuple(str(x) for x in list(uf))
                except Exception:
                    fills = []
                    unfilled = tuple(sorted(target.keys()))
            else:
                fills = []
                unfilled_list: list[str] = []
                for tk, w in target.items():
                    op = opens.get(str(tk)) if isinstance(opens, Mapping) else None
                    if not is_open_fillable(op):
                        unfilled_list.append(str(tk))
                        continue
                    from src.backtest.execution import Fill as _Fill

                    exec_date = decision_date
                    try:
                        if execution is not None and hasattr(execution, "calendar"):
                            exec_date = execution.calendar.next_session(decision_date)
                    except Exception:
                        exec_date = decision_date
                    fills.append(_Fill(ticker=str(tk), execution_date=exec_date, price=float(op), target_weight=float(w)))  # type: ignore[arg-type]
                unfilled = tuple(sorted(unfilled_list))
        else:
            fills = []
            unfilled = ()
        new_weights = {str(f.ticker): float(f.target_weight) for f in fills} if fills else {}

    # INV-WEIGHT-3 weights_after_open uses weights_before_open baseline
    weights_after_open: dict[str, float] = {}
    turnover_weight = 0.0
    if is_hold:
        # HOLD intent sessions produce post_fill_gross equal to open effective gross without turnover
        # For hold, weights_after_open is weights_before_open filtered
        weights_after_open = {k: float(v) for k, v in weights_before_open.items() if abs(float(v)) > 1e-12}
        turnover_weight = 0.0
        transaction_cost = 0.0
    else:
        union_keys = set(weights_before_open.keys()) | set(target.keys()) | set(new_weights.keys()) | set(unfilled)
        for tk in union_keys:
            tk_s = str(tk)
            if tk_s in unfilled:
                weights_after_open[tk_s] = float(weights_before_open.get(tk_s, 0.0))
            elif tk_s in new_weights:
                weights_after_open[tk_s] = float(new_weights[tk_s])
            elif tk_s in target:
                weights_after_open[tk_s] = float(target[tk_s])
            else:
                weights_after_open[tk_s] = 0.0
        weights_after_open = {k: float(v) for k, v in weights_after_open.items() if abs(float(v)) > 1e-12}
        all_t = set(weights_before_open.keys()) | set(weights_after_open.keys())
        turnover_weight = sum(abs(float(weights_after_open.get(t, 0.0)) - float(weights_before_open.get(t, 0.0))) for t in all_t)
        try:
            traded_notional = float(turnover_weight) * float(equity_open)
            transaction_cost = float(cost_model.charge(traded_notional)) if traded_notional else 0.0
        except Exception:
            transaction_cost = 0.0
    if is_hold:
        transaction_cost = 0.0

    equity_after_cost = float(equity_open) - float(transaction_cost)
    if equity_after_cost < 0:
        equity_after_cost = 0.0

    shares_after: dict[str, float] = {}
    cash_after: float
    if is_hold:
        shares_after = dict(prior_state.shares)
        cash_after = float(prior_state.cash) - float(transaction_cost) if transaction_cost else float(prior_state.cash)
    else:
        cash_after = float(equity_after_cost)
        unfilled_set = set(unfilled)
        for tk in list(weights_after_open.keys()):
            if tk in unfilled_set:
                prior_sh = float(prior_state.shares.get(tk, 0.0))
                shares_after[tk] = prior_sh
                continue
            w = float(weights_after_open[tk])
            if abs(w) < 1e-12:
                continue
            op = opens.get(tk) if isinstance(opens, Mapping) else None
            if op is None or not is_open_fillable(op):
                prior_sh = float(prior_state.shares.get(tk, 0.0))
                if abs(prior_sh) > 1e-12:
                    shares_after[tk] = prior_sh
                continue
            try:
                opf = float(op)
                if opf <= 0 or opf != opf:
                    continue
            except Exception:
                continue
            sh = w * float(equity_after_cost) / float(opf)
            if abs(sh) > 1e-12:
                shares_after[tk] = float(sh)
        open_value = 0.0
        for tk, sh in shares_after.items():
            op = opens.get(tk) if isinstance(opens, Mapping) else None
            if op is None:
                continue
            try:
                opf = float(op)
                open_value += float(sh) * opf
            except Exception:
                continue
        cash_after = float(equity_after_cost) - float(open_value)

    equity_close = float(cash_after)
    for tk, sh in shares_after.items():
        cl = closes.get(tk) if isinstance(closes, Mapping) else None
        if cl is None:
            cl = opens.get(tk) if isinstance(opens, Mapping) else None
        if cl is None:
            continue
        try:
            clf = float(cl)
            shf = float(sh)
            if clf != clf or shf != shf:
                continue
            equity_close += shf * clf
        except Exception:
            continue
    if equity_close < 0:
        equity_close = 0.0

    if equity_prev != 0:
        session_return = float(equity_close / float(equity_prev) - 1.0)
    else:
        session_return = 0.0

    weights_after_close: dict[str, float] = {}
    if equity_close != 0:
        for tk, sh in shares_after.items():
            cl = closes.get(tk) if isinstance(closes, Mapping) else None
            if cl is None:
                cl = opens.get(tk) if isinstance(opens, Mapping) else None
            if cl is None:
                continue
            try:
                clf = float(cl)
                shf = float(sh)
                w = shf * clf / float(equity_close)
                if abs(w) > 1e-12:
                    weights_after_close[tk] = float(w)
            except Exception:
                continue
    # INV-GROSS-2 post_fill_gross from open
    post_fill_gross = _gross_exposure(weights_after_open, leverage_multiples if isinstance(leverage_multiples, Mapping) else {})
    close_realized_gross = _gross_exposure(weights_after_close, leverage_multiples if isinstance(leverage_multiples, Mapping) else {})
    effective_gross = float(close_realized_gross)
    gross_violation = False
    if exposure_limits is not None:
        try:
            _, max_gross, _ = exposure_limits
            gross_limit = float(max_gross)
            gross_violation = bool(post_fill_gross > gross_limit + 1e-9)
        except Exception:
            gross_violation = False
    fill_count = len(fills)
    unfilled_count = len(unfilled)
    diagnostics = SessionTransitionDiagnostics(
        turnover_weight=float(turnover_weight),
        transaction_cost=float(transaction_cost),
        fill_count=int(fill_count),
        unfilled_count=int(unfilled_count),
        target_gross=float(target_gross),
        post_fill_gross=float(post_fill_gross),
        close_realized_gross=float(close_realized_gross),
        effective_gross=float(effective_gross),
        gross_violation=bool(gross_violation),
        cash_session=bool(is_cash),
    )
    state = PortfolioLedgerState(cash=float(cash_after), shares=dict(shares_after))
    return PortfolioTransitionResult(
        state=state,
        equity_close=float(equity_close),
        session_return=float(session_return),
        weights_after_close=dict(weights_after_close),
        diagnostics=diagnostics,
        fills=tuple(fills),
        unfilled=tuple(unfilled),
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
        return RollingDiagnostics(
            gross_violation_count=0,
            effective_gross_max=0.0,
            turnover_mean=0.0,
            fill_count=0,
            unfilled_count=0,
        )
    gross_violation_count = 0
    effective_gross_max = 0.0
    turnover_sum = 0.0
    fill_sum = 0
    unfilled_sum = 0
    for s in sessions:
        try:
            # per INV-GROSS-3 violation count based on post_fill gross violation flag, not close drift
            if bool(getattr(s, "gross_violation", False)):
                gross_violation_count += 1
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
    return RollingDiagnostics(
        gross_violation_count=int(gross_violation_count),
        effective_gross_max=float(effective_gross_max),
        turnover_mean=float(turnover_mean),
        fill_count=int(fill_sum),
        unfilled_count=int(unfilled_sum),
    )
