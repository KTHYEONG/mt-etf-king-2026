# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.backtest.costs import CostModel
from src.backtest.execution import NextOpenExecution
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
        # filter tiny?
        return {k: float(v) for k, v in out.items()}


@dataclass(frozen=True)
class SessionTransitionDiagnostics:
    turnover_weight: float
    transaction_cost: float
    fill_count: int
    unfilled_count: int
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
    # clamp cash?
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
    # equity levels
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
    # weights before
    try:
        weights_before = prior_state.weights_at_prices(prev_closes) if prev_closes else prior_state.weights_at_prices(opens)
    except Exception:
        weights_before = {}

    is_hold = bool(getattr(intent, "kind", "") == "hold")
    is_cash = bool(getattr(intent, "kind", "") == "cash")

    target: dict[str, float] = {}
    # build target
    if is_hold:
        target = {}
    elif is_cash:
        # liquidate all holdings
        target = {str(k): 0.0 for k in prior_state.shares.keys()}
        # also consider intent weights if any non-empty (should be empty for cash)
        try:
            w_int = dict(getattr(intent, "weights", {}))
            for k, v in w_int.items():
                # cash intent ignores weights, keep 0
                target[str(k)] = 0.0
        except Exception:
            pass
    else:
        raw = {}
        try:
            raw = dict(getattr(intent, "weights", {})) if getattr(intent, "weights", None) is not None else {}
        except Exception:
            raw = {}
        # normalize
        try:
            from src.portfolio.constraints import normalize_weights

            if raw:
                target = normalize_weights(raw, max_weight=1.0)
            else:
                target = {}
        except Exception:
            target = {str(k): float(v) for k, v in raw.items()} if raw else {}
        # exposure limits
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

    # ADV cap
    if not is_hold and not is_cash and target is not None and adv_by_ticker is not None and max_order_to_adv is not None:
        if target:
            try:
                from src.backtest.liquidity import cap_target_weights_by_adv

                # equity for cap is equity_open
                adv_map = {str(k): float(v) for k, v in dict(adv_by_ticker).items() if v is not None}
                target = cap_target_weights_by_adv(target, weights_before, float(equity_open), adv_map, float(max_order_to_adv))
            except Exception:
                pass

    fills: list = []
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
                # fallback: fill if open exists
                fills = []
                unfilled_list: list[str] = []
                for tk, w in target.items():
                    op = opens.get(str(tk)) if isinstance(opens, Mapping) else None
                    if op is None:
                        unfilled_list.append(str(tk))
                        continue
                    try:
                        pf = float(op)
                        if pf <= 0 or pf != pf:
                            unfilled_list.append(str(tk))
                            continue
                    except Exception:
                        unfilled_list.append(str(tk))
                        continue
                    from src.backtest.execution import Fill

                    exec_date = decision_date
                    try:
                        if execution is not None and hasattr(execution, "calendar"):
                            exec_date = execution.calendar.next_session(decision_date)
                    except Exception:
                        exec_date = decision_date
                    fills.append(Fill(ticker=str(tk), execution_date=exec_date, price=float(pf), target_weight=float(w)))
                unfilled = tuple(sorted(unfilled_list))
        else:
            fills = []
            unfilled = ()
        new_weights = {str(f.ticker): float(f.target_weight) for f in fills} if fills else {}

    # weights_after_open
    weights_after_open: dict[str, float] = {}
    turnover_weight = 0.0
    if is_hold:
        weights_after_open = {}
        turnover_weight = 0.0
        transaction_cost = 0.0
    else:
        # build weights_after_open merging
        # union of weights_before, target, new_weights, unfilled
        union_keys = set(weights_before.keys()) | set(target.keys()) | set(new_weights.keys()) | set(unfilled)
        for tk in union_keys:
            tk_s = str(tk)
            if tk_s in unfilled:
                weights_after_open[tk_s] = float(weights_before.get(tk_s, 0.0))
            elif tk_s in new_weights:
                weights_after_open[tk_s] = float(new_weights[tk_s])
            elif tk_s in target:
                weights_after_open[tk_s] = float(target[tk_s])
            else:
                # prior holding not in target (should have been in target as 0 after cap), but if missing treat as 0
                weights_after_open[tk_s] = 0.0
        # filter zero
        weights_after_open = {k: float(v) for k, v in weights_after_open.items() if abs(float(v)) > 1e-12}
        # turnover
        all_t = set(weights_before.keys()) | set(weights_after_open.keys())
        turnover_weight = sum(abs(float(weights_after_open.get(t, 0.0)) - float(weights_before.get(t, 0.0))) for t in all_t)
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

    # build shares_after and cash_after
    shares_after: dict[str, float] = {}
    cash_after: float
    if is_hold:
        shares_after = dict(prior_state.shares)
        # cash reduced by cost
        cash_after = float(prior_state.cash) - float(transaction_cost) if transaction_cost else float(prior_state.cash)
        # ensure cash doesn't go negative due to cost > cash? equity_after_cost already ensures
        # adjust cash to keep equity_after_cost invariant: cash_after = equity_after_cost - sum shares*open
        # but for hold we keep shares, so cash_after should be prior cash - cost
        # verify equity_after_cost == cash_after + sum shares*open
        # if mismatch due to prior cash not equal equity_open - shares*open? It should match
    else:
        # for non-hold, compute shares from weights_after_open
        # For unfilled, keep prior shares instead of recomputed
        cash_after = float(equity_after_cost)
        # first handle unfilled: keep prior shares
        unfilled_set = set(unfilled)
        for tk in list(weights_after_open.keys()):
            if tk in unfilled_set:
                prior_sh = float(prior_state.shares.get(tk, 0.0))
                shares_after[tk] = prior_sh
                # cash not yet adjusted for these shares? We'll adjust later via subtraction of open value
                # We'll need to handle cash After as equity_after - sum shares*open
                # So we can't subtract using recomputed weight method for unfilled
                continue
            w = float(weights_after_open[tk])
            if abs(w) < 1e-12:
                continue
            op = opens.get(tk) if isinstance(opens, Mapping) else None
            if op is None:
                # no open price, keep prior share if any
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
        # cash_after is equity_after_cost minus market value at open of shares_after
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
        # handle case where shares_after empty (cash session) -> cash_after == equity_after_cost

    # equity_close = cash_after + sum shares*closes
    equity_close = float(cash_after)
    for tk, sh in shares_after.items():
        cl = closes.get(tk) if isinstance(closes, Mapping) else None
        if cl is None:
            # if no close, use open? fallback to open value
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
    # if no shares (all cash), equity_close == cash_after
    if equity_close < 0:
        equity_close = 0.0

    # session return relative to equity_prev
    if equity_prev != 0:
        session_return = float(equity_close / float(equity_prev) - 1.0)
    else:
        session_return = 0.0

    # weights_after_close for diagnostics and result
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
    # effective gross
    effective_gross = _gross_exposure(weights_after_close, leverage_multiples if isinstance(leverage_multiples, Mapping) else {})
    gross_limit = None
    gross_violation = False
    if exposure_limits is not None:
        try:
            _, max_gross, _ = exposure_limits
            gross_limit = float(max_gross)
            gross_violation = bool(effective_gross > gross_limit + 1e-9)
        except Exception:
            gross_violation = False
    # diagnostics
    fill_count = len(fills)
    unfilled_count = len(unfilled)
    diagnostics = SessionTransitionDiagnostics(
        turnover_weight=float(turnover_weight),
        transaction_cost=float(transaction_cost),
        fill_count=int(fill_count),
        unfilled_count=int(unfilled_count),
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
    # import locally to avoid circular
    try:
        from src.tournament.simulator import RollingDiagnostics
    except Exception:
        # fallback define minimal
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
            eg = float(getattr(s, "effective_gross", 0.0) or 0.0)
            if eg > effective_gross_max:
                effective_gross_max = float(eg)
            if bool(getattr(s, "gross_violation", False)) or eg > float(gross_limit) + 1e-9:
                gross_violation_count += 1
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

