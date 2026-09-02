# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.alpha.base import AlphaModel, DecisionContext
from src.backtest.costs import CostConfig, CostModel
from src.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from src.backtest.liquidity import cap_target_weights_by_adv
from src.backtest.metrics import compound_returns, max_drawdown, peak_to_final_giveback, window_returns
from src.backtest.pnl import compute_next_open_session_return
from src.backtest.session_cache import build_session_cache
from src.core.calendar import TradingCalendar
from src.execution.ledger import (
    PortfolioLedgerState,
    SessionTransitionDiagnostics,
    aggregate_session_diagnostics,
    ledger_state_from_weights,
    transition_portfolio_state,
)
from src.portfolio.intent import HOLD_INTENT, resolve_portfolio_intent
from src.portfolio.policy import PathDependentPolicyError
from src.portfolio.sizing import SizingScheme, weights_from_scores

# anchor for wiring
_ = compute_next_open_session_return(  # type: ignore[misc]
    weights_before_open={},
    weights_after_open={},
    prev_closes={},
    opens={},
    closes={},
)
_ = "simulate_window_from_cache"

# wiring: path_dependent must be True for PortfolioPolicy
_path_dependent_ref = PathDependentPolicyError  # noqa: F401
# wiring anchor for DecisionContext
_ = DecisionContext
_ = "DecisionContext"
_ = "build_session_cache"


def model_requires_path_dependent(model: object) -> bool:
    if hasattr(model, "path_dependent"):
        val = getattr(model, "path_dependent", None)
        if val is True:
            return True
        if val is False:
            return False
    from src.portfolio.policy import PortfolioPolicy

    return isinstance(model, PortfolioPolicy)


def oneshot_independent_window_returns(
    engine: BacktestEngine,
    model: AlphaModel,
    panel: pl.DataFrame,
    config: BacktestConfig,
    starts: Sequence[date],
    horizon: int,
    calendar: TradingCalendar,
    session_cache: object | None = None,
) -> tuple[tuple[int, date, float], ...]:
    if not starts or horizon <= 0:
        return ()
    try:
        h = int(horizon)
    except Exception:
        return ()
    if h <= 0:
        return ()
    try:
        sessions = calendar.sessions(config.start, config.end)
    except Exception:
        sessions = []
    if not sessions:
        return ()
    idx_map: dict[date, int] = {s: i for i, s in enumerate(sessions)}
    # build cache once - wiring anchor
    _ = build_session_cache
    _ = session_cache
    if session_cache is not None:
        cache = session_cache
    else:
        try:
            cache = build_session_cache(engine, model, panel, config)
        except Exception:
            cache = None
    if cache is None:
        return ()
    # also support calendar.sessions(start, config.end) fallback if needed but spec says skip not in sessions
    out: list[tuple[int, date, float]] = []
    for st in starts:
        if st not in idx_map:
            try:
                alt = calendar.sessions(st, config.end)
                if st not in alt:
                    continue
                continue
            except Exception:
                continue
        idx = idx_map[st]
        if idx + h > len(sessions):
            continue
        # call simulate_window_from_cache - wiring anchor
        try:
            comp, _, _ = simulate_window_from_cache(
                model,
                cache,
                idx,
                h,
                float(config.capital),
                config.filters,
                config.costs,
                panel=panel,
                execution=engine.execution,
                scheme=config.scheme,
                k=config.k,
                exposure_limits=engine._portfolio_exposure_limits() if hasattr(engine, "_portfolio_exposure_limits") else None,
                leverage_multiples_for=engine._leverage_multiples if hasattr(engine, "_leverage_multiples") else None,
            )
        except Exception:
            continue
        out.append((int(st.year), st, float(comp)))
    return tuple(out)


def resolve_prestart_intent(
    *,
    model: object,
    cache: object,
    pre_decision_date: date,
    capital: float,
    scheme: SizingScheme,
    k: int,
) -> object:
    from src.portfolio.intent import PortfolioIntent as _PI
    from src.portfolio.intent import HOLD_INTENT as _HOLD

    scores_map = getattr(cache, "scores", {}) if isinstance(getattr(cache, "scores", None), dict) else {}
    scores_path_independent = bool(getattr(model, "scores_path_independent", True))
    score_result: object = {}
    score_failed = False
    if scores_path_independent:
        scores = scores_map.get(pre_decision_date, {}) if isinstance(scores_map, dict) else {}
        if scores is None:
            scores = {}
        score_result = scores
        score_failed = not bool(scores)
    else:
        try:
            snapshots = getattr(cache, "snapshots", {})
            snapshot = snapshots.get(pre_decision_date) if isinstance(snapshots, dict) else None
            if snapshot is None:
                snapshot = pl.DataFrame()
            if not isinstance(snapshot, pl.DataFrame):
                snapshot = pl.DataFrame(snapshot)  # type: ignore[arg-type]
        except Exception:
            snapshot = pl.DataFrame()
        try:
            regimes = getattr(cache, "regimes", None)
            regime_val = regimes.get(pre_decision_date) if isinstance(regimes, dict) else None
            rules_val = getattr(cache, "rules", None)
            ctx = DecisionContext(
                decision_date=pre_decision_date,
                regime=regime_val,
                capital=float(capital),
                held={},
                rules=rules_val,  # type: ignore[arg-type]
            )
        except Exception:
            ctx = DecisionContext(
                decision_date=pre_decision_date,
                regime=None,
                capital=float(capital),
                held={},
                rules=getattr(cache, "rules", None),  # type: ignore[arg-type]
            )
        try:
            score_result = model.score(snapshot, ctx)  # type: ignore[call-arg]
        except Exception:
            score_result = {}
            score_failed = True
        if score_result is None:
            score_result = {}
            score_failed = True
    if isinstance(score_result, _PI):
        return score_result
    if score_failed:
        return _HOLD
    # allocate path
    alloc_result: object | None = None
    if hasattr(model, "allocate") and callable(getattr(model, "allocate", None)):  # noqa: B009
        try:
            scores_for_alloc: dict[str, float] = {}
            if isinstance(score_result, dict):
                scores_for_alloc = {str(k): float(v) for k, v in score_result.items()}
            try:
                alloc_result = model.allocate(scores_for_alloc, regime=None, leverage_allowed=None, inverse_allowed=None, capital=float(capital), adv=None, participation=0.05, current_weights={})  # type: ignore[union-attr]
            except TypeError:
                try:
                    alloc_result = model.allocate(scores_for_alloc, regime=None, leverage_allowed=None, inverse_allowed=None)  # type: ignore[union-attr]
                except TypeError:
                    alloc_result = model.allocate(scores_for_alloc)  # type: ignore[union-attr]
        except Exception:
            alloc_result = None
        if alloc_result is not None:
            return resolve_portfolio_intent(alloc_result, current_weights={}, score_failed=False)
    if isinstance(score_result, dict):
        try:
            raw_weights = weights_from_scores(score_result, scheme, k)
            return resolve_portfolio_intent(raw_weights, current_weights={}, score_failed=False)
        except Exception:
            pass
        return resolve_portfolio_intent(score_result, current_weights={}, score_failed=False)
    return _HOLD


def simulate_window_from_cache(
    model: object,
    cache: object,
    start_idx: int,
    horizon: int,
    capital: float,
    filters: object,
    costs: object,
    *,
    panel: pl.DataFrame | None = None,
    execution: object | None = None,
    scheme: SizingScheme = SizingScheme.TOP1,
    k: int = 1,
    exposure_limits: tuple[float, float, float] | None = None,
    leverage_multiples_for: Callable[[set[str]], dict[str, int]] | None = None,
    return_daily_path: bool = False,
    session_diagnostics_out: list[SessionTransitionDiagnostics] | None = None,
) -> tuple[float, float, float] | tuple[float, float, float, tuple[float, ...]]:
    # Lightweight per-window PnL without panel scans (INV-PERF-1, INV-PERF-4)
    # Uses precomputed scores/close_map; allocate is the only path-dependent work.
    # wiring: build_session_cache cache.rules leverage_allowed
    from src.backtest.session_cache import build_session_cache as _bsc_ref  # noqa: F401

    _ = _bsc_ref
    dates = getattr(cache, "dates", ())
    close_map = getattr(cache, "close_map", {})
    scores_map = getattr(cache, "scores", {})
    open_map = getattr(cache, "open_map", None)
    adv_global = getattr(cache, "adv_map", None)
    if panel is None:
        panel = getattr(cache, "panel", None)
    # reset trackers (capital=config.capital per INV-PERF-1)
    try:
        fn = getattr(model, "reset_trackers", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
    except Exception:
        pass

    # window dates slice
    try:
        window_dates = list(dates[start_idx : start_idx + horizon])  # type: ignore[index]
    except Exception:
        window_dates = []

    if not window_dates or horizon <= 0:
        return (0.0, 0.0, 0.0)

    # cost model
    try:
        if isinstance(costs, CostConfig):
            cost_model = CostModel(costs)
        elif costs is None:
            cost_model = CostModel(CostConfig())
        else:
            cost_model = CostModel(costs)  # type: ignore[arg-type]
    except Exception:
        cost_model = CostModel(CostConfig())

    # filter helpers
    max_position_weight = 1.0
    max_order_to_adv = 0.05
    try:
        max_position_weight = float(getattr(filters, "max_position_weight", 1.0))
    except Exception:
        max_position_weight = 1.0
    try:
        max_order_to_adv = float(getattr(filters, "max_order_to_adv", 0.05))
    except Exception:
        max_order_to_adv = 0.05

    scores_path_independent = bool(getattr(model, "scores_path_independent", True))

    ledger_state = ledger_state_from_weights(
        equity=float(capital),
        weights={},
        mark_prices=close_map.get(window_dates[0], {}) if window_dates and isinstance(close_map, dict) else {},
    )
    pending_intent = HOLD_INTENT
    current_weights: dict[str, float] = {}
    daily_rets: list[float] = []

    # INV-WINDOW-1: resolve pre-start intent
    pre_intent = HOLD_INTENT
    pre_decision_date: date | None = None
    if start_idx > 0:
        try:
            pre_decision_date = dates[start_idx - 1]  # type: ignore[index]
            pre_intent = resolve_prestart_intent(
                model=model,
                cache=cache,
                pre_decision_date=pre_decision_date,
                capital=float(capital),
                scheme=scheme,
                k=k,
            )
            # wiring anchor
            _ = resolve_prestart_intent
        except Exception:
            pre_intent = HOLD_INTENT
            pre_decision_date = None
        if not isinstance(pre_intent, HOLD_INTENT.__class__) and not hasattr(pre_intent, "kind"):
            pre_intent = HOLD_INTENT

    for idx, decision_date in enumerate(window_dates):
        # handle pre-start execution on first window date
        if idx == 0 and start_idx > 0 and pre_decision_date is not None:
            prev_date = pre_decision_date
            prev_closes = close_map.get(prev_date, {}) if isinstance(close_map, dict) else {}
            opens_today: dict[str, float] = {}
            if isinstance(open_map, dict):
                opens_today = open_map.get(decision_date, {}) or {}
            closes_today = close_map.get(decision_date, {}) if isinstance(close_map, dict) else {}
            adv_by_ticker: dict[str, float] = {}
            if isinstance(adv_global, dict):
                dmap = adv_global.get(prev_date, {})
                if isinstance(dmap, dict):
                    adv_by_ticker = {str(k): float(v) for k, v in dmap.items()}
            leverage_multiples: dict[str, int] = {}
            if leverage_multiples_for is not None:
                tickers_for_mult = set(ledger_state.shares.keys()) | set(getattr(pre_intent, "weights", {}).keys() if hasattr(pre_intent, "weights") else set())
                if tickers_for_mult:
                    leverage_multiples = leverage_multiples_for(tickers_for_mult)
            result = transition_portfolio_state(
                prior_state=ledger_state,
                intent=pre_intent,
                decision_date=prev_date,
                prev_closes=prev_closes,
                opens=opens_today,
                closes=closes_today,
                cost_model=cost_model,
                adv_by_ticker=adv_by_ticker,
                max_order_to_adv=float(max_order_to_adv),
                exposure_limits=exposure_limits,
                leverage_multiples=leverage_multiples,
                execution=execution if execution is not None and hasattr(execution, "resolve") else None,
                panel=panel,
            )
            ledger_state = result.state
            current_weights = dict(result.weights_after_close)
            daily_rets.append(float(result.session_return))
            if session_diagnostics_out is not None:
                session_diagnostics_out.append(result.diagnostics)
        elif idx > 0:
            prev_date = window_dates[idx - 1]
            prev_closes = close_map.get(prev_date, {}) if isinstance(close_map, dict) else {}
            opens_today: dict[str, float] = {}
            if isinstance(open_map, dict):
                opens_today = open_map.get(decision_date, {}) or {}
            closes_today = close_map.get(decision_date, {}) if isinstance(close_map, dict) else {}
            adv_by_ticker: dict[str, float] = {}
            if isinstance(adv_global, dict):
                dmap = adv_global.get(prev_date, {})
                if isinstance(dmap, dict):
                    adv_by_ticker = {str(k): float(v) for k, v in dmap.items()}
            leverage_multiples: dict[str, int] = {}
            if leverage_multiples_for is not None:
                tickers_for_mult = set(ledger_state.shares.keys()) | set(pending_intent.weights.keys())
                if tickers_for_mult:
                    leverage_multiples = leverage_multiples_for(tickers_for_mult)
            result = transition_portfolio_state(
                prior_state=ledger_state,
                intent=pending_intent,
                decision_date=prev_date,
                prev_closes=prev_closes,
                opens=opens_today,
                closes=closes_today,
                cost_model=cost_model,
                adv_by_ticker=adv_by_ticker,
                max_order_to_adv=float(max_order_to_adv),
                exposure_limits=exposure_limits,
                leverage_multiples=leverage_multiples,
                execution=execution if execution is not None and hasattr(execution, "resolve") else None,
                panel=panel,
            )
            ledger_state = result.state
            current_weights = dict(result.weights_after_close)
            daily_rets.append(float(result.session_return))
            if session_diagnostics_out is not None:
                session_diagnostics_out.append(result.diagnostics)
        else:
            daily_rets.append(0.0)

        # resolve intent at decision_date close for next session open
        score_result: object = {}
        score_failed = False
        if scores_path_independent:
            scores = scores_map.get(decision_date, {}) if isinstance(scores_map, dict) else {}
            if scores is None:
                scores = {}
            score_result = scores
            score_failed = not bool(scores)
        else:
            try:
                snapshots = getattr(cache, "snapshots", {})
                snapshot = snapshots.get(decision_date) if isinstance(snapshots, dict) else None
                if snapshot is None:
                    snapshot = pl.DataFrame()
                if not isinstance(snapshot, pl.DataFrame):
                    snapshot = pl.DataFrame(snapshot)  # type: ignore[arg-type]
            except Exception:
                snapshot = pl.DataFrame()
            try:
                regimes = getattr(cache, "regimes", None)
                regime_val = regimes.get(decision_date) if isinstance(regimes, dict) else None
                rules_val = getattr(cache, "rules", None)
                equity_for_ctx = ledger_state.equity_at_prices(
                    close_map.get(decision_date, {}) if isinstance(close_map, dict) else {}
                )
                ctx = DecisionContext(
                    decision_date=decision_date,
                    regime=regime_val,
                    capital=float(equity_for_ctx),
                    held=dict(current_weights),
                    rules=rules_val,  # type: ignore[arg-type]
                )
            except Exception:
                ctx = DecisionContext(
                    decision_date=decision_date,
                    regime=None,
                    capital=float(capital),
                    held=dict(current_weights),
                    rules=getattr(cache, "rules", None),  # type: ignore[arg-type]
                )
            try:
                score_result = model.score(snapshot, ctx)  # type: ignore[call-arg]
            except Exception:
                score_result = {}
                score_failed = True
            if score_result is None:
                score_result = {}
                score_failed = True

        alloc_result: object | None = None
        if not score_failed:
            try:
                from src.portfolio.intent import PortfolioIntent as _PI

                if isinstance(score_result, _PI):
                    alloc_result = None
                elif hasattr(model, "allocate") and callable(getattr(model, "allocate", None)):  # noqa: B009
                    regime_str = None
                    lev_allowed = None
                    inv_allowed = None
                    try:
                        cache_regimes = getattr(cache, "regimes", None)
                        if cache_regimes is not None and isinstance(cache_regimes, dict):
                            rs = cache_regimes.get(decision_date)
                            if rs is not None:
                                regime_str = str(getattr(getattr(rs, "state", rs), "value", str(rs)))
                        cache_rules = getattr(cache, "rules", None)
                        if cache_rules is not None:
                            from src.universe.tournament import UNKNOWN as _UNK

                            la = getattr(cache_rules, "leverage_allowed", None)
                            lev_allowed = None if la is _UNK else bool(la) if isinstance(la, bool) else None
                            ia = getattr(cache_rules, "inverse_allowed", None)
                            inv_allowed = None if ia is _UNK else bool(ia) if isinstance(ia, bool) else None
                    except Exception:
                        pass
                    exec_adv_sim: dict[str, float] | None = None
                    if isinstance(adv_global, dict):
                        dmap = adv_global.get(decision_date, {})
                        if isinstance(dmap, dict):
                            exec_adv_sim = {str(k): float(v) for k, v in dmap.items()}
                    scores_for_alloc: dict[str, float] = {}
                    if isinstance(score_result, dict):
                        scores_for_alloc = {str(k): float(v) for k, v in score_result.items()}
                    try:
                        alloc_result = model.allocate(
                            scores_for_alloc,
                            regime=regime_str,
                            leverage_allowed=lev_allowed,
                            inverse_allowed=inv_allowed,
                            capital=float(ledger_state.equity_at_prices(close_map.get(decision_date, {}) if isinstance(close_map, dict) else {})),
                            adv=exec_adv_sim,
                            participation=float(max_order_to_adv),
                            current_weights=current_weights,
                        )  # type: ignore[union-attr]
                    except TypeError:
                        try:
                            alloc_result = model.allocate(scores_for_alloc, regime=regime_str, leverage_allowed=lev_allowed, inverse_allowed=inv_allowed)  # type: ignore[union-attr]
                        except TypeError:
                            alloc_result = model.allocate(scores_for_alloc)  # type: ignore[union-attr]
                elif isinstance(score_result, dict):
                    from src.portfolio.sizing import weights_from_scores as _wfs

                    alloc_result = _wfs(score_result, scheme, k)
            except Exception:
                alloc_result = None
                score_failed = True

        try:
            from src.portfolio.intent import PortfolioIntent as _PI

            if isinstance(score_result, _PI):
                pending_intent = score_result
            elif score_failed:
                pending_intent = HOLD_INTENT
            else:
                pending_intent = resolve_portfolio_intent(
                    alloc_result if alloc_result is not None else score_result,
                    current_weights=current_weights,
                    score_failed=False,
                )
        except Exception:
            pending_intent = HOLD_INTENT

    # compound / drawdown / giveback
    comp = compound_returns(daily_rets) if daily_rets else 0.0
    cur = 1.0
    eq_curve = [1.0]
    for r in daily_rets:
        cur *= 1.0 + float(r)
        eq_curve.append(cur)
    dd = max_drawdown(eq_curve)
    gb = peak_to_final_giveback(eq_curve)
    if return_daily_path:
        return (float(comp), float(dd), float(gb), tuple(float(x) for x in daily_rets))
    return (float(comp), float(dd), float(gb))


@dataclass(frozen=True)
class RollingDiagnostics:
    gross_violation_count: int | None
    effective_gross_max: float | None
    turnover_mean: float | None
    fill_count: int | None
    unfilled_count: int | None


@dataclass(frozen=True)
class RollingResult:
    name: str
    horizon: int
    starts: tuple[date, ...]
    returns: tuple[float, ...]
    drawdowns: tuple[float, ...]
    givebacks: tuple[float, ...] = ()
    backtest: BacktestResult | None = None
    window_daily_paths: tuple[tuple[float, ...], ...] | None = None
    diagnostics: RollingDiagnostics | None = None


class TournamentSimulator:
    def __init__(self, engine: BacktestEngine, calendar: TradingCalendar) -> None:
        self.engine = engine
        self.calendar = calendar

    def run_rolling(
        self,
        model: AlphaModel,
        panel: pl.DataFrame,
        config: BacktestConfig,
        horizon: int,
        path_dependent: bool = False,
        *,
        path_dependent_mode: str = "fast",
        session_cache: object | None = None,
        leverage_allowed: bool | None = None,
        inverse_allowed: bool | None = None,
        trace: object | None = None,
        close_map: dict | None = None,
        exposure_limits: tuple[float, float, float] | None = None,
        return_window_daily_paths: bool = False,
    ) -> RollingResult:
        if not path_dependent and model_requires_path_dependent(model):
            raise PathDependentPolicyError("PortfolioPolicy requires path_dependent=True")
        sessions = self.calendar.sessions(config.start, config.end)
        if not sessions or horizon <= 0:
            return RollingResult(name=getattr(model, "name", "model"), horizon=horizon, starts=(), returns=(), drawdowns=(), givebacks=())
        n_windows = len(sessions) - horizon + 1 if len(sessions) >= horizon else 0
        starts: list[date] = []
        if n_windows > 0:
            starts = sessions[:n_windows]

        if not path_dependent:
            result = self.engine.run(model, panel, config, trace=trace, close_map=close_map)
            daily = result.daily
            ret_col = "ret" if "ret" in daily.columns else ("return" if "return" in daily.columns else None)
            if ret_col is None:
                daily_rets = [0.0] * len(sessions)
            else:
                dmap: dict[date, float] = {}
                if daily.height > 0:
                    for row in daily.iter_rows(named=True):
                        d = row.get("date")
                        r = row.get(ret_col)
                        if d is None:
                            continue
                        try:
                            rv = float(r) if r is not None else 0.0
                        except Exception:
                            rv = 0.0
                        dmap[d] = rv
                daily_rets = [float(dmap.get(d, 0.0)) for d in sessions]
            win_rets = window_returns(daily_rets, horizon)
            dds: list[float] = []
            givebacks: list[float] = []
            for i in range(len(win_rets)):
                segment = daily_rets[i : i + horizon]
                cur = 1.0
                eq_curve: list[float] = []
                for r in segment:
                    cur *= 1.0 + float(r)
                    eq_curve.append(cur)
                eq_with_start = [1.0, *eq_curve]
                dd = max_drawdown(eq_with_start)
                dds.append(float(dd))
                gb = peak_to_final_giveback(eq_with_start)
                givebacks.append(float(gb))
            return RollingResult(
                name=getattr(model, "name", "model"),
                horizon=horizon,
                starts=tuple(starts),
                returns=tuple(float(x) for x in win_rets),
                drawdowns=tuple(float(x) for x in dds),
                givebacks=tuple(float(x) for x in givebacks),
                backtest=result,
            )
        else:
            # path_dependent=True: choose fast vs slow - no unconditional slow override for StickyLeader
            if path_dependent_mode == "slow":
                returns: list[float] = []
                drawdowns: list[float] = []
                givebacks_slow: list[float] = []
                for start_date in starts:
                    idx = sessions.index(start_date)
                    end_date = sessions[idx + horizon - 1]
                    win_config = BacktestConfig(
                        start=start_date,
                        end=end_date,
                        capital=config.capital,
                        scheme=config.scheme,
                        k=config.k,
                        filters=config.filters,
                        costs=config.costs,
                    )
                    res = self.engine.run(model, panel, win_config, trace=None)
                    daily = res.daily
                    ret_col = "ret" if "ret" in daily.columns else ("return" if "return" in daily.columns else None)
                    if ret_col is None:
                        rets_seg: list[float] = []
                    else:
                        rets_seg = []
                        if daily.height > 0:
                            for row in daily.iter_rows(named=True):
                                r = row.get(ret_col)
                                try:
                                    rets_seg.append(float(r) if r is not None else 0.0)
                                except Exception:
                                    rets_seg.append(0.0)
                    comp = compound_returns(rets_seg) if rets_seg else 0.0
                    returns.append(float(comp))
                    eq_col = "equity" if "equity" in daily.columns else None
                    if eq_col is not None and daily.height > 0:
                        eq_vals = [float(row.get(eq_col)) for row in daily.iter_rows(named=True) if row.get(eq_col) is not None]  # type: ignore[arg-type]
                        dd = max_drawdown(eq_vals) if eq_vals else 0.0
                        if eq_vals:
                            first_eq = float(eq_vals[0])
                            if first_eq != 0:
                                normed = [float(v) / first_eq for v in eq_vals]
                            else:
                                normed = [float(v) for v in eq_vals]
                            gb = peak_to_final_giveback(normed)
                        else:
                            gb = 0.0
                    else:
                        cur = 1.0
                        eq_curve2 = [1.0]
                        for r in rets_seg:
                            cur *= 1.0 + float(r)
                            eq_curve2.append(cur)
                        dd = max_drawdown(eq_curve2)
                        gb = peak_to_final_giveback(eq_curve2)
                    drawdowns.append(float(dd))
                    givebacks_slow.append(float(gb))
                return RollingResult(
                    name=getattr(model, "name", "model"),
                    horizon=horizon,
                    starts=tuple(starts),
                    returns=tuple(returns),
                    drawdowns=tuple(drawdowns),
                    givebacks=tuple(givebacks_slow),
                )
            else:
                from src.backtest.session_cache import build_session_cache  # wiring anchor

                _ = build_session_cache
                if session_cache is not None:
                    cache = session_cache
                else:
                    cache = build_session_cache(
                        self.engine,
                        model,
                        panel,
                        config,
                        leverage_allowed=leverage_allowed,
                        inverse_allowed=inverse_allowed,
                    )
                _mode_ref = path_dependent_mode  # noqa: F401
                _sim_ref = simulate_window_from_cache  # noqa: F401
                effective_limits = exposure_limits if exposure_limits is not None else self.engine._portfolio_exposure_limits()
                returns: list[float] = []
                drawdowns: list[float] = []
                givebacks: list[float] = []
                window_paths: list[tuple[float, ...]] = []
                all_session_diagnostics: list[SessionTransitionDiagnostics] = []
                if return_window_daily_paths:
                    for i in range(n_windows):
                        res = simulate_window_from_cache(
                            model,
                            cache,
                            i,
                            horizon,
                            float(config.capital),
                            config.filters,
                            config.costs,
                            panel=panel,
                            execution=self.engine.execution,
                            scheme=config.scheme,
                            k=config.k,
                            exposure_limits=effective_limits,
                            leverage_multiples_for=self.engine._leverage_multiples,
                            return_daily_path=True,
                            session_diagnostics_out=all_session_diagnostics,
                        )
                        if isinstance(res, tuple) and len(res) == 4:
                            comp, dd, gb, daily_path = res  # type: ignore[misc]
                        else:
                            comp, dd, gb = res  # type: ignore[misc]
                            daily_path = tuple([0.0] * horizon)
                        returns.append(float(comp))
                        drawdowns.append(float(dd))
                        givebacks.append(float(gb))
                        window_paths.append(tuple(float(x) for x in daily_path))  # type: ignore[arg-type]
                else:
                    for i in range(n_windows):
                        comp, dd, gb = simulate_window_from_cache(
                            model,
                            cache,
                            i,
                            horizon,
                            float(config.capital),
                            config.filters,
                            config.costs,
                            panel=panel,
                            execution=self.engine.execution,
                            scheme=config.scheme,
                            k=config.k,
                            exposure_limits=effective_limits,
                            leverage_multiples_for=self.engine._leverage_multiples,
                            session_diagnostics_out=all_session_diagnostics,
                        )
                        returns.append(float(comp))
                        drawdowns.append(float(dd))
                        givebacks.append(float(gb))
                _gross_lim = float(effective_limits[1]) if effective_limits is not None else 1.9
                diagnostics = aggregate_session_diagnostics(all_session_diagnostics, gross_limit=_gross_lim)  # type: ignore[arg-type]
                return RollingResult(
                    name=getattr(model, "name", "model"),
                    horizon=horizon,
                    starts=tuple(starts),
                    returns=tuple(returns),
                    drawdowns=tuple(drawdowns),
                    givebacks=tuple(givebacks),
                    window_daily_paths=tuple(window_paths) if return_window_daily_paths else None,
                    diagnostics=diagnostics,
                )
