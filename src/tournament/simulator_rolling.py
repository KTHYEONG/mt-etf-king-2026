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
from src.backtest.session_grid import resolve_session_grid  # noqa: F401
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

# wiring: path_dependent must be True for PortfolioPolicy
# wiring anchor for DecisionContext




from src.tournament.simulator_windows import model_requires_path_dependent, simulate_window_from_cache


@dataclass(frozen=True)
class RollingDiagnostics:
    gross_violation_count: int | None
    effective_gross_max: float | None
    turnover_mean: float | None
    fill_count: int | None
    unfilled_count: int | None
    carry_gross_drift_count: int | None = None
    delever_required_count: int | None = None


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
    window_primary_holdings: tuple[tuple[str | None, ...], ...] | None = None


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
        try:
            sessions = list(resolve_session_grid(self.calendar.sessions(config.start, config.end), panel).sessions)
        except Exception:
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
                slow_traces: list[tuple[str | None, ...]] = []
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
                    try:  # pragma: no cover - malformed legacy trade trace
                        win_sessions = sessions[idx : idx + horizon]
                        by_d: dict[object, str | None] = {}
                        trades = getattr(res, "trades", None)
                        if trades is not None and getattr(trades, "height", 0) > 0:
                            dd_col = "decision_date" if "decision_date" in trades.columns else ("date" if "date" in trades.columns else None)
                            w_col = "weight_after" if "weight_after" in trades.columns else ("weight" if "weight" in trades.columns else None)
                            t_col = "ticker" if "ticker" in trades.columns else None
                            if dd_col and w_col and t_col:
                                for _d in trades[dd_col].unique().to_list():
                                    try:
                                        sub = trades.filter(pl.col(dd_col) == _d)
                                    except Exception:  # pragma: no cover - malformed trade frame
                                        continue
                                    best_t: str | None = None
                                    best_w = 0.0
                                    for _row in sub.iter_rows(named=True):
                                        try:
                                            _wf = float(_row.get(w_col, 0.0))
                                        except Exception:  # pragma: no cover - malformed trade weight
                                            continue
                                        _rt = _row.get(t_col)
                                        _ts = str(_rt) if _rt is not None else None
                                        if _ts is None:
                                            continue
                                        if _wf > best_w or (_wf == best_w and best_t is not None and _ts < best_t):
                                            if _wf > 0:
                                                best_w = _wf
                                                best_t = _ts
                                        elif best_t is None and _wf > 0:
                                            best_w = _wf
                                            best_t = _ts
                                    by_d[_d] = best_t
                        last_h: str | None = None
                        trace: list[str | None] = []
                        for _s in win_sessions:
                            if _s in by_d:
                                last_h = by_d[_s]
                            trace.append(last_h)
                        slow_traces.append(tuple(trace))
                    except Exception:  # pragma: no cover - malformed legacy trade trace
                        slow_traces.append(tuple([None] * horizon))
                return RollingResult(
                    name=getattr(model, "name", "model"),
                    horizon=horizon,
                    starts=tuple(starts),
                    returns=tuple(returns),
                    drawdowns=tuple(drawdowns),
                    givebacks=tuple(givebacks_slow),
                    window_primary_holdings=tuple(slow_traces) if slow_traces else None,
                )
            else:
                from src.backtest.session_cache import build_session_cache  # wiring anchor

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
                fast_traces: list[tuple[str | None, ...]] = []
                all_session_diagnostics: list[SessionTransitionDiagnostics] = []
                if return_window_daily_paths:
                    for i in range(n_windows):
                        primary_holdings: list[str | None] = []
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
                            primary_holdings_out=primary_holdings,
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
                        fast_traces.append(tuple(primary_holdings) if len(primary_holdings) == horizon else tuple((list(primary_holdings) + [None] * horizon)[:horizon]))
                else:
                    for i in range(n_windows):
                        primary_holdings = []
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
                            primary_holdings_out=primary_holdings,
                        )
                        returns.append(float(comp))
                        drawdowns.append(float(dd))
                        givebacks.append(float(gb))
                        fast_traces.append(tuple(primary_holdings) if len(primary_holdings) == horizon else tuple((list(primary_holdings) + [None] * horizon)[:horizon]))
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
                    window_primary_holdings=tuple(fast_traces) if fast_traces else None,
                )
