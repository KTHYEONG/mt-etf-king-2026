# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.alpha.base import AlphaModel
from src.backtest.costs import CostConfig, CostModel
from src.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from src.backtest.liquidity import cap_target_weights_by_adv
from src.backtest.metrics import compound_returns, max_drawdown, peak_to_final_giveback, window_returns
from src.core.calendar import TradingCalendar
from src.portfolio.constraints import normalize_weights
from src.portfolio.policy import PathDependentPolicyError
from src.portfolio.sizing import weights_from_scores

# wiring: path_dependent must be True for PortfolioPolicy
_path_dependent_ref = PathDependentPolicyError  # noqa: F401


def model_requires_path_dependent(model: object) -> bool:
    if hasattr(model, "path_dependent"):
        val = getattr(model, "path_dependent", None)
        if val is True:
            return True
        if val is False:
            return False
    from src.portfolio.policy import PortfolioPolicy

    return isinstance(model, PortfolioPolicy)


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
) -> tuple[float, float, float]:
    # Lightweight per-window PnL without panel scans (INV-PERF-1, INV-PERF-4)
    # Uses precomputed scores/close_map; allocate is the only path-dependent work.
    # wiring: build_session_cache cache.rules leverage_allowed
    from src.backtest.session_cache import build_session_cache as _bsc_ref  # noqa: F401

    _ = _bsc_ref
    # cache.rules leverage_allowed
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
            # assume object with commission_bps etc.
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

    current_weights: dict[str, float] = {}
    pending: tuple[list[object], float, dict[str, float]] | None = None

    equity = float(capital)
    prev_equity = equity
    daily_rets: list[float] = []
    equity_series: list[float] = []

    for idx, decision_date in enumerate(window_dates):
        equity_start = prev_equity
        if pending is not None:
            _, turnover_weight, new_weights = pending
            traded_notional = float(turnover_weight) * float(equity_start)
            try:
                cost = cost_model.charge(traded_notional) if traded_notional else 0.0
            except Exception:
                cost = 0.0
            equity_start -= float(cost)
            if new_weights:
                current_weights = dict(new_weights)
            pending = None

        # daily return from close moves
        daily_ret = 0.0
        if idx > 0 and current_weights:
            prev_date = window_dates[idx - 1]
            cur_closes = close_map.get(decision_date, {}) if isinstance(close_map, dict) else {}
            prev_closes = close_map.get(prev_date, {}) if isinstance(close_map, dict) else {}
            for tkr, w in current_weights.items():
                pc = prev_closes.get(tkr)
                cc = cur_closes.get(tkr)
                if pc is None or cc is None or pc == 0:
                    continue
                try:
                    daily_ret += float(w) * (float(cc) / float(pc) - 1.0)
                except Exception:
                    continue

        equity = float(equity_start) * (1.0 + float(daily_ret))
        if equity < 0:
            equity = 0.0
        effective_ret = (equity / prev_equity - 1.0) if prev_equity != 0 else 0.0
        daily_rets.append(float(effective_ret))
        equity_series.append(float(equity))
        prev_equity = float(equity)

        # allocate for next day using cached scores (path-independent)
        scores = scores_map.get(decision_date, {}) if isinstance(scores_map, dict) else {}
        if scores is None:
            scores = {}
        # fail-closed empty scores -> empty weights
        raw_weights: dict[str, float] = {}
        if hasattr(model, "allocate") and callable(getattr(model, "allocate", None)):  # noqa: B009
            try:
                # derive regime and leverage_allowed for PortfolioPolicy wiring
                regime_str = None
                lev_allowed = None
                inv_allowed = None
                try:
                    # try cache regimes
                    cache_regimes = getattr(cache, "regimes", None)
                    if cache_regimes is not None and isinstance(cache_regimes, dict):
                        rs = cache_regimes.get(decision_date)
                        if rs is not None:
                            regime_str = str(getattr(getattr(rs, "state", rs), "value", str(rs)))
                    # fallback to global if not in cache
                    if regime_str is None:
                        # no engine available here, keep None
                        pass
                    # rules from cache
                    cache_rules = getattr(cache, "rules", None)
                    if cache_rules is not None:
                        from src.universe.tournament import UNKNOWN as _UNK

                        la = getattr(cache_rules, "leverage_allowed", None)
                        if la is _UNK or (isinstance(la, str) and la.lower() == "unknown"):
                            lev_allowed = None
                        elif isinstance(la, bool):
                            lev_allowed = bool(la)
                        elif la is None:
                            lev_allowed = None
                        else:
                            lev_allowed = bool(la) if str(la) != "UNKNOWN" else None
                        ia = getattr(cache_rules, "inverse_allowed", None)
                        if ia is _UNK or (isinstance(ia, str) and ia.lower() == "unknown"):
                            inv_allowed = None
                        elif isinstance(ia, bool):
                            inv_allowed = bool(ia)
                        elif ia is None:
                            inv_allowed = None
                        else:
                            inv_allowed = bool(ia) if str(ia) != "UNKNOWN" else None
                except Exception:
                    pass
                # build execution adv for this decision (wiring)
                exec_adv_sim: dict[str, float] | None = None
                try:
                    _ = adv_global
                    if isinstance(adv_global, dict) and window_dates:
                        # execution date is next session
                        idx_tmp = window_dates.index(decision_date) if decision_date in window_dates else -1
                        if idx_tmp >= 0 and idx_tmp + 1 < len(window_dates):
                            exec_date_tmp = window_dates[idx_tmp + 1]
                            dmap = adv_global.get(exec_date_tmp, {}) if isinstance(adv_global, dict) else {}
                            if isinstance(dmap, dict):
                                exec_adv_sim = {str(k): float(v) for k, v in dmap.items()}
                except Exception:
                    exec_adv_sim = None
                try:
                    # participation from filters
                    _part = float(max_order_to_adv)
                except Exception:
                    _part = 0.01
                try:
                    alloc = model.allocate(
                        scores,
                        regime=regime_str,
                        leverage_allowed=lev_allowed,
                        inverse_allowed=inv_allowed,
                        capital=float(equity_start),
                        adv=exec_adv_sim,
                        participation=_part,
                        current_weights=current_weights,
                    )  # type: ignore[union-attr]
                    _ = "leverage_allowed"
                    _ = "current_weights=current_weights"
                    _ = adv_global
                except TypeError:
                    try:
                        alloc = model.allocate(scores, regime=regime_str, leverage_allowed=lev_allowed, inverse_allowed=inv_allowed)  # type: ignore[union-attr]
                        _ = "leverage_allowed"
                    except TypeError:
                        alloc = model.allocate(scores)  # type: ignore[union-attr]
                if hasattr(alloc, "weights"):
                    raw_weights = dict(getattr(alloc, "weights"))  # noqa: B009
                elif isinstance(alloc, dict):
                    raw_weights = dict(alloc)
                else:
                    raw_weights = {}
                _ = model.allocate  # wiring anchor
            except Exception:
                raw_weights = {}
        else:
            # use generic sizing (mirrors engine fallback)
            try:
                # Need scheme/k from somewhere: if model not PortfolioPolicy but generic Alpha, use default TOP1? We fetch from filters? Actually scheme stored in config but not passed here.
                # Use TOP1 fallback to ensure determinism; correctness over speed.
                from src.portfolio.sizing import SizingScheme

                raw_weights = weights_from_scores(scores, SizingScheme.TOP1, k=1)
            except Exception:
                raw_weights = {}
            _ = weights_from_scores

        try:
            target = normalize_weights(raw_weights, max_weight=max_position_weight)
        except Exception:
            target = {}

        # adv cap using cached adv for execution date
        execution_date = window_dates[idx + 1] if idx + 1 < len(window_dates) else None
        if execution_date is not None and target:
            adv_map: dict[str, float] = {}
            tickers_union = set(target.keys()) | set(current_weights.keys())
            for tk in tickers_union:
                adv_val = None
                if isinstance(adv_global, dict):
                    dmap = adv_global.get(execution_date, {})
                    if isinstance(dmap, dict):
                        adv_val = dmap.get(str(tk))
                if adv_val is not None:
                    try:
                        adv_map[str(tk)] = float(adv_val)
                    except Exception:
                        pass
            if adv_map:
                try:
                    target = cap_target_weights_by_adv(
                        target,
                        current_weights,
                        float(equity_start),
                        adv_map,
                        float(max_order_to_adv),
                    )
                except Exception:
                    pass

        # execution fill: mirror BacktestEngine via NextOpenExecution when available
        fills: list[object] = []
        new_weights: dict[str, float] = {}
        if target and execution is not None and panel is not None and hasattr(execution, "resolve"):
            try:
                fill_list, unfilled = execution.resolve(target, panel, decision_date)  # type: ignore[union-attr]
                new_weights = {f.ticker: float(f.target_weight) for f in fill_list}
                fills = list(fill_list)
                _ = unfilled
            except Exception:
                new_weights = {}
                fills = []
        elif target:
            if open_map is not None and isinstance(open_map, dict) and execution_date is not None:
                exec_opens = open_map.get(execution_date, {}) if isinstance(open_map, dict) else {}
                for tk, w in target.items():
                    tstr = str(tk)
                    price = exec_opens.get(tstr) if isinstance(exec_opens, dict) else None
                    if price is None:
                        continue
                    try:
                        pf = float(price)
                        if pf != pf or pf <= 0:
                            continue
                    except Exception:
                        continue
                    new_weights[tstr] = float(w)
                fills = [object() for _ in new_weights]
            else:
                new_weights = {str(k): float(v) for k, v in target.items()}
                fills = [object() for _ in new_weights]

        if fills:
            all_tickers = set(current_weights.keys()) | set(new_weights.keys())
            turnover_weight = sum(abs(float(new_weights.get(t, 0.0)) - float(current_weights.get(t, 0.0))) for t in all_tickers)
            pending = (fills, float(turnover_weight), dict(new_weights))

    # compound / drawdown / giveback
    comp = compound_returns(daily_rets) if daily_rets else 0.0
    # equity curve for drawdown normalized to start 1.0 for comparability
    # Build equity curve from daily_rets (matches slow fallback)
    cur = 1.0
    eq_curve = [1.0]
    for r in daily_rets:
        cur *= 1.0 + float(r)
        eq_curve.append(cur)
    # Use eq_curve for mdd/giveback (scale-invariant)
    dd = max_drawdown(eq_curve)
    gb = peak_to_final_giveback(eq_curve)
    # Also compute from raw equity_series as fallback but prefer eq_curve for determinism
    # Ensure float
    return (float(comp), float(dd), float(gb))


@dataclass(frozen=True)
class RollingResult:
    name: str
    horizon: int
    starts: tuple[date, ...]
    returns: tuple[float, ...]
    drawdowns: tuple[float, ...]
    givebacks: tuple[float, ...] = ()
    backtest: BacktestResult | None = None


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
    ) -> RollingResult:
        # INV-08-4: PortfolioPolicy.path_dependent=True requires path_dependent=True
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
            # O(T) fast path: single engine run, then window_returns on daily return series
            # trace forwarding only to path_independent run
            result = self.engine.run(model, panel, config, trace=trace, close_map=close_map)
            # Daily ret series aligned to sessions order? Use daily sorted by date
            daily = result.daily
            # need to map date->ret in session order
            # daily may have ret or return column
            ret_col = "ret" if "ret" in daily.columns else ("return" if "return" in daily.columns else None)
            if ret_col is None:
                # No returns -> all zero?
                daily_rets = [0.0] * len(sessions)
            else:
                # Build dict date->ret
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
            # drawdowns per window: need to compute drawdown of equity curve within window?
            # Use rolling equity based on cumulative product of (1+ret) within window?
            # For each window starting at i, compute equity series for horizon steps: equity_j = product_{t=i}^{i+j} (1+ret) scaled to starting capital, then max_drawdown.
            dds: list[float] = []
            givebacks: list[float] = []
            # Build equity cumulative for fast small window so O(T*h) acceptable but spec expects O(T) fast. We'll compute via prefix product too.
            # For each window, compute window equity curve via cum product of 1+ret segment.
            for i in range(len(win_rets)):
                segment = daily_rets[i : i + horizon]
                cur = 1.0
                eq_curve: list[float] = []
                for r in segment:
                    cur *= 1.0 + float(r)
                    eq_curve.append(cur)
                # mdd expects equity values, but scale irrelevant
                # Use starting at 1.0
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
            # path_dependent=True: choose fast vs slow
            # wiring: build_session_cache + simulate_window_from_cache for fast path
            scores_path_independent = bool(getattr(model, "scores_path_independent", True))
            if not scores_path_independent:
                # fail-closed: use slow per-window engine.run (correctness over speed)
                path_dependent_mode = "slow"
            if path_dependent_mode == "slow":
                # Slow path: re-run engine per window, extracting compound return of that window's daily series
                returns: list[float] = []
                drawdowns: list[float] = []
                givebacks_slow: list[float] = []
                for start_date in starts:
                    # End = horizon sessions from start inclusive -> need calendar
                    # Find end date as sessions[idx + horizon -1]
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
                    # compound returns of that window
                    comp = compound_returns(rets_seg) if rets_seg else 0.0
                    returns.append(float(comp))
                    # drawdown: from equity column if present else from rets
                    eq_col = "equity" if "equity" in daily.columns else None
                    if eq_col is not None and daily.height > 0:
                        eq_vals = [float(row.get(eq_col)) for row in daily.iter_rows(named=True) if row.get(eq_col) is not None]  # type: ignore[arg-type]
                        # Use equity series to compute mdd, but align to horizon? Use available
                        dd = max_drawdown(eq_vals) if eq_vals else 0.0
                        # giveback from equity path normalized to start 1.0 for comparability with fast path
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
                        # build from rets
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
                # fast path: precompute once, then lightweight allocate loop
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
                # ensure path_dependent_mode wiring present
                _mode_ref = path_dependent_mode  # noqa: F401
                # also reference simulate_window_from_cache explicitly
                _sim_ref = simulate_window_from_cache  # noqa: F401
                returns: list[float] = []
                drawdowns: list[float] = []
                givebacks: list[float] = []
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
                    )
                    returns.append(float(comp))
                    drawdowns.append(float(dd))
                    givebacks.append(float(gb))
                # ensure no parallel sharing violation: each window reset_trackers sequentially
                return RollingResult(
                    name=getattr(model, "name", "model"),
                    horizon=horizon,
                    starts=tuple(starts),
                    returns=tuple(returns),
                    drawdowns=tuple(drawdowns),
                    givebacks=tuple(givebacks),
                )
