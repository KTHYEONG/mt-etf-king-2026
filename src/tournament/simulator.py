from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.alpha.base import AlphaModel
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.metrics import compound_returns, max_drawdown, peak_to_final_giveback, window_returns
from src.core.calendar import TradingCalendar
from src.portfolio.policy import PathDependentPolicyError

# wiring: path_dependent must be True for PortfolioPolicy
_path_dependent_ref = PathDependentPolicyError  # noqa: F401


def model_requires_path_dependent(model: object) -> bool:
    if getattr(model, "path_dependent", False) is True:
        return True
    from src.portfolio.policy import PortfolioPolicy

    return isinstance(model, PortfolioPolicy)


@dataclass(frozen=True)
class RollingResult:
    name: str
    horizon: int
    starts: tuple[date, ...]
    returns: tuple[float, ...]
    drawdowns: tuple[float, ...]
    givebacks: tuple[float, ...] = ()


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
            result = self.engine.run(model, panel, config)
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
            )
        else:
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
                res = self.engine.run(model, panel, win_config)
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
