# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


def build_close_map(panel: pl.DataFrame) -> dict[date, dict[str, float]]:
    out: dict[date, dict[str, float]] = {}
    if panel.height == 0 or not {"date", "ticker", "close"} <= set(panel.columns):
        return out
    for row in panel.iter_rows(named=True):
        d = row.get("date")
        t = row.get("ticker")
        c = row.get("close")
        if d is None or t is None or c is None:
            continue
        try:
            cf = float(c)
        except Exception:
            continue
        t_str = str(t)
        out.setdefault(d, {})[t_str] = cf
    return out


def _build_open_map(panel: pl.DataFrame) -> dict[date, dict[str, float]]:
    out: dict[date, dict[str, float]] = {}
    if panel.height == 0 or not {"date", "ticker", "open"} <= set(panel.columns):
        return out
    for row in panel.iter_rows(named=True):
        d = row.get("date")
        t = row.get("ticker")
        o = row.get("open")
        if d is None or t is None or o is None:
            continue
        try:
            of = float(o)
        except Exception:
            continue
        t_str = str(t)
        out.setdefault(d, {})[t_str] = of
    return out


@dataclass(frozen=True)
class SessionInputs:
    dates: tuple[date, ...]
    close_map: dict[date, dict[str, float]]
    scores: dict[date, dict[str, float]]
    universes: dict[date, object]
    snapshots: dict[date, pl.DataFrame]
    open_map: dict[date, dict[str, float]] | None = None
    adv_map: dict[date, dict[str, float]] | None = None
    panel: pl.DataFrame | None = None
    regimes: dict[date, object] | None = None
    rules: object | None = None


def build_session_cache(engine, model, panel: pl.DataFrame, config) -> SessionInputs:
    close_map = build_close_map(panel)
    open_map = _build_open_map(panel)
    try:
        sessions = engine.calendar.sessions(config.start, config.end)
    except Exception:
        sessions = []
    dates_t = tuple(sessions)
    universes: dict[date, object] = {}
    snapshots: dict[date, pl.DataFrame] = {}
    scores: dict[date, dict[str, float]] = {}
    # prebuild rules once
    rules = None
    try:
        from pathlib import Path as _Path

        from src.universe.tournament import TournamentRules

        try:
            rules = TournamentRules.from_yaml(_Path("configs/tournament.yaml"))
        except Exception:
            comm_val = config.costs.commission_bps if config.costs.commission_bps is not None else 0.0
            slip_val = config.costs.slippage_bps if config.costs.slippage_bps is not None else 0.0
            filt = config.filters
            rules = TournamentRules(
                name="default",
                start_date=config.start,
                end_date=config.end,
                initial_capital=int(config.capital),
                category="autonomous",
                leverage_allowed=True,
                inverse_allowed=True,
                max_weight=1.0,
                cash_allowed=True,
                sponsor_etf_only=False,
                manifest_path=None,
                issuer_whitelist=None,
                commission_bps=float(comm_val),
                slippage_bps=float(slip_val),
                max_order_to_adv=filt.max_order_to_adv,
                stress_grid=(0.01, 0.02, 0.05, 0.10),
            )
    except Exception:
        rules = None

    use_scores = bool(getattr(model, "scores_path_independent", True))

    # build adv_map per session for later cap (if universe present)
    adv_map: dict[date, dict[str, float]] = {}

    for d in sessions:
        # universe
        try:
            snap_uni = engine.universe.get(d, config.filters)
        except Exception:
            snap_uni = None
        if snap_uni is not None:
            universes[d] = snap_uni
        # snapshot
        try:
            snap = engine.features.snapshot(panel, snap_uni) if snap_uni is not None else panel.filter(pl.col("date") == d) if "date" in panel.columns else panel
        except Exception:
            try:
                snap = panel.filter(pl.col("date") == d) if "date" in panel.columns else panel
            except Exception:
                snap = panel
        snapshots[d] = snap
        # scores
        if use_scores:
            try:
                from src.alpha.base import DecisionContext

                regime_snap = None
                if getattr(engine, "regimes", None) is not None:
                    try:
                        regime_snap = engine.regimes.get(d)  # type: ignore[attr-defined]
                    except Exception:
                        regime_snap = None
                ctx = DecisionContext(
                    decision_date=d,
                    regime=regime_snap,
                    capital=float(config.capital),
                    held={},
                    rules=rules,  # type: ignore[arg-type]
                )
                sc = model.score(snap, ctx)  # type: ignore[call-arg]
            except Exception:
                sc = {}
            if sc is None:
                sc = {}
            try:
                scores[d] = {str(k): float(v) for k, v in dict(sc).items()}
            except Exception:
                scores[d] = {}
        else:
            scores[d] = {}

        # adv map keyed by execution_date (next session), matching BacktestEngine.run
        try:
            execution_date = None
            try:
                execution_date = engine.calendar.next_session(d)
            except Exception:
                execution_date = None
            if execution_date is not None:
                mp: dict[str, float] = {}
                if "ticker" in panel.columns and "date" in panel.columns:
                    try:
                        sub = panel.filter(pl.col("date") == d).select(pl.col("ticker").unique())
                        tickers = [str(x) for x in sub.to_series().to_list()] if sub.height > 0 else []
                    except Exception:
                        tickers = []
                    for tk in tickers:
                        try:
                            av = engine.universe.adv(str(tk), execution_date)
                        except Exception:
                            av = None
                        if av is not None:
                            try:
                                mp[str(tk)] = float(av)
                            except Exception:
                                pass
                if mp:
                    adv_map[execution_date] = mp
        except Exception:
            pass

    return SessionInputs(
        dates=dates_t,
        close_map=close_map,
        scores=scores,
        universes=universes,
        snapshots=snapshots,
        open_map=open_map,
        adv_map=adv_map,
        panel=panel,
        regimes=getattr(engine, "regimes", None) if hasattr(engine, "regimes") else None,  # type: ignore[attr-defined]
        rules=rules,
    )
