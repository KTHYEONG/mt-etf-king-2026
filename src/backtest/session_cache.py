# mypy: ignore-errors
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.backtest.session_grid import resolve_session_grid  # noqa: F401


def build_close_map(panel: pl.DataFrame) -> dict[date, dict[str, float]]:
    out: dict[date, dict[str, float]] = {}
    if panel.height == 0 or not {"date", "ticker", "close"} <= set(panel.columns):
        return out
    dates = panel["date"].to_list()
    tickers = panel["ticker"].to_list()
    closes = panel["close"].to_list()
    for d, t, c in zip(dates, tickers, closes, strict=False):
        if d is None or t is None or c is None:
            continue
        try:
            cf = float(c)
        except Exception:
            continue
        import math

        if not math.isfinite(cf):
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
    model_name: str | None = None


def _is_fillable_sticky_model(model: object) -> bool:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    if bool(getattr(cfg, "exclude_synthetic", False)):
        return True
    try:
        mfr = float(getattr(cfg, "min_fill_ratio", 0.0) or 0.0)
    except Exception:
        return False
    return math.isfinite(mfr) and mfr > 0


def _tradable_day_tickers(panel: pl.DataFrame, decision_date: date) -> tuple[str, ...]:
    if panel.height == 0:
        return ()
    snap = panel.filter(pl.col("date") == decision_date) if "date" in panel.columns else panel
    if "is_tradable" in snap.columns:
        snap = snap.filter(pl.col("is_tradable"))
    if "ticker" not in snap.columns or snap.height == 0:
        return ()
    return tuple(sorted({str(t) for t in snap.get_column("ticker").to_list()}))


def _build_fillable_score_snapshot(
    engine: object,
    panel: pl.DataFrame,
    decision_date: date,
    tickers: tuple[str, ...],
    filters: object,
) -> pl.DataFrame:
    from src.universe.provider import UniverseMode, UniverseSnapshot

    mode = getattr(filters, "mode", UniverseMode.DEPLOYMENT)
    uni = UniverseSnapshot(
        as_of=decision_date,
        mode=mode,
        tickers=tickers,
        dropped={},
        filters=filters,  # type: ignore[arg-type]
    )
    try:
        return engine.features.snapshot(panel, uni)  # type: ignore[union-attr]
    except Exception:
        if "date" in panel.columns:
            return panel.filter(pl.col("date") == decision_date).filter(pl.col("ticker").is_in(list(tickers)))
        return panel.filter(pl.col("ticker").is_in(list(tickers)))


def build_session_cache(engine, model, panel: pl.DataFrame, config, *, leverage_allowed: bool | None = None, inverse_allowed: bool | None = None) -> SessionInputs:
    close_map = build_close_map(panel)
    open_map = _build_open_map(panel)
    try:
        sessions = list(resolve_session_grid(engine.calendar.sessions(config.start, config.end), panel).sessions)
    except Exception:
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

    # INV-12-3: leverage_allowed/inverse_allowed resolved from CLI must be stored on rules
    if rules is not None and (leverage_allowed is not None or inverse_allowed is not None):
        try:
            from dataclasses import replace as _replace

            kw: dict[str, object] = {}
            if leverage_allowed is not None:
                kw["leverage_allowed"] = leverage_allowed
            if inverse_allowed is not None:
                kw["inverse_allowed"] = inverse_allowed
            rules = _replace(rules, **kw)  # type: ignore[arg-type]
        except Exception:
            try:
                if leverage_allowed is not None:
                    object.__setattr__(rules, "leverage_allowed", leverage_allowed)  # type: ignore[attr-defined]
                if inverse_allowed is not None:
                    object.__setattr__(rules, "inverse_allowed", inverse_allowed)  # type: ignore[attr-defined]
            except Exception:
                pass

    use_scores = bool(getattr(model, "scores_path_independent", True))
    fillable_score_panel = _is_fillable_sticky_model(model)

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
        # snapshot: fillable sticky scores across tradable panel; execution universe unchanged
        try:
            if fillable_score_panel:
                tradable = _tradable_day_tickers(panel, d)
                snap = _build_fillable_score_snapshot(engine, panel, d, tradable, config.filters)
            elif snap_uni is not None:
                snap = engine.features.snapshot(panel, snap_uni)
            else:
                snap = panel.filter(pl.col("date") == d) if "date" in panel.columns else panel
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

        # adv_map per session - PIT uses decision_date ADV (INV-ADV-PIT)
        try:
            from src.backtest.engine import build_execution_adv

            _ = build_execution_adv
            # wiring: engine.universe.adv(str(tk), d)
            mp: dict[str, float] = {}
            if "ticker" in panel.columns and "date" in panel.columns:
                try:
                    sub = panel.filter(pl.col("date") == d).select(pl.col("ticker").unique())
                    tickers = [str(x) for x in sub.to_series().to_list()] if sub.height > 0 else []
                except Exception:
                    tickers = []
                # use build_execution_adv for decision_date PIT
                try:
                    adv_pit = build_execution_adv(engine, tickers, d)
                    for tk in tickers:
                        if str(tk) in adv_pit:
                            mp[str(tk)] = float(adv_pit[str(tk)])
                        else:
                            try:
                                av = engine.universe.adv(str(tk), d)
                                if av is not None:
                                    mp[str(tk)] = float(av)
                            except Exception:
                                pass
                except Exception:
                    for tk in tickers:
                        try:
                            av = engine.universe.adv(str(tk), d)
                            if av is not None:
                                mp[str(tk)] = float(av)
                        except Exception:
                            pass
            if mp:
                adv_map[d] = mp
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
        model_name=str(getattr(model, "name", "")) or None,
    )
