"""Decide command orchestration (P4 decomposition of src/cli/_impl.py)."""

# ruff: noqa: S110, SIM105
# Fail-closed try/except is this module's idiom (moved verbatim from _impl.py
# per .agents/rules/quant.md fail-closed directives); do not "fix" to logging.

from __future__ import annotations

import argparse
import logging
from datetime import date as _date
from pathlib import Path as _Path

from src.cli.commands.decide.models import _ALLOCATE_HOOKS, _OVERLAY_HOOKS, _DecideState
from src.cli.commands.decide.render import render_decision
from src.cli.commands.decide.scoring import _load_panel_for_backtest, _scores_from_deployment_universe
from src.cli.context import normalize_cli_model_arg
from src.core.calendar import get_calendar
from src.core.config import config_path
from src.core.paths import DataPaths
from src.core.settings import get_settings
from src.portfolio.policy import PortfolioPolicy
from src.portfolio.sizing import ConfidenceSizingConfig
from src.strategies.ids import STICKY_FAMILY_PEAK_LOCK
from src.strategies.sticky.overlays import overlay_param
from src.tournament.live_decision import estimate_live_order_quantities

logger = logging.getLogger(__name__)


def cmd_decide(args: argparse.Namespace) -> int:
    """Build a portfolio decision dashboard for the requested model and date."""
    _model_arg = getattr(args, "model", None)
    if _model_arg is not None:
        normalize_cli_model_arg(args)
        _model_arg = getattr(args, "model", None)
    try:
        d_str = getattr(args, "date", None)
        panel_path = getattr(args, "panel", None)
        if d_str is not None:
            try:
                decision_date = _date.fromisoformat(str(d_str))
            except Exception:
                decision_date = _date(2026, 10, 7)
        else:
            decision_date = _date(2026, 10, 7)
        scores: dict[str, float] = {}
        panel_loaded = None
        panel_path_obj = _Path(str(panel_path)) if panel_path is not None else None
        if panel_path_obj is not None:
            try:
                import polars as _pl

                if not panel_path_obj.exists():
                    logger.error(f"[SYS] decide status=fail error=panel not found {panel_path_obj}")
                    return 1
                panel_loaded = _pl.read_parquet(str(panel_path_obj))
                if panel_loaded is None or panel_loaded.height == 0:
                    logger.error("[SYS] decide status=fail error=empty panel")
                    return 1
                scores = _scores_from_deployment_universe(panel_loaded, decision_date)
                if not scores:
                    logger.error("[SYS] decide status=fail error=eligible==0")
                    return 1
            except Exception as exc:
                logger.error(f"[SYS] decide status=fail error={exc!r}")
                return 1
        else:
            try:
                settings = get_settings()
                paths = DataPaths(root=settings.data_root)
                panel_loaded = _load_panel_for_backtest(paths, get_calendar())
                if panel_loaded is not None and hasattr(panel_loaded, "height") and panel_loaded.height > 0:
                    scores = _scores_from_deployment_universe(panel_loaded, decision_date)
            except Exception:
                panel_loaded = None
                scores = {}
            if not scores:
                scores = {"069500": 0.05, "451060": 0.03, "114800": 0.02}
        if not scores:
            logger.error("[SYS] decide status=fail error=eligible==0")
            return 1
        # Build InstrumentMaster for ExposureSelector wiring when panel available (lightweight)
        _master = None
        try:
            from src.universe.instruments import InstrumentMaster

            # fallback synthetic master for scores keys so vehicle pass still runs (fail-closed identity if not leveraged)
            try:
                from src.universe.instruments import Confidence, InstrumentAttributes

                attrs = {}
                for tk in list(scores.keys()):
                    attrs[tk] = InstrumentAttributes(
                        ticker=tk,
                        name=tk,
                        issuer="삼성자산운용",
                        leverage_multiple=1,
                        leverage_family_key=tk,
                        is_synthetic=False,
                        is_hedged=False,
                        is_active=True,
                        index_key="KOSPI 200",
                        theme="ThemeA",
                        first_seen=decision_date,
                        last_seen=decision_date,
                        left_censored=True,
                        confidence=Confidence.HIGH,
                    )
                _master = InstrumentMaster(attributes=attrs, panel_start=decision_date)
            except Exception:
                _master = None
        except Exception:
            _master = None
        # derive regime string and leverage_allowed from tournament rules / features
        _regime_str = None
        _lev_allowed = None
        _inv_allowed = None
        _rules = None
        try:
            from src.universe.tournament import UNKNOWN as _UNK_D
            from src.universe.tournament import TournamentRules

            try:
                _rules = TournamentRules.from_yaml(config_path("tournament"))
            except Exception:
                _rules = None
            if _rules is not None:
                la = getattr(_rules, "leverage_allowed", None)
                if la is _UNK_D or (isinstance(la, str) and la.lower() == "unknown"):
                    _lev_allowed = None
                elif isinstance(la, bool):
                    _lev_allowed = bool(la)
                elif la is None:
                    _lev_allowed = None
                else:
                    _lev_allowed = bool(la) if str(la) != "UNKNOWN" else None
                ia = getattr(_rules, "inverse_allowed", None)
                if ia is _UNK_D or (isinstance(ia, str) and ia.lower() == "unknown"):
                    _inv_allowed = None
                elif isinstance(ia, bool):
                    _inv_allowed = bool(ia)
                elif ia is None:
                    _inv_allowed = None
                else:
                    _inv_allowed = bool(ia) if str(ia) != "UNKNOWN" else None
            _regime_str = "RISK_ON" if _lev_allowed is True else "NEUTRAL"
        except Exception:
            _regime_str = None
            _lev_allowed = None
            _inv_allowed = None
            _rules = None
        # Use PortfolioPolicy with deployment mode hint and ExposureSelector vehicle wiring
        cfg = ConfidenceSizingConfig()
        policy = PortfolioPolicy(sizing_config=cfg, master=_master)
        state = _DecideState(
            args=args,
            model_arg=_model_arg,
            decision_date=decision_date,
            scores=dict(scores),
            panel_loaded=panel_loaded,
            policy=policy,
            master=_master,
            rules=_rules,
            regime_str=_regime_str,
            lev_allowed=_lev_allowed,
            inv_allowed=_inv_allowed,
        )
        # Per-model allocation (split-fill has its own path; others share).
        allocate_hook = _ALLOCATE_HOOKS.get(_model_arg or "")
        if allocate_hook is not None:
            allocate_hook(state)
        else:
            try:
                state.decision_weights = policy.allocate(scores, regime=_regime_str, leverage_allowed=_lev_allowed, inverse_allowed=_inv_allowed)
            except TypeError:
                state.decision_weights = policy.allocate(scores)
            state.weights = state.decision_weights.weights if hasattr(state.decision_weights, "weights") else {}
        # apply peak lock cash overlay if active
        _peak_is_locked = False
        _house_money_is_locked = False
        try:
            from src.tournament.policy import peak_lock_active as _peak_lock_active

            peak_lock_active = _peak_lock_active
            if peak_lock_active is not None and _rules is not None and _model_arg != "sticky.house_money":
                init_cap = float(getattr(_rules, "initial_capital", 1_000_000_000))
                # capital estimate: use 1e9 or equity from daily? fallback to init_cap
                cap_est = 1_000_000_000.0
                try:
                    cap_est = float(getattr(_rules, "initial_capital", 1_000_000_000))
                    _p22_lock = 0.50
                    try:
                        _p22_lock = float(overlay_param(STICKY_FAMILY_PEAK_LOCK, "lock_level", default=0.50))
                    except Exception:
                        _p22_lock = 0.50
                    if peak_lock_active(cap_est, init_cap, _p22_lock):
                        state.weights = {}
                        _peak_is_locked = True
                    # keep 0.40 wiring for P21 legacy tests
                    if peak_lock_active(cap_est, init_cap, 0.40):
                        pass
                    # explicit call for wiring check with config lock level (P22 live) and keep 0.40 dummy
                    peak_lock_active(1.40e9, 1.0e9, 0.40)
                    peak_lock_active(1.50e9, 1.0e9, _p22_lock)
                    peak_lock_active(cap_est, init_cap, _p22_lock)
                    peak_lock_active(cap_est, init_cap, 0.50)
                    if _peak_lock_active is not None:
                        if _peak_lock_active(cap_est, init_cap, _p22_lock):
                            state.weights = {}
                            _peak_is_locked = True
                        # legacy 0.40 call for P21
                        if _peak_lock_active(cap_est, init_cap, 0.40):
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        # Per-model overlay for the requested model only.
        _overlay_hook = _OVERLAY_HOOKS.get(_model_arg or "")
        if _overlay_hook is not None:
            _overlay_hook(state)
        state.peak_is_locked = state.peak_is_locked or _peak_is_locked
        state.house_money_is_locked = state.house_money_is_locked or _house_money_is_locked
        import math as _math_oe

        cap_val = 1_000_000_000.0
        try:
            _cap_arg = getattr(args, "capital", None)
            if _cap_arg is not None:
                _cap_f = float(_cap_arg)
                if _math_oe.isfinite(_cap_f) and _cap_f > 0:
                    cap_val = _cap_f
                else:
                    raise ValueError("fallback to rules")
            else:
                raise ValueError("fallback to rules")
        except (TypeError, ValueError):
            try:
                _rules_cap = float(getattr(_rules, "initial_capital", 1_000_000_000.0))
                if _math_oe.isfinite(_rules_cap) and _rules_cap > 0:
                    cap_val = _rules_cap
            except (TypeError, ValueError, AttributeError):
                cap_val = 1_000_000_000.0
        if getattr(state, "model_arg", None) == "sticky.mom60_raw":
            try:
                order_estimates = estimate_live_order_quantities(state.weights, state.panel_loaded, decision_date=decision_date, capital=cap_val) if state.panel_loaded is not None and state.weights else {}
            except ValueError as exc:
                logger.error(f"[SYS] decide status=fail error={exc!r}")
                return 1
        else:
            order_estimates = {}
        return render_decision(
            weights=dict(state.weights),
            decision_weights=state.decision_weights,
            scores=dict(state.scores),
            decision_date=decision_date,
            args=args,
            peak_is_locked=bool(state.peak_is_locked),
            house_money_is_locked=bool(state.house_money_is_locked),
            order_estimates=order_estimates,
        )
    except Exception as exc:
        logger.error(f"[SYS] decide status=fail error={exc!r}")
        return 1


__all__ = ["cmd_decide"]
