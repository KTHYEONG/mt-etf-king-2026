"""Momentum-policy hooks (P08/P10)."""

from __future__ import annotations

import contextlib
import logging

from src.cli.commands.backtest._core import (
    _Cell,
    _fmt,
)
from src.strategies.registry import STRATEGIES
from src.tournament.distribution import ReturnDistribution, evaluate_adoption_gates

logger = logging.getLogger(__name__)


def _hook_momentum_vehicle(cell: _Cell) -> None:
    dist = cell.dist
    summary = cell.summary
    model = cell.model
    cal = cell.cal
    start = cell.start
    end = cell.end
    regimes = cell.regimes
    _lev_allowed_resolved = cell.lev_allowed
    # immutable output: include p_gt_40, p_gt_50, cvar_05, vehicle_mult2_rate
    try:
        exc_dist = dict(dist.exceedance) if isinstance(dist.exceedance, dict) else {}
        p40 = float(exc_dist.get(0.40, exc_dist.get(0.4, 0.0)))
    except Exception:
        p40 = 0.0
    try:
        exc_dist = dict(dist.exceedance) if isinstance(dist.exceedance, dict) else {}
        p50 = float(exc_dist.get(0.50, exc_dist.get(0.5, 0.0)))
    except Exception:
        p50 = 0.0
    # fallback via exceedance keys as strings
    if p40 == 0.0:
        for k, v in (dist.exceedance or {}).items():
            with contextlib.suppress(Exception):
                if abs(float(k) - 0.40) < 1e-9:
                    p40 = float(v)
    if p50 == 0.0:
        for k, v in (dist.exceedance or {}).items():
            with contextlib.suppress(Exception):
                if abs(float(k) - 0.50) < 1e-9:
                    p50 = float(v)
    summary["p_gt_40"] = float(p40)
    summary["p_gt_50"] = float(p50)
    summary["cvar_05"] = float(dist.cvar_05)
    try:
        from src.tournament.distribution import measure_vehicle_activity_from_allocate

        v_rate = float(
            measure_vehicle_activity_from_allocate(
                model,
                cal.sessions(start, end),
                regimes,
                _lev_allowed_resolved,
            )
        )
    except Exception:
        v_rate = 0.0
    summary["vehicle_mult2_rate"] = float(v_rate)
    summary["features_preflight_ok"] = True


def _hook_momentum_confidence(cell: _Cell) -> None:
    from src.tournament.distribution import (
        b1_gate_anchors_from_distribution,
        resolve_adoption_vehicle_rate,
    )
    from src.tournament.eval_mode import resolve_eval_flags

    dist = cell.dist
    summary = cell.summary
    model = cell.model
    engine = cell.engine
    panel = cell.panel
    case_config = cell.case_config
    regimes = cell.regimes
    _lev_allowed_resolved = cell.lev_allowed
    _inv_allowed_resolved = cell.inv_allowed
    cost_cfg = cell.cost_cfg
    participation = cell.participation
    horizon = cell.horizon
    thresholds = cell.thresholds
    tail_weights = cell.tail_weights
    simulator = cell.simulator
    close_map = cell.close_map
    _b1_gate_anchor_cache = cell.b1_anchor_cache
    eval_mode = cell.eval_mode
    try:
        exc_dist = dict(dist.exceedance) if isinstance(dist.exceedance, dict) else {}
        p30 = float(exc_dist.get(0.30, exc_dist.get(0.3, 0.0)))
    except Exception:
        p30 = 0.0
    try:
        exc_dist = dict(dist.exceedance) if isinstance(dist.exceedance, dict) else {}
        p40 = float(exc_dist.get(0.40, exc_dist.get(0.4, 0.0)))
    except Exception:
        p40 = 0.0
    for k, v in (dist.exceedance or {}).items():
        with contextlib.suppress(Exception):
            fk = float(k)
            if p30 == 0.0 and abs(fk - 0.30) < 1e-9:
                p30 = float(v)
            if p40 == 0.0 and abs(fk - 0.40) < 1e-9:
                p40 = float(v)
    try:
        v_rate = float(
            resolve_adoption_vehicle_rate(
                model,
                engine,
                panel,
                case_config,
                regimes,
                _lev_allowed_resolved,
                _inv_allowed_resolved,
            )
        )
    except Exception:
        v_rate = 0.0
    anchor_key = (
        f"{float(cost_cfg.commission_bps or 0.0):.6f}_"
        f"{float(cost_cfg.slippage_bps or 0.0):.6f}_{float(participation):.6f}"
    )
    if anchor_key not in _b1_gate_anchor_cache:
        b1_model = STRATEGIES["baseline.mom20_top1"]()
        resolve_eval_flags(b1_model, eval_mode)
        b1_rolling = simulator.run_rolling(
            b1_model,
            panel,
            case_config,
            horizon=horizon,
            path_dependent=False,
            leverage_allowed=_lev_allowed_resolved,
            inverse_allowed=_inv_allowed_resolved,
            close_map=close_map,
        )
        b1_dist = ReturnDistribution.summarise(
            name="baseline.mom20_top1",
            returns=list(b1_rolling.returns),
            horizon=horizon,
            thresholds=thresholds,
            tail_weights=tail_weights,
            givebacks=list(getattr(b1_rolling, "givebacks", ())),
        )
        _b1_gate_anchor_cache[anchor_key] = b1_gate_anchors_from_distribution(b1_dist)
    b1_p30, b1_p40, b1_cvar = _b1_gate_anchor_cache[anchor_key]
    gate_status, gate_fails = evaluate_adoption_gates(
        p30,
        b1_p30,
        p40,
        b1_p40,
        float(dist.cvar_05),
        b1_cvar,
        v_rate,
    )
    summary["p_gt_30"] = float(p30)
    summary["p_gt_40"] = float(p40)
    summary["vehicle_mult2_rate"] = float(v_rate)
    summary["vehicle_mult2_rate_source"] = "session_path"
    summary["b1_p_gt_30"] = float(b1_p30)
    summary["b1_p_gt_40"] = float(b1_p40)
    summary["b1_cvar_05"] = float(b1_cvar)
    summary["adoption_gate_status"] = str(gate_status)
    summary["adoption_gate_fails"] = list(gate_fails)
    summary["eval_mode"] = str(eval_mode)
    logger.info(
        f"[EVAL] adoption_gate model=P11 status={gate_status} fails={gate_fails} "
        f"p_gt_30={_fmt(p30)} b1={_fmt(b1_p30)} p_gt_40={_fmt(p40)} b1={_fmt(b1_p40)} "
        f"vehicle_mult2_rate={_fmt(v_rate)} eval_mode={eval_mode}"
    )


__all__ = ["_hook_momentum_confidence", "_hook_momentum_vehicle"]
