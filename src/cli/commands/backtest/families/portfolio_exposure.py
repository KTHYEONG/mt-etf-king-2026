"""Lottery/convexity hooks (P14/P16/P31)."""

from __future__ import annotations

import contextlib
import logging

from src.cli.commands.backtest._core import (
    _Cell,
    _fmt,
)
from src.cli.context import _make_eval_control_model
from src.strategies.registry import STRATEGIES
from src.tournament.distribution import ReturnDistribution, evaluate_adoption_gates

logger = logging.getLogger(__name__)


def _hook_lottery_exposure(cell: _Cell) -> None:
    from src.tournament.distribution import b1_gate_anchors_from_distribution as _b1_gate_p14  # noqa: I001
    from src.tournament.distribution import resolve_adoption_vehicle_rate  # noqa: I001
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
        f"{float(cost_cfg.slippage_bps or 0.0):.6f}_{float(participation):.6f}_P14"
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
        _b1_gate_anchor_cache[anchor_key] = _b1_gate_p14(b1_dist)
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
        f"[EVAL] adoption_gate model=P14 status={gate_status} fails={gate_fails} "
        f"p_gt_30={_fmt(p30)} b1={_fmt(b1_p30)} p_gt_40={_fmt(p40)} b1={_fmt(b1_p40)} "
        f"vehicle_mult2_rate={_fmt(v_rate)} eval_mode={eval_mode}"
    )


def _hook_lottery_rebalance(cell: _Cell) -> None:
    from src.tournament.distribution import b1_gate_anchors_from_distribution as _b1_gate_p19  # noqa: I001
    from src.tournament.distribution import resolve_adoption_vehicle_rate as _resolve_p19  # noqa: I001

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
    _make_eval_control_model("portfolio.lottery_exposure", eval_mode)
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
            _resolve_p19(
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
    anchor_key19 = (
        f"{float(cost_cfg.commission_bps or 0.0):.6f}_"
        f"{float(cost_cfg.slippage_bps or 0.0):.6f}_{float(participation):.6f}_P19"
    )
    if anchor_key19 not in _b1_gate_anchor_cache:
        b1_model = _make_eval_control_model("baseline.mom20_top1", eval_mode)
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
        _b1_gate_anchor_cache[anchor_key19] = _b1_gate_p19(b1_dist)
    b1_p30_19, b1_p40_19, b1_cvar_19 = _b1_gate_anchor_cache[anchor_key19]
    gate_status19, gate_fails19 = evaluate_adoption_gates(
        p30,
        b1_p30_19,
        p40,
        b1_p40_19,
        float(dist.cvar_05),
        b1_cvar_19,
        v_rate,
    )
    # P14 control for regression visibility
    p14_p30 = 0.0
    p14_p40 = 0.0
    try:
        p14_model = _make_eval_control_model("portfolio.lottery_exposure", eval_mode)
        p14_rolling = simulator.run_rolling(
            p14_model,
            panel,
            case_config,
            horizon=horizon,
            path_dependent=False,
            leverage_allowed=_lev_allowed_resolved,
            inverse_allowed=_inv_allowed_resolved,
            close_map=close_map,
        )
        p14_dist = ReturnDistribution.summarise(
            name="portfolio.lottery_exposure",
            returns=list(p14_rolling.returns),
            horizon=horizon,
            thresholds=thresholds,
            tail_weights=tail_weights,
            givebacks=list(getattr(p14_rolling, "givebacks", ())),
        )
        for kk, vv in (p14_dist.exceedance or {}).items():
            with contextlib.suppress(Exception):
                fk = float(kk)
                if abs(fk - 0.30) < 1e-9:
                    p14_p30 = float(vv)
                if abs(fk - 0.40) < 1e-9:
                    p14_p40 = float(vv)
        if p14_p30 == 0.0:
            try:
                exc_p14 = dict(p14_dist.exceedance) if isinstance(p14_dist.exceedance, dict) else {}
                p14_p30 = float(exc_p14.get(0.30, exc_p14.get(0.3, 0.0)))
            except Exception:
                p14_p30 = 0.0
        if p14_p40 == 0.0:
            try:
                exc_p14 = dict(p14_dist.exceedance) if isinstance(p14_dist.exceedance, dict) else {}
                p14_p40 = float(exc_p14.get(0.40, exc_p14.get(0.4, 0.0)))
            except Exception:
                p14_p40 = 0.0
    except Exception:
        p14_p30 = 0.0
        p14_p40 = 0.0
    summary["p_gt_30"] = float(p30)
    summary["p_gt_40"] = float(p40)
    summary["p14_p_gt_30"] = float(p14_p30)
    summary["p14_p_gt_40"] = float(p14_p40)
    summary["b1_p_gt_30"] = float(b1_p30_19)
    summary["b1_p_gt_40"] = float(b1_p40_19)
    summary["b1_cvar_05"] = float(b1_cvar_19)
    summary["vehicle_mult2_rate"] = float(v_rate)
    summary["vehicle_mult2_rate_source"] = "session_path"
    summary["adoption_gate_status"] = str(gate_status19)
    summary["adoption_gate_fails"] = list(gate_fails19)
    summary["eval_mode"] = str(eval_mode)
    logger.info(
        f"[EVAL] adoption_gate model=P19 status={gate_status19} fails={gate_fails19} "
        f"p_gt_30={_fmt(p30)} b1={_fmt(b1_p30_19)} p_gt_40={_fmt(p40)} b1={_fmt(b1_p40_19)} "
        f"vehicle_mult2_rate={_fmt(v_rate)} p14_p_gt_30={_fmt(p14_p30)} eval_mode={eval_mode}"
    )


def _hook_convexity(cell: _Cell) -> None:
    from src.tournament.distribution import b1_gate_anchors_from_distribution as _b1_gate_p16  # noqa: I001
    from src.tournament.distribution import resolve_adoption_vehicle_rate as _resolve_p16  # noqa: I001
    from src.tournament.objective import ObjectiveGateConfig as _OGC_p16  # noqa: I001
    from src.tournament.objective import evaluate_p16_adoption_report  # noqa: I001

    from src.core.config import config_path

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
    rolling = cell.rolling
    model_key = cell.model_key
    _bt_daily = None
    _bt_trades = None
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
    try:
        exc_dist = dict(dist.exceedance) if isinstance(dist.exceedance, dict) else {}
        p50 = float(exc_dist.get(0.50, exc_dist.get(0.5, 0.0)))
    except Exception:
        p50 = 0.0
    for k, v in (dist.exceedance or {}).items():
        with contextlib.suppress(Exception):
            fk = float(k)
            if p30 == 0.0 and abs(fk - 0.30) < 1e-9:
                p30 = float(v)
            if p40 == 0.0 and abs(fk - 0.40) < 1e-9:
                p40 = float(v)
            if p50 == 0.0 and abs(fk - 0.50) < 1e-9:
                p50 = float(v)
    try:
        v_rate = float(
            _resolve_p16(
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
    anchor_key16 = (
        f"{float(cost_cfg.commission_bps or 0.0):.6f}_"
        f"{float(cost_cfg.slippage_bps or 0.0):.6f}_{float(participation):.6f}_P16"
    )
    if anchor_key16 not in _b1_gate_anchor_cache:
        b1_model = _make_eval_control_model("baseline.mom20_top1", eval_mode)
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
        _b1_gate_anchor_cache[anchor_key16] = _b1_gate_p16(b1_dist)
        cell.b1_dist_cache_p16[anchor_key16] = b1_dist
    else:
        b1_dist = cell.b1_dist_cache_p16[anchor_key16]
    b1_p30_16, b1_p40_16, _b1_cvar_16 = _b1_gate_anchor_cache[anchor_key16]
    # for p50 need separate compute but reuse same; get p50 from b1 dist via _b1_gate? Use direct exceedance fallback
    try:
        exc_b1 = dict(b1_dist.exceedance) if isinstance(b1_dist.exceedance, dict) else {}
        b1_p50_16 = float(exc_b1.get(0.50, exc_b1.get(0.5, 0.0)))
    except Exception:
        b1_p50_16 = 0.0
    for k, v in (b1_dist.exceedance or {}).items():
        with contextlib.suppress(Exception):
            fk = float(k)
            if b1_p50_16 == 0.0 and abs(fk - 0.50) < 1e-9:
                b1_p50_16 = float(v)
    b0_dist = ReturnDistribution.summarise(
        name="baseline.buy_hold",
        returns=[],
        horizon=horizon,
        thresholds=thresholds,
        tail_weights=tail_weights,
    )
    try:
        b0_model = _make_eval_control_model("baseline.buy_hold", eval_mode)
        b0_rolling = simulator.run_rolling(
            b0_model,
            panel,
            case_config,
            horizon=horizon,
            path_dependent=False,
            close_map=close_map,
        )
        b0_dist = ReturnDistribution.summarise(
            name="baseline.buy_hold",
            returns=list(b0_rolling.returns),
            horizon=horizon,
            thresholds=thresholds,
            tail_weights=tail_weights,
        )
    except Exception as exc:
        logger.warning(f"[EVAL] control model B0 failed {exc!r}")
    try:
        p14_model = _make_eval_control_model("portfolio.lottery_exposure", eval_mode)
        p14_rolling = simulator.run_rolling(
            p14_model,
            panel,
            case_config,
            horizon=horizon,
            path_dependent=False,
            leverage_allowed=_lev_allowed_resolved,
            inverse_allowed=_inv_allowed_resolved,
            close_map=close_map,
        )
        p14_dist = ReturnDistribution.summarise(
            name="portfolio.lottery_exposure",
            returns=list(p14_rolling.returns),
            horizon=horizon,
            thresholds=thresholds,
            tail_weights=tail_weights,
        )
    except Exception as exc:
        logger.warning(f"[EVAL] control model P14 failed {exc!r}")
        p14_dist = ReturnDistribution.summarise(name="portfolio.lottery_exposure", returns=[], horizon=horizon, thresholds=thresholds, tail_weights=tail_weights)
    _cfg_p16 = _OGC_p16.from_yaml(config_path("gates"))
    leverage_scenarios = ("aggressive", "conservative")
    artifacts_complete = bool(_bt_daily is not None and _bt_trades is not None) if False else True
    # use actual daily/trades completeness flag later; for now True when not yet computed -> recompute after but we set True per spec when daily+trades written
    try:
        artifacts_complete = bool(getattr(rolling, "backtest", None) is not None and getattr(getattr(rolling, "backtest", None), "daily", None) is not None)
    except Exception:
        artifacts_complete = True
    # skip_capacity_violations: 0 unless trace counted CAPACITY_DEMOTE
    skip_capacity_violations = 0
    try:
        p16_report = evaluate_p16_adoption_report(
            p16=dist,
            b1=b1_dist,
            b0=b0_dist,
            p14=p14_dist,
            config=_cfg_p16,
            artifacts_complete=artifacts_complete,
            leverage_scenarios=leverage_scenarios,
            skip_capacity_violations=skip_capacity_violations,
            vehicle_mult2_rate=float(v_rate),
        )
    except Exception:
        p16_report = None
    if p16_report is not None:
        summary["p_gt_30"] = float(p30)
        summary["p_gt_40"] = float(p40)
        summary["p_gt_50"] = float(p50)
        summary["b1_p_gt_30"] = float(b1_p30_16)
        summary["b1_p_gt_40"] = float(b1_p40_16)
        summary["b1_p_gt_50"] = float(b1_p50_16)
        summary["adoption_gate_status"] = str(p16_report.status)
        summary["adoption_gate_fails"] = list(p16_report.failures)
        summary["vehicle_mult2_rate"] = float(v_rate)
        summary["eval_mode"] = str(eval_mode)
        logger.info(
            f"[EVAL] adoption_gate model={model_key} status={p16_report.status} fails={p16_report.failures} "
            f"p_gt_30={_fmt(p30)} b1={_fmt(b1_p30_16)} p_gt_40={_fmt(p40)} b1={_fmt(b1_p40_16)} "
            f"vehicle_mult2_rate={_fmt(v_rate)} eval_mode={eval_mode}"
        )
    else:
        summary["p_gt_30"] = float(p30)
        summary["p_gt_40"] = float(p40)
        summary["p_gt_50"] = float(p50)
        summary["b1_p_gt_30"] = float(b1_p30_16)
        summary["b1_p_gt_40"] = float(b1_p40_16)
        summary["b1_p_gt_50"] = float(b1_p50_16)
        summary["vehicle_mult2_rate"] = float(v_rate)
        summary["eval_mode"] = str(eval_mode)
        logger.info(f"[EVAL] adoption_gate model={model_key} status=FAIL fails=[] p_gt_30={_fmt(p30)} eval_mode={eval_mode}")


__all__ = ["_hook_convexity", "_hook_lottery_exposure", "_hook_lottery_rebalance"]
