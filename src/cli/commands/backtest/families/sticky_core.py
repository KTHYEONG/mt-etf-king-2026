"""Sticky core-family forensics hook: impulse_crash (P4)."""

from __future__ import annotations

import contextlib
import logging
from datetime import date
from typing import Final

from src.cli.commands.backtest._core import ForensicsHook as _Hook
from src.cli.commands.backtest._core import _Cell, _fmt
from src.cli.context import _make_eval_control_model
from src.core.config import config_path
from src.tournament.distribution import ReturnDistribution, evaluate_adoption_gates, locked_window_returns

logger = logging.getLogger(__name__)


def _hook_impulse_crash(cell: _Cell) -> None:
    from src.tournament.distribution import b1_gate_anchors_from_distribution as _b1_gate_p21  # noqa: I001
    from src.tournament.distribution import locked_window_returns as _locked_p21  # noqa: I001
    from src.tournament.distribution import resolve_adoption_vehicle_rate as _resolve_p21  # noqa: I001
    from src.tournament.distribution import ruin_probability as _ruin_p21  # noqa: I001
    from src.tournament.objective import ObjectiveGateConfig as _OGC_p21  # noqa: I001
    from src.tournament.objective import evaluate_objective_gates as _eval_obj_p21  # noqa: I001

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
    cal = cell.cal
    start = cell.start
    end = cell.end
    _make_eval_control_model("baseline.buy_hold", eval_mode)
    _make_eval_control_model("baseline.mom20_top1", eval_mode)
    # unlocked p's from dist (for log)
    try:
        exc_dist = dict(dist.exceedance) if isinstance(dist.exceedance, dict) else {}
        p30_unlocked = float(exc_dist.get(0.30, exc_dist.get(0.3, 0.0)))
    except Exception:
        p30_unlocked = 0.0
    try:
        exc_dist = dict(dist.exceedance) if isinstance(dist.exceedance, dict) else {}
        p40_unlocked = float(exc_dist.get(0.40, exc_dist.get(0.4, 0.0)))
    except Exception:
        p40_unlocked = 0.0
    for k, v in (dist.exceedance or {}).items():
        with contextlib.suppress(Exception):
            fk = float(k)
            if p30_unlocked == 0.0 and abs(fk - 0.30) < 1e-9:
                p30_unlocked = float(v)
            if p40_unlocked == 0.0 and abs(fk - 0.40) < 1e-9:
                p40_unlocked = float(v)
    # locked returns from rolling.backtest.daily
    locked_rets: list[float] = []
    try:
        daily_df = getattr(getattr(rolling, "backtest", None), "daily", None)
        if daily_df is not None and hasattr(daily_df, "columns"):
            ret_col = "ret" if "ret" in daily_df.columns else ("return" if "return" in daily_df.columns else None)
            if ret_col is not None:
                sess = cal.sessions(start, end)
                dmap: dict[date, float] = {}
                for row in daily_df.iter_rows(named=True):
                    d = row.get("date")
                    r = row.get(ret_col)
                    if d is None:
                        continue
                    try:
                        dmap[d] = float(r) if r is not None else 0.0
                    except Exception:
                        dmap[d] = 0.0
                locked_daily = [float(dmap.get(d, 0.0)) for d in sess]
                locked_rets = _locked_p21(locked_daily, horizon, 0.40)
            else:
                locked_rets = []
        else:
            locked_rets = []
    except Exception:
        locked_rets = []
    if locked_rets:
        locked_dist = ReturnDistribution.summarise(
            name="P21_locked",
            returns=locked_rets,
            horizon=horizon,
            thresholds=thresholds,
            tail_weights=tail_weights,
        )
    else:
        locked_dist = dist
    try:
        exc_locked = dict(locked_dist.exceedance) if isinstance(locked_dist.exceedance, dict) else {}
        p30_locked = float(exc_locked.get(0.30, exc_locked.get(0.3, 0.0)))
    except Exception:
        p30_locked = 0.0
    try:
        exc_locked = dict(locked_dist.exceedance) if isinstance(locked_dist.exceedance, dict) else {}
        p40_locked = float(exc_locked.get(0.40, exc_locked.get(0.4, 0.0)))
    except Exception:
        p40_locked = 0.0
    for k, v in (locked_dist.exceedance or {}).items():
        with contextlib.suppress(Exception):
            fk = float(k)
            if p30_locked == 0.0 and abs(fk - 0.30) < 1e-9:
                p30_locked = float(v)
            if p40_locked == 0.0 and abs(fk - 0.40) < 1e-9:
                p40_locked = float(v)
    try:
        v_rate = float(
            _resolve_p21(
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
    anchor_key21 = (
        f"{float(cost_cfg.commission_bps or 0.0):.6f}_"
        f"{float(cost_cfg.slippage_bps or 0.0):.6f}_{float(participation):.6f}_P21"
    )
    if anchor_key21 not in _b1_gate_anchor_cache:
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
        # compute B1 locked for gate
        b1_locked_rets: list[float] = []
        try:
            b1_daily = getattr(getattr(b1_rolling, "backtest", None), "daily", None)
            if b1_daily is not None and hasattr(b1_daily, "columns"):
                ret_col_b1 = "ret" if "ret" in b1_daily.columns else ("return" if "return" in b1_daily.columns else None)
                if ret_col_b1 is not None:
                    sess_b1 = cal.sessions(start, end)
                    dmap_b1: dict[date, float] = {}
                    for row in b1_daily.iter_rows(named=True):
                        d = row.get("date")
                        r = row.get(ret_col_b1)
                        if d is None:
                            continue
                        try:
                            dmap_b1[d] = float(r) if r is not None else 0.0
                        except Exception:
                            dmap_b1[d] = 0.0
                    b1_daily_list = [float(dmap_b1.get(d, 0.0)) for d in sess_b1]
                    b1_locked_rets = locked_window_returns(b1_daily_list, horizon, 0.40)
                else:
                    b1_locked_rets = []
            else:
                b1_locked_rets = []
        except Exception:
            b1_locked_rets = []
        if b1_locked_rets:
            b1_locked_dist = ReturnDistribution.summarise(
                name="B1_locked",
                returns=b1_locked_rets,
                horizon=horizon,
                thresholds=thresholds,
                tail_weights=tail_weights,
            )
        else:
            b1_locked_dist = ReturnDistribution.summarise(
                name="baseline.mom20_top1",
                returns=list(b1_rolling.returns),
                horizon=horizon,
                thresholds=thresholds,
                tail_weights=tail_weights,
                givebacks=list(getattr(b1_rolling, "givebacks", ())),
            )
        _b1_gate_anchor_cache[anchor_key21] = _b1_gate_p21(b1_locked_dist)
    b1_p30_21, b1_p40_21, b1_cvar_21 = _b1_gate_anchor_cache[anchor_key21]
    gate_status21, gate_fails21 = evaluate_adoption_gates(
        p30_locked,
        b1_p30_21,
        p40_locked,
        b1_p40_21,
        float(locked_dist.cvar_05),
        b1_cvar_21,
        v_rate,
    )
    # ruin uses unlocked
    ruin = 0.0
    try:
        ruin = float(_ruin_p21(list(rolling.returns), -0.25))
    except Exception:
        ruin = 0.0
    # B0 objective gates (use locked_dist vs B0)
    b0_p30 = 0.0
    b0_p40 = 0.0
    b0_cvar = 0.0
    obj_res = None
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
        for kk, vv in (b0_dist.exceedance or {}).items():
            with contextlib.suppress(Exception):
                fk = float(kk)
                if abs(fk - 0.30) < 1e-9:
                    b0_p30 = float(vv)
                if abs(fk - 0.40) < 1e-9:
                    b0_p40 = float(vv)
        if b0_p30 == 0.0:
            try:
                exc_b0 = dict(b0_dist.exceedance) if isinstance(b0_dist.exceedance, dict) else {}
                b0_p30 = float(exc_b0.get(0.30, exc_b0.get(0.3, 0.0)))
            except Exception:
                b0_p30 = 0.0
        if b0_p40 == 0.0:
            try:
                exc_b0 = dict(b0_dist.exceedance) if isinstance(b0_dist.exceedance, dict) else {}
                b0_p40 = float(exc_b0.get(0.40, exc_b0.get(0.4, 0.0)))
            except Exception:
                b0_p40 = 0.0
        b0_cvar = float(b0_dist.cvar_05)
        _cfg_p21 = _OGC_p21.from_yaml(config_path("gates"))
        obj_res = _eval_obj_p21(locked_dist, b0_dist, _cfg_p21)
    except Exception:
        obj_res = None
    if obj_res is not None:
        summary["objective_gate_status"] = str(obj_res.status)
        summary["objective_gate_fails"] = list(obj_res.failures)
        summary["objective_ruin_probability"] = float(obj_res.ruin_probability)
        ruin = float(obj_res.ruin_probability)
        if str(obj_res.status) != "PASS":
            gate_status21 = "FAIL"
            for _fail in obj_res.failures:
                if _fail not in gate_fails21:
                    gate_fails21.append(str(_fail))
    summary["p_gt_30"] = float(p30_locked)
    summary["p_gt_40"] = float(p40_locked)
    summary["p_gt_30_unlocked"] = float(p30_unlocked)
    summary["p_gt_40_unlocked"] = float(p40_unlocked)
    summary["p_gt_40_locked"] = float(p40_locked)
    summary["b1_p_gt_30"] = float(b1_p30_21)
    summary["b1_p_gt_40"] = float(b1_p40_21)
    summary["b1_cvar_05"] = float(b1_cvar_21)
    summary["b0_p_gt_30"] = float(b0_p30)
    summary["b0_p_gt_40"] = float(b0_p40)
    summary["b0_cvar_05"] = float(b0_cvar)
    summary["vehicle_mult2_rate"] = float(v_rate)
    summary["vehicle_mult2_rate_source"] = "session_path"
    summary["ruin"] = float(ruin)
    summary["adoption_gate_status"] = str(gate_status21)
    summary["adoption_gate_fails"] = list(gate_fails21)
    summary["eval_mode"] = str(eval_mode)
    logger.info(
        f"[EVAL] adoption_gate model=P21 status={gate_status21} fails={gate_fails21} "
        f"p_gt_30={_fmt(p30_locked)} b1={_fmt(b1_p30_21)} p_gt_40={_fmt(p40_locked)} b1={_fmt(b1_p40_21)} "
        f"p_gt_40_unlocked={_fmt(p40_unlocked)} p_gt_40_locked={_fmt(p40_locked)} "
        f"vehicle_mult2_rate={_fmt(v_rate)} ruin={_fmt(ruin)} eval_mode={eval_mode}"
    )


CORE_HOOKS: Final[dict[str, _Hook]] = {
    "sticky.impulse_crash": _hook_impulse_crash,
}

__all__ = ["CORE_HOOKS"]
