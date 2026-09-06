"""Sticky mom60 forensics hooks: peak_lock, concentrated (P4)."""

from __future__ import annotations

import contextlib
import logging
from datetime import date
from typing import Any, Final

from src.cli.commands.backtest._core import ForensicsHook as _Hook
from src.cli.commands.backtest._core import _Cell, _fmt
from src.cli.context import _make_eval_control_model
from src.core.config import config_path
from src.strategies.ids import STICKY_MOM60_CONCENTRATED, STICKY_MOM60_PEAK_LOCK
from src.strategies.sticky.overlays import overlay_param
from src.tournament.distribution import ReturnDistribution

logger = logging.getLogger(__name__)


def _hook_mom60_peak_lock(cell: _Cell) -> None:
    from src.tournament.distribution import championship_lock_returns as _champ_p24  # noqa: I001
    from src.tournament.distribution import evaluate_p24_adoption_gates as _eval_p24  # noqa: I001
    from src.tournament.distribution import b1_gate_anchors_from_distribution as _b1_gate_p24  # noqa: I001
    from src.tournament.distribution import locked_window_returns as _locked_p24  # noqa: I001
    from src.tournament.distribution import resolve_adoption_vehicle_rate as _resolve_p24  # noqa: I001

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
    _b1_gate_dist_cache_p24 = cell.b1_dist_cache_p24
    eval_mode = cell.eval_mode
    rolling = cell.rolling
    cal = cell.cal
    start = cell.start
    end = cell.end
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
    # championship lock on daily for P24
    champ_rets: list[float] = []
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
                champ_daily = [float(dmap.get(d, 0.0)) for d in sess]
                _p24_lock = float(overlay_param(STICKY_MOM60_PEAK_LOCK, "lock_level", default=0.50))
                _p24_trail = float(overlay_param(STICKY_MOM60_PEAK_LOCK, "trail", default=0.0, aliases=("trail_level",)))
                champ_rets = _champ_p24(champ_daily, horizon, _p24_lock, _p24_trail)
            else:
                champ_rets = []
        else:
            champ_rets = []
    except Exception:
        champ_rets = []
    if champ_rets:
        champ_dist = ReturnDistribution.summarise(
            name="P24_champ",
            returns=champ_rets,
            horizon=horizon,
            thresholds=thresholds,
            tail_weights=tail_weights,
        )
    else:
        champ_dist = dist
    try:
        exc_champ = dict(champ_dist.exceedance) if isinstance(champ_dist.exceedance, dict) else {}
        p30_c = float(exc_champ.get(0.30, exc_champ.get(0.3, 0.0)))
    except Exception:
        p30_c = 0.0
    try:
        exc_champ = dict(champ_dist.exceedance) if isinstance(champ_dist.exceedance, dict) else {}
        p40_c = float(exc_champ.get(0.40, exc_champ.get(0.4, 0.0)))
    except Exception:
        p40_c = 0.0
    try:
        exc_champ = dict(champ_dist.exceedance) if isinstance(champ_dist.exceedance, dict) else {}
        p50_c = float(exc_champ.get(0.50, exc_champ.get(0.5, 0.0)))
    except Exception:
        p50_c = 0.0
    for k, v in (champ_dist.exceedance or {}).items():
        with contextlib.suppress(Exception):
            fk = float(k)
            if p30_c == 0.0 and abs(fk - 0.30) < 1e-9:
                p30_c = float(v)
            if p40_c == 0.0 and abs(fk - 0.40) < 1e-9:
                p40_c = float(v)
            if p50_c == 0.0 and abs(fk - 0.50) < 1e-9:
                p50_c = float(v)
    try:
        v_rate = float(
            _resolve_p24(
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
    anchor_key24 = (
        f"{float(cost_cfg.commission_bps or 0.0):.6f}_"
        f"{float(cost_cfg.slippage_bps or 0.0):.6f}_{float(participation):.6f}_P24"
    )
    if anchor_key24 not in _b1_gate_anchor_cache:
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
                    b1_locked_rets = _locked_p24(b1_daily_list, horizon, 0.40)
                else:
                    b1_locked_rets = []
            else:
                b1_locked_rets = []
        except Exception:
            b1_locked_rets = []
        if b1_locked_rets:
            b1_locked_dist: Any = ReturnDistribution.summarise(
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
        _b1_gate_anchor_cache[anchor_key24] = _b1_gate_p24(b1_locked_dist)
        _b1_gate_dist_cache_p24[anchor_key24] = b1_locked_dist
    else:
        b1_locked_dist = _b1_gate_dist_cache_p24.get(anchor_key24)
        if b1_locked_dist is None:
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
            b1_locked_rets = []
            try:
                b1_daily = getattr(getattr(b1_rolling, "backtest", None), "daily", None)
                if b1_daily is not None and hasattr(b1_daily, "columns"):
                    ret_col_b1 = "ret" if "ret" in b1_daily.columns else ("return" if "return" in b1_daily.columns else None)
                    if ret_col_b1 is not None:
                        sess_b1 = cal.sessions(start, end)
                        dmap_b1 = {}
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
                        b1_locked_rets = _locked_p24(b1_daily_list, horizon, 0.40)
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
            _b1_gate_dist_cache_p24[anchor_key24] = b1_locked_dist
    b1_p30_24, b1_p40_24, b1_cvar_24 = _b1_gate_anchor_cache[anchor_key24]
    b1_p50_24 = 0.0
    try:
        exc_b1_locked_dist = dict(b1_locked_dist.exceedance) if isinstance(b1_locked_dist.exceedance, dict) else {}
        b1_p50_24 = float(exc_b1_locked_dist.get(0.50, exc_b1_locked_dist.get(0.5, 0.0)))
        for kk, vv in (b1_locked_dist.exceedance or {}).items():
            with contextlib.suppress(Exception):
                if abs(float(kk) - 0.50) < 1e-9:
                    b1_p50_24 = float(vv)
    except Exception:
        b1_p50_24 = 0.0
    gate_status24, gate_fails24 = _eval_p24(
        p30_c,
        b1_p30_24,
        p40_c,
        b1_p40_24,
        p50_c,
        b1_p50_24,
        float(champ_dist.cvar_05),
        b1_cvar_24,
        v_rate,
    )
    summary["p_gt_30"] = float(p30_c)
    summary["p_gt_40"] = float(p40_c)
    summary["p_gt_50"] = float(p50_c)
    summary["b1_p_gt_30"] = float(b1_p30_24)
    summary["b1_p_gt_40"] = float(b1_p40_24)
    summary["b1_p_gt_50"] = float(b1_p50_24)
    summary["b1_cvar_05"] = float(b1_cvar_24)
    summary["vehicle_mult2_rate"] = float(v_rate)
    summary["vehicle_mult2_rate_source"] = "session_path"
    summary["adoption_gate_status"] = str(gate_status24)
    summary["adoption_gate_fails"] = list(gate_fails24)
    summary["eval_mode"] = str(eval_mode)
    logger.info(
        f"[EVAL] adoption_gate model=P24 status={gate_status24} fails={gate_fails24} "
        f"p_gt_30={_fmt(p30_c)} b1={_fmt(b1_p30_24)} p_gt_40={_fmt(p40_c)} b1={_fmt(b1_p40_24)} p_gt_50={_fmt(p50_c)} b1={_fmt(b1_p50_24)} "
        f"vehicle_mult2_rate={_fmt(v_rate)} eval_mode={eval_mode}"
    )


def _hook_mom60_concentrated(cell: _Cell) -> None:
    from src.portfolio.constraints import load_p26_exposure_limits as _load_p26_exp_bt  # noqa: I001
    from src.reporting.exposure_metrics import summarise_realised_exposure as _summarise_exposure_p26  # noqa: I001
    from src.tournament.distribution import execution_faithful_late_lock_returns as _exec_faith_p26  # noqa: I001
    from src.tournament.eval_mode import resolve_eval_flags as _ref_champ_p26
    from src.tournament.objective import ChampionshipObjectiveConfig as _COC_champ_p26  # noqa: I001
    from src.tournament.objective import evaluate_championship_adoption as _eval_champ_p26  # noqa: I001
    from src.tournament.simulator import RollingDiagnostics
    from src.strategies.registry import STRATEGIES

    summary = cell.summary
    panel = cell.panel
    case_config = cell.case_config
    horizon = cell.horizon
    simulator = cell.simulator
    close_map = cell.close_map
    _lev_allowed_resolved = cell.lev_allowed
    _inv_allowed_resolved = cell.inv_allowed
    eval_mode = cell.eval_mode
    rolling = cell.rolling
    cal = cell.cal
    start = cell.start
    end = cell.end
    _path_mode = cell.path_mode
    _shared_cache = cell.shared_cache
    master = cell.master
    try:
        _arm_p26 = float(overlay_param(STICKY_MOM60_CONCENTRATED, "arm", default=0.50))
    except Exception:
        _arm_p26 = 0.50
    try:
        _lr_p26 = int(overlay_param(STICKY_MOM60_CONCENTRATED, "lock_remaining", default=5))
    except Exception:
        _lr_p26 = 5
    try:
        _mg26_bt = float(_load_p26_exp_bt()[1])
    except Exception:
        _mg26_bt = 1.90
    _daily_p26: list[float] = []
    try:
        daily_df_p26 = getattr(getattr(rolling, "backtest", None), "daily", None)
        if daily_df_p26 is not None and hasattr(daily_df_p26, "columns"):
            ret_col_p26 = "ret" if "ret" in daily_df_p26.columns else ("return" if "return" in daily_df_p26.columns else None)
            if ret_col_p26 is not None:
                sess_p26 = cal.sessions(start, end)
                dmap_p26: dict[date, float] = {}
                for row in daily_df_p26.iter_rows(named=True):
                    d = row.get("date")
                    r = row.get(ret_col_p26)
                    if d is None:
                        continue
                    try:
                        dmap_p26[d] = float(r) if r is not None else 0.0
                    except Exception:
                        dmap_p26[d] = 0.0
                _daily_p26 = [float(dmap_p26.get(d, 0.0)) for d in sess_p26]
    except Exception:
        _daily_p26 = []
    try:
        _exec_overlay_p26 = _exec_faith_p26(_daily_p26, horizon, _arm_p26, _lr_p26) if _daily_p26 else []
        incumbent_returns_p26: list[float] = []
        try:
            b21_model_p26 = STRATEGIES["sticky.impulse_crash"]()
            _b21_flags_p26 = _ref_champ_p26(b21_model_p26, eval_mode)
            from src.portfolio.constraints import resolve_exposure_limits_for_model as _resolve_exp_p26

            _p21_alpha_limits_p26 = _resolve_exp_p26("sticky.impulse_crash", comparison_mode="alpha_equal")
            b21_rolling_p26 = simulator.run_rolling(
                b21_model_p26,
                panel,
                case_config,
                horizon=horizon,
                path_dependent=_b21_flags_p26.path_dependent,
                path_dependent_mode=_path_mode,
                session_cache=_shared_cache,
                leverage_allowed=_lev_allowed_resolved,
                inverse_allowed=_inv_allowed_resolved,
                close_map=close_map,
                exposure_limits=_p21_alpha_limits_p26,
            )
            incumbent_returns_p26 = list(b21_rolling_p26.returns)
        except Exception:
            incumbent_returns_p26 = []
        raw_returns_champ_p26 = list(rolling.returns)
        gross_viol_p26 = None
        effective_gross_max_p26 = None
        try:
            _diag_p26 = getattr(rolling, "diagnostics", None)
            if _diag_p26 is not None:
                gross_viol_p26 = getattr(_diag_p26, "gross_violation_count", None)
                effective_gross_max_p26 = getattr(_diag_p26, "effective_gross_max", None)
                RollingDiagnostics(gross_violation_count=gross_viol_p26, effective_gross_max=effective_gross_max_p26, turnover_mean=None, fill_count=None, unfilled_count=None)
                if gross_viol_p26 is not None:
                    summary["gross_violation_count"] = int(gross_viol_p26)
                else:
                    summary["gross_violation_count"] = None
            else:
                _bt_p26 = getattr(rolling, "backtest", None)
                _trades_p26 = getattr(_bt_p26, "trades", None) if _bt_p26 is not None else None
                if _trades_p26 is not None:
                    _exp_p26 = _summarise_exposure_p26(
                        cal.sessions(start, end),
                        _trades_p26,
                        (),
                        master,
                        epsilon=1e-9,
                        max_gross=_mg26_bt,
                    )
                    gross_viol_p26 = int(_exp_p26.gross_violation_count)
                    effective_gross_max_p26 = float(_exp_p26.effective_gross_max)
                    summary["gross_violation_count"] = gross_viol_p26
                    summary["effective_gross_max"] = effective_gross_max_p26
                else:
                    gross_viol_p26 = None
                    summary["gross_violation_count"] = None
        except Exception:
            gross_viol_p26 = None
            effective_gross_max_p26 = None
        try:
            _champ_cfg_p26 = _COC_champ_p26.from_yaml(
                config_path("gates"),
                config_path("portfolio"),
            )
        except Exception:
            _champ_cfg_p26 = None
        if _champ_cfg_p26 is not None:
            exec_parity_p26 = bool(
                _exec_overlay_p26
                and incumbent_returns_p26
                and raw_returns_champ_p26
                and len(_exec_overlay_p26) == len(raw_returns_champ_p26)
                and len(_exec_overlay_p26) == len(incumbent_returns_p26)
            )
            _champ_result_p26 = _eval_champ_p26(
                candidate_returns=_exec_overlay_p26,
                incumbent_returns=incumbent_returns_p26,
                raw_returns=raw_returns_champ_p26,
                horizon=horizon,
                config=_champ_cfg_p26,
                execution_parity=exec_parity_p26,
                gross_violation_count=gross_viol_p26,
                era_pairs=None,
            )
            summary["executable_overlay"] = {
                "p_gt_30": float(sum(1 for r in _exec_overlay_p26 if r > 0.30) / len(_exec_overlay_p26))
                if _exec_overlay_p26
                else 0.0
            }
            summary["raw"] = {
                "p_gt_30": float(sum(1 for r in raw_returns_champ_p26 if r > 0.30) / len(raw_returns_champ_p26))
                if raw_returns_champ_p26
                else 0.0
            }
            summary["incumbent_p21"] = {
                "p_gt_30": float(sum(1 for r in incumbent_returns_p26 if r > 0.30) / len(incumbent_returns_p26))
                if incumbent_returns_p26
                else 0.0
            }
            summary["championship_gate_status"] = str(_champ_result_p26.status)
            summary["championship_gate_failures"] = list(_champ_result_p26.failures)
            summary["adoption_gate_status"] = str(_champ_result_p26.status)
            summary["adoption_gate_fails"] = list(_champ_result_p26.failures)
            logger.info(
                f"[EVAL] championship_gate model=P26 status={_champ_result_p26.status} "
                f"failures={_champ_result_p26.failures} gross_violation_count={gross_viol_p26} "
                f"execution_parity={exec_parity_p26}"
            )
    except Exception as _exc_champ_p26:
        logger.warning(f"[EVAL] P26 championship gate failed {_exc_champ_p26!r}")


MOM60_HOOKS: Final[dict[str, _Hook]] = {
    "sticky.mom60_peak_lock": _hook_mom60_peak_lock,
    "sticky.mom60_concentrated": _hook_mom60_concentrated,
}

__all__ = ["MOM60_HOOKS"]
