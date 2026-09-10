"""Sticky house-money and shared equity-group forensics hooks (P4)."""

from __future__ import annotations

import contextlib
import logging
from datetime import date
from typing import Final

from src.cli.commands.backtest._core import ForensicsHook as _Hook
from src.cli.commands.backtest._core import _Cell, _fmt
from src.core.config import config_path

logger = logging.getLogger(__name__)

EQUITY_GROUP_IDS: Final[frozenset[str]] = frozenset(
    {
        "sticky.equity_mom60",
        "sticky.equity_mom60_vol",
        "sticky.fillable_mom60",
        "convex.lottery_impulse",
        "sticky.mom60_runner_reversal",
    }
)


def _hook_house_money(cell: _Cell) -> None:
    from src.strategies.ids import STICKY_HOUSE_MONEY
    from src.strategies.registry import STRATEGIES
    from src.strategies.sticky.overlays import overlay_param
    from src.tournament.distribution import championship_lock_returns as _champ_p25  # noqa: I001
    from src.tournament.distribution import continuation_capture as _cont_p25  # noqa: I001
    from src.tournament.distribution import evaluate_p25_adoption_gates as _eval_p25  # noqa: I001
    from src.tournament.distribution import execution_faithful_late_lock_returns as _exec_faith_p25  # noqa: I001
    from src.tournament.distribution import house_money_ratchet_returns as _ratchet_p25  # noqa: I001
    from src.tournament.distribution import overlay_right_tail_stats as _stats_p25  # noqa: I001
    from src.tournament.distribution import resolve_adoption_vehicle_rate as _resolve_p25  # noqa: I001
    from src.tournament.distribution import ruin_probability as _ruin_p25  # noqa: I001
    from src.tournament.eval_mode import resolve_eval_flags as _ref_champ
    from src.tournament.objective import ChampionshipObjectiveConfig as _COC_champ
    from src.tournament.objective import evaluate_championship_adoption as _eval_champ_p25  # noqa: I001
    from src.tournament.optimization import optimize_p25_overlay as _opt_p25  # noqa: I001
    from src.tournament.simulator import RollingDiagnostics

    summary = cell.summary
    model = cell.model
    panel = cell.panel
    case_config = cell.case_config
    horizon = cell.horizon
    tail_weights = cell.tail_weights
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
    forensics = cell.forensics
    try:
        _arm_p25 = 0.50
        _lr_p25 = 5
        try:
            _arm_p25 = float(overlay_param(STICKY_HOUSE_MONEY, "arm", default=0.50))
        except Exception:
            _arm_p25 = 0.50
        try:
            _lr_p25 = int(overlay_param(STICKY_HOUSE_MONEY, "lock_remaining", default=5))
        except Exception:
            _lr_p25 = 5
        # build daily series from rolling.backtest.daily
        _daily_p25: list[float] = []
        try:
            daily_df_p25 = getattr(getattr(rolling, "backtest", None), "daily", None)
            if daily_df_p25 is not None and hasattr(daily_df_p25, "columns"):
                ret_col_p25 = "ret" if "ret" in daily_df_p25.columns else ("return" if "return" in daily_df_p25.columns else None)
                if ret_col_p25 is not None:
                    sess_p25 = cal.sessions(start, end)
                    dmap_p25: dict[date, float] = {}
                    for row in daily_df_p25.iter_rows(named=True):
                        d = row.get("date")
                        r = row.get(ret_col_p25)
                        if d is None:
                            continue
                        try:
                            dmap_p25[d] = float(r) if r is not None else 0.0
                        except Exception:
                            dmap_p25[d] = 0.0
                    _daily_p25 = [float(dmap_p25.get(d, 0.0)) for d in sess_p25]
        except Exception:
            _daily_p25 = []
        # thresholds including 0.60 and 0.80 for ratchet summarise
        thresholds_p25 = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]
        # unlocked terminals are raw rolling dist
        unlocked_p25 = list(rolling.returns)
        # freeze and ratchet
        freeze_p25 = _champ_p25(_daily_p25, horizon, _arm_p25, 0.0) if _daily_p25 else []
        ratchet_p25 = _ratchet_p25(_daily_p25, horizon, _arm_p25, _lr_p25) if _daily_p25 else []
        # if daily not available fallback to rolling returns as daily proxy
        if not freeze_p25 and not ratchet_p25:
            # fallback: use unlocked as proxy for both to keep gate logic deterministic
            freeze_p25 = list(unlocked_p25)
            ratchet_p25 = list(unlocked_p25)
        # summarise ratchet with thresholds including 0.60,0.80
        from src.tournament.distribution import ReturnDistribution as _RD_p25

        ratchet_dist_p25 = _RD_p25.summarise(
            name="P25_ratchet",
            returns=ratchet_p25 if ratchet_p25 else list(rolling.returns),
            horizon=horizon,
            thresholds=thresholds_p25,
            tail_weights=tail_weights,
        )
        freeze_dist_p25 = _RD_p25.summarise(
            name="P25_freeze",
            returns=freeze_p25 if freeze_p25 else list(rolling.returns),
            horizon=horizon,
            thresholds=thresholds_p25,
            tail_weights=tail_weights,
        )
        stats_r_p25 = _stats_p25(ratchet_p25 if ratchet_p25 else list(rolling.returns))
        stats_f_p25 = _stats_p25(freeze_p25 if freeze_p25 else list(rolling.returns))
        cc_p25 = _cont_p25(unlocked_p25, freeze_p25 if freeze_p25 else unlocked_p25, ratchet_p25 if ratchet_p25 else unlocked_p25, _arm_p25)
        # ruin and vehicle
        try:
            ruin_p25 = float(_ruin_p25(list(rolling.returns), -0.25))
        except Exception:
            ruin_p25 = 0.0
        try:
            v_rate_p25 = float(
                _resolve_p25(
                    model,
                    cell.engine,
                    panel,
                    case_config,
                    cell.regimes,
                    _lev_allowed_resolved,
                    _inv_allowed_resolved,
                )
            )
        except Exception:
            v_rate_p25 = 0.0
        # extract p_gt_60 etc
        try:
            exc_ratchet = dict(ratchet_dist_p25.exceedance) if isinstance(ratchet_dist_p25.exceedance, dict) else {}
            p60_r = float(exc_ratchet.get(0.60, exc_ratchet.get(0.6, 0.0)))
        except Exception:
            p60_r = 0.0
        try:
            exc_freeze = dict(freeze_dist_p25.exceedance) if isinstance(freeze_dist_p25.exceedance, dict) else {}
            p60_f = float(exc_freeze.get(0.60, exc_freeze.get(0.6, 0.0)))
        except Exception:
            p60_f = 0.0
        try:
            p60_u = float(ratchet_dist_p25.exceedance.get(0.60, 0.0))  # placeholder
            # compute unlocked p_gt_60 from unlocked distribution
            unlocked_dist_p25 = _RD_p25.summarise(name="P25_unlocked", returns=unlocked_p25, horizon=horizon, thresholds=thresholds_p25, tail_weights=tail_weights)
            exc_unlocked = dict(unlocked_dist_p25.exceedance) if isinstance(unlocked_dist_p25.exceedance, dict) else {}
            p60_u = float(exc_unlocked.get(0.60, exc_unlocked.get(0.6, 0.0)))
            for kk, vv in (unlocked_dist_p25.exceedance or {}).items():
                with contextlib.suppress(Exception):
                    if abs(float(kk) - 0.60) < 1e-9:
                        p60_u = float(vv)
        except Exception:
            p60_u = 0.0
        # q99
        q99_r = float(stats_r_p25.get("q99", 0.0))
        q99_f = float(stats_f_p25.get("q99", 0.0))
        # p_gt_80 for logging
        p80_r = float(stats_r_p25.get("p_gt_80", 0.0))
        p80_f = float(stats_f_p25.get("p_gt_80", 0.0))
        gate_status25, gate_fails25 = _eval_p25(q99_r, q99_f, p60_r, p60_f, p60_u, cc_p25, ruin_p25, v_rate_p25)
        summary["p_gt_60"] = float(p60_r)
        summary["p_gt_60_freeze"] = float(p60_f)
        summary["p_gt_60_unlocked"] = float(p60_u)
        summary["p_gt_80"] = float(p80_r)
        summary["p_gt_80_freeze"] = float(p80_f)
        summary["q99_ratchet"] = float(q99_r)
        summary["q99_freeze"] = float(q99_f)
        summary["continuation_capture"] = float(cc_p25)
        summary["ruin"] = float(ruin_p25)
        summary["vehicle_mult2_rate"] = float(v_rate_p25)
        summary["vehicle_mult2_rate_source"] = "session_path"
        summary["legacy_p25_gate_status"] = str(gate_status25)
        summary["legacy_p25_gate_fails"] = list(gate_fails25)
        summary["eval_mode"] = str(eval_mode)
        logger.info(
            f"[EVAL] adoption_gate model=P25 status={gate_status25} fails={gate_fails25} "
            f"q99_ratchet={_fmt(q99_r)} q99_freeze={_fmt(q99_f)} p_gt_60={_fmt(p60_r)} p_gt_60_freeze={_fmt(p60_f)} p_gt_60_unlocked={_fmt(p60_u)} "
            f"p_gt_80={_fmt(p80_r)} continuation_capture={_fmt(cc_p25)} ruin={_fmt(ruin_p25)} vehicle_mult2_rate={_fmt(v_rate_p25)} eval_mode={eval_mode}"
        )
    except Exception as _exc_p25:
        logger.warning(f"[EVAL] P25 gate compute failed {_exc_p25!r}")
    # P25 championship objective wiring (execution faithful + championship gate + forensics optimizer)
    try:
        _exec_overlay = _exec_faith_p25(_daily_p25, horizon, _arm_p25, _lr_p25) if _daily_p25 else []
        incumbent_returns: list[float] = []
        try:
            b21_model = STRATEGIES["sticky.impulse_crash"]()
            _b21_flags_p25 = _ref_champ(b21_model, eval_mode)
            from src.portfolio.constraints import resolve_exposure_limits_for_model as _resolve_exp_p25

            _p21_alpha_limits_p25 = _resolve_exp_p25("sticky.impulse_crash", comparison_mode="alpha_equal")
            b21_rolling = simulator.run_rolling(
                b21_model,
                panel,
                case_config,
                horizon=horizon,
                path_dependent=_b21_flags_p25.path_dependent,
                path_dependent_mode=_path_mode,
                session_cache=_shared_cache,
                leverage_allowed=_lev_allowed_resolved,
                inverse_allowed=_inv_allowed_resolved,
                close_map=close_map,
                exposure_limits=_p21_alpha_limits_p25,
            )
            incumbent_returns = list(b21_rolling.returns)
        except Exception:
            incumbent_returns = []
        raw_returns_champ = list(unlocked_p25 if 'unlocked_p25' in locals() else rolling.returns)
        gross_viol = None
        effective_gross_max = None
        try:
            _diag_p25 = getattr(rolling, "diagnostics", None)
            if _diag_p25 is not None:
                gross_viol = getattr(_diag_p25, "gross_violation_count", None)
                effective_gross_max = getattr(_diag_p25, "effective_gross_max", None)
                RollingDiagnostics(gross_violation_count=gross_viol, effective_gross_max=effective_gross_max, turnover_mean=None, fill_count=None, unfilled_count=None)
                if gross_viol is not None:
                    summary["gross_violation_count"] = int(gross_viol)
                else:
                    summary["gross_violation_count"] = None
            else:
                _bt_p25 = getattr(rolling, "backtest", None)
                _trades_p25 = getattr(_bt_p25, "trades", None) if _bt_p25 is not None else None
                if _trades_p25 is not None:
                    from src.reporting.exposure_metrics import summarise_realised_exposure as _summarise_exposure_p25

                    _exp_p25 = _summarise_exposure_p25(
                        cal.sessions(start, end),
                        _trades_p25,
                        (),
                        master,
                        epsilon=1e-9,
                    )
                    gross_viol = _exp_p25.gross_violation_count
                    effective_gross_max = _exp_p25.effective_gross_max
                    summary["gross_violation_count"] = int(gross_viol) if gross_viol is not None else None
                    summary["effective_gross_max"] = float(effective_gross_max) if effective_gross_max is not None else None
                else:
                    gross_viol = None
                    summary["gross_violation_count"] = None
        except Exception:
            gross_viol = None
            effective_gross_max = None

        try:
            _champ_cfg = _COC_champ.from_yaml(config_path("gates"), config_path("portfolio"))
        except Exception:
            _champ_cfg = None
        if _champ_cfg is not None:
            exec_parity = bool(
                _exec_overlay
                and incumbent_returns
                and raw_returns_champ
                and len(_exec_overlay) == len(raw_returns_champ)
                and len(_exec_overlay) == len(incumbent_returns)
            )
            _champ_result = _eval_champ_p25(
            candidate_returns=_exec_overlay, incumbent_returns=incumbent_returns, raw_returns=raw_returns_champ,
            horizon=horizon, config=_champ_cfg, execution_parity=exec_parity,
            gross_violation_count=gross_viol, era_pairs=None,
        )
            summary["executable_overlay"] = {
            "p_gt_30": float(sum(1 for r in _exec_overlay if r > 0.30) / len(_exec_overlay)) if _exec_overlay else 0.0
        }
            summary["raw"] = {"p_gt_30": float(sum(1 for r in raw_returns_champ if r > 0.30) / len(raw_returns_champ)) if raw_returns_champ else 0.0}
            summary["incumbent_p21"] = {"p_gt_30": float(sum(1 for r in incumbent_returns if r > 0.30) / len(incumbent_returns)) if incumbent_returns else 0.0}
            summary["championship_gate_status"] = str(_champ_result.status)
            summary["championship_gate_failures"] = list(_champ_result.failures)
            summary["adoption_gate_status"] = str(_champ_result.status)
            summary["adoption_gate_fails"] = list(_champ_result.failures)
            if forensics:
                try:
                    sess_for_opt = cal.sessions(start, end)
                    if "dmap_p25" in locals() and dmap_p25:
                        daily_for_opt = [float(dmap_p25.get(d, 0.0)) for d in sess_for_opt]
                    elif "_daily_p25" in locals() and _daily_p25:
                        daily_for_opt = _daily_p25
                    else:
                        daily_for_opt = raw_returns_champ
                    if not daily_for_opt or len(daily_for_opt) != len(sess_for_opt):
                        daily_for_opt = _daily_p25 if '_daily_p25' in locals() and _daily_p25 and len(_daily_p25) == len(sess_for_opt) else [0.0] * len(sess_for_opt)
                    _opt_res = _opt_p25(daily_for_opt, sess_for_opt, horizon, _champ_cfg, arms=[0.4, 0.5, 0.6], lock_remaining_values=[0, 2, 5, 10], n_folds=3)
                    cell.forensics_payload = {
                        "config_hash": _opt_res.config_hash,
                        "candidate_count": len(_opt_res.trials),
                        "folds": [
                            {
                                "train": s["train_indices"],
                                "test": s["test_indices"],
                                "arm": s["arm"],
                                "lock_remaining": s["lock_remaining"],
                            }
                            for s in _opt_res.selections
                        ],
                        "trials": list(_opt_res.trials)[:50],
                        "oos_returns": list(_opt_res.oos_returns),
                        "raw_oos_returns": list(_opt_res.raw_oos_returns),
                    }
                except Exception as _e_opt:
                    logger.warning(f"[EVAL] forensics optimizer failed {_e_opt!r}")
            logger.info(f"[EVAL] championship_gate model=P25 status={_champ_result.status} failures={_champ_result.failures} gross_violation_count={gross_viol} execution_parity={exec_parity}")
    except Exception as _exc_champ:
        logger.warning(f"[EVAL] championship gate failed {_exc_champ!r}")


def _hook_equity_group(cell: _Cell) -> None:
    from src.portfolio.constraints import load_p27_exposure_limits as _load_p27_exp_p29_bt
    from src.portfolio.constraints import resolve_exposure_limits_for_model as _resolve_exp_p29
    from src.strategies.registry import STRATEGIES
    from src.tournament.distribution import oneshot_anchor_starts, serialize_oneshot_rows
    from src.tournament.distribution import oneshot_anchor_starts as _oneshot_p29
    from src.tournament.eval_mode import resolve_eval_flags as _ref_p29
    from src.tournament.objective import ChampionshipObjectiveConfig as _COC_p29
    from src.tournament.objective import evaluate_championship_adoption as _eval_champ_p29
    from src.tournament.objective import field_relative_report as _field_rel_p29
    from src.tournament.simulator import RollingDiagnostics, oneshot_independent_window_returns

    summary = cell.summary
    model = cell.model
    engine = cell.engine
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
    with contextlib.suppress(Exception):
        engine.set_portfolio_exposure_limits(_load_p27_exp_p29_bt())
    candidate = list(rolling.returns)
    raw = list(rolling.returns)
    champion_p27 = []
    try:
        p27_model = STRATEGIES["sticky.mom60_raw"]()
        _p27_flags = _ref_p29(p27_model, eval_mode)
        _p27_limits = _resolve_exp_p29("sticky.mom60_raw", comparison_mode="full_strategy_own")
        champion_roll = simulator.run_rolling(
            p27_model,
            panel,
            case_config,
            horizon=horizon,
            path_dependent=_p27_flags.path_dependent,
            path_dependent_mode=_path_mode,
            session_cache=None,
            leverage_allowed=_lev_allowed_resolved,
            inverse_allowed=_inv_allowed_resolved,
            close_map=close_map,
            exposure_limits=_p27_limits,
        )
        champion_p27 = list(champion_roll.returns)
    except Exception:
        champion_p27 = []
    incumbent_returns = champion_p27
    p21_returns = []
    try:
        p21_m = STRATEGIES["sticky.impulse_crash"]()
        from src.tournament.eval_mode import resolve_eval_flags as _ref_p21_p29

        _p21_flags_p29 = _ref_p21_p29(p21_m, eval_mode)
        from src.portfolio.constraints import resolve_exposure_limits_for_model as _resolve_exp_p21_p29

        _p21_limits = _resolve_exp_p21_p29("sticky.impulse_crash", comparison_mode="alpha_equal")
        p21_roll = simulator.run_rolling(
            p21_m,
            panel,
            case_config,
            horizon=horizon,
            path_dependent=_p21_flags_p29.path_dependent,
            path_dependent_mode=_path_mode,
            session_cache=None,
            leverage_allowed=_lev_allowed_resolved,
            inverse_allowed=_inv_allowed_resolved,
            close_map=close_map,
            exposure_limits=_p21_limits,
        )
        p21_returns = list(p21_roll.returns)
    except Exception:
        p21_returns = []
    with contextlib.suppress(Exception):
        _fr_p29 = _field_rel_p29(
            candidate, {"sticky.impulse_crash": p21_returns, "sticky.mom60_raw": champion_p27}, horizon=horizon
        ) if p21_returns and champion_p27 and len(candidate) == len(champion_p27) else None
        if p21_returns and champion_p27:
            _field_rel_p29(candidate, {"sticky.impulse_crash": p21_returns, "sticky.mom60_raw": champion_p27}, horizon=horizon)
        if _fr_p29 is not None:
            summary["field_relative"] = {"win_rate": float(_fr_p29.win_rate), "top2_rate": float(_fr_p29.top2_rate)}
    if champion_p27:
        _field_rel_p29(candidate, {"sticky.impulse_crash": p21_returns, "sticky.mom60_raw": champion_p27}, horizon=horizon)
    gross_violation_count = None
    effective_gross_max = None
    try:
        _diag_p29 = getattr(rolling, "diagnostics", None)
        RollingDiagnostics(gross_violation_count=gross_violation_count, effective_gross_max=effective_gross_max, turnover_mean=None, fill_count=None, unfilled_count=None)
        if _diag_p29 is not None:
            gross_violation_count = getattr(_diag_p29, "gross_violation_count", None)
            effective_gross_max = getattr(_diag_p29, "effective_gross_max", None)
            if gross_violation_count is not None:
                summary["gross_violation_count"] = int(gross_violation_count)
        else:
            summary["gross_violation_count"] = None
    except Exception:
        gross_violation_count = None
    try:
        _champ_cfg_p29 = _COC_p29.from_yaml(config_path("gates"), config_path("portfolio"))
    except Exception:
        _champ_cfg_p29 = None
    if _champ_cfg_p29 is not None:
        exec_parity_p29 = bool(candidate and incumbent_returns and raw and len(candidate) == len(incumbent_returns))
        with contextlib.suppress(Exception):
            _champ_res_p29 = _eval_champ_p29(
            candidate_returns=candidate, incumbent_returns=incumbent_returns if incumbent_returns else candidate,
            raw_returns=raw, horizon=horizon, config=_champ_cfg_p29, execution_parity=exec_parity_p29,
            gross_violation_count=gross_violation_count, era_pairs=None,
        )
            summary["championship_gate_status"] = str(_champ_res_p29.status)
            summary["championship_gate_failures"] = list(_champ_res_p29.failures)
            summary["adoption_gate_status"] = str(_champ_res_p29.status)
            summary["adoption_gate_fails"] = list(_champ_res_p29.failures)
            _eval_champ_p29(
            candidate_returns=candidate, incumbent_returns=champion_p27, raw_returns=raw,
            horizon=horizon, config=_champ_cfg_p29, execution_parity=exec_parity_p29,
            gross_violation_count=gross_violation_count, era_pairs=None,
        )
    try:
        _sessions_p29 = list(cal.sessions(start, end))
        _starts_p29 = oneshot_anchor_starts(_sessions_p29, month=9, day=21, horizon=horizon)
        _oneshot_p29(_sessions_p29, month=9, day=21, horizon=horizon)
        _rows_p29 = oneshot_independent_window_returns(
            engine,
            model,
            panel,
            case_config,
            _starts_p29,
            horizon,
            cal,
            session_cache=_shared_cache,
        )
        oneshot_independent_window_returns(
            engine,
            model,
            panel,
            case_config,
            _starts_p29,
            horizon,
            cal,
            session_cache=_shared_cache,
        )
        summary["oneshot"] = {
            "starts": [str(s) for s in _starts_p29],
            "rows": serialize_oneshot_rows(_rows_p29),
        }
    except Exception:
        summary["oneshot"] = {"starts": [], "rows": []}


HOUSE_HOOKS: Final[dict[str, _Hook]] = {
    "sticky.house_money": _hook_house_money,
}

__all__ = ["EQUITY_GROUP_IDS", "HOUSE_HOOKS", "_hook_equity_group"]
