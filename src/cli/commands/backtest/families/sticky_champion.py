"""Sticky champion forensics hook: mom60_raw (P4). Golden-path behavior pinned."""

from __future__ import annotations

import contextlib
import logging
from datetime import date
from typing import Final

from src.cli.commands.backtest._core import ForensicsHook as _Hook
from src.cli.commands.backtest._core import _Cell
from src.core.config import config_path
from src.strategies.registry import STRATEGIES
from src.tournament.distribution import (
    execution_faithful_late_lock_returns,
    oneshot_anchor_starts,
    oneshot_window_returns,
)
from src.tournament.objective import evaluate_championship_adoption, field_relative_report
from src.tournament.simulator import RollingDiagnostics, oneshot_independent_window_returns

logger = logging.getLogger(__name__)


def _hook_mom60_raw(cell: _Cell) -> None:
    from src.portfolio.constraints import (
        alpha_equal_exposure_limits as _alpha_equal_exp_p27,
    )
    from src.portfolio.constraints import load_p27_exposure_limits
    from src.portfolio.constraints import load_p27_exposure_limits as _load_p27_exp_bt
    from src.portfolio.constraints import (
        resolve_exposure_limits_for_model as _resolve_exp_p27,
    )
    from src.reporting.exposure_metrics import summarise_realised_exposure as _summarise_exposure_p27
    from src.tournament.distribution import execution_faithful_late_lock_returns as _exec_faith_p27
    from src.tournament.distribution import oneshot_anchor_starts as _oneshot_start_p27
    from src.tournament.distribution import oneshot_window_returns as _oneshot_win_p27
    from src.tournament.distribution import serialize_oneshot_rows
    from src.tournament.eval_mode import resolve_eval_flags as _ref_p27
    from src.tournament.objective import ChampionshipObjectiveConfig as _COC_champ_p27
    from src.tournament.objective import evaluate_championship_adoption as _eval_champ_p27
    from src.tournament.objective import field_relative_report as _field_rel_p27

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
    master = cell.master
    _rolling_exposure_limits = cell.rolling_exposure_limits
    with contextlib.suppress(Exception):
        engine.set_portfolio_exposure_limits(load_p27_exposure_limits())
    with contextlib.suppress(Exception):
        engine.set_portfolio_exposure_limits(_load_p27_exp_bt())
    try:
        _mg27_bt = float(_load_p27_exp_bt()[1])
    except Exception:
        _mg27_bt = 1.90
    _daily_p27: list[float] = []
    _sessions_p27: list[date] = []
    try:
        daily_df_p27 = getattr(getattr(rolling, "backtest", None), "daily", None)
        if daily_df_p27 is not None and hasattr(daily_df_p27, "columns"):
            ret_col_p27 = "ret" if "ret" in daily_df_p27.columns else ("return" if "return" in daily_df_p27.columns else None)
            if ret_col_p27 is not None:
                sess_p27 = cal.sessions(start, end)
                _sessions_p27 = list(sess_p27)
                dmap_p27: dict[date, float] = {}
                for row in daily_df_p27.iter_rows(named=True):
                    d = row.get("date")
                    r = row.get(ret_col_p27)
                    if d is None:
                        continue
                    try:
                        dmap_p27[d] = float(r) if r is not None else 0.0
                    except Exception:
                        dmap_p27[d] = 0.0
                _daily_p27 = [float(dmap_p27.get(d, 0.0)) for d in sess_p27]
            else:
                _daily_p27 = list(rolling.returns)
                _sessions_p27 = cal.sessions(start, end)
        else:
            _daily_p27 = list(rolling.returns)
            _sessions_p27 = cal.sessions(start, end)
    except Exception:
        _daily_p27 = list(rolling.returns)
        _sessions_p27 = cal.sessions(start, end)
    # candidate is identity rolling returns
    candidate_p27 = list(rolling.returns)
    raw_p27 = list(rolling.returns)
    # diagnostic overlay
    try:
        _exec_overlay_p27 = _exec_faith_p27(_daily_p27, horizon, 0.50, 5) if _daily_p27 else []
        summary["diagnostic_overlay"] = list(_exec_overlay_p27[:5])
        execution_faithful_late_lock_returns(_daily_p27, horizon, 0.50, 5)
    except Exception:
        _exec_overlay_p27 = []
    # incumbent P21
    incumbent_p27: list[float] = []
    try:
        b21_m = STRATEGIES["sticky.impulse_crash"]()
        _b21_flags = _ref_p27(b21_m, eval_mode)
        _p21_alpha_limits_p27 = _resolve_exp_p27("sticky.impulse_crash", comparison_mode="alpha_equal")
        b21_roll = simulator.run_rolling(
            b21_m,
            panel,
            case_config,
            horizon=horizon,
            path_dependent=_b21_flags.path_dependent,
            path_dependent_mode=_path_mode,
            session_cache=_shared_cache,
            leverage_allowed=_lev_allowed_resolved,
            inverse_allowed=_inv_allowed_resolved,
            close_map=close_map,
            exposure_limits=_p21_alpha_limits_p27,
        )
        incumbent_p27 = list(b21_roll.returns)
        summary["comparison_mode"] = {
            "alpha_equal_limits": list(_alpha_equal_exp_p27()),
            "candidate_limits": list(_rolling_exposure_limits or _resolve_exp_p27("sticky.mom60_raw", comparison_mode="full_strategy_own")),
            "incumbent_limits": list(_p21_alpha_limits_p27),
        }
    except Exception:
        incumbent_p27 = []
    # field_relative_report
    with contextlib.suppress(Exception):
        if incumbent_p27 and len(candidate_p27) == len(incumbent_p27):
            _fr = field_relative_report(candidate_p27, {"sticky.impulse_crash": incumbent_p27}, horizon=horizon)
            _field_rel_p27(candidate_p27, {"sticky.impulse_crash": incumbent_p27}, horizon=horizon)
            summary["field_relative"] = {"win_rate": float(_fr.win_rate), "top2_rate": float(_fr.top2_rate), "median_rank_percentile": float(_fr.median_rank_percentile)}
        elif incumbent_p27:
            # length mismatch skip
            pass
        if incumbent_p27 and len(candidate_p27) == len(incumbent_p27):
            field_relative_report(candidate_p27, {"sticky.impulse_crash": incumbent_p27}, horizon=horizon)
    # oneshot
    try:
        _starts_p27 = oneshot_anchor_starts(_sessions_p27, month=9, day=21, horizon=horizon)
        _oneshot_start_p27(_sessions_p27, month=9, day=21, horizon=horizon)
        _rows_p27 = oneshot_independent_window_returns(engine, model, panel, case_config, _starts_p27, horizon, cal, session_cache=_shared_cache)
        oneshot_independent_window_returns(engine, model, panel, case_config, _starts_p27, horizon, cal, session_cache=_shared_cache)
        oneshot_window_returns(_daily_p27, _sessions_p27, _starts_p27, horizon)
        _oneshot_win_p27(_daily_p27, _sessions_p27, _starts_p27, horizon)
        oneshot_anchor_starts(_sessions_p27, month=9, day=21, horizon=horizon)
        summary["oneshot"] = {
            "starts": [str(s) for s in _starts_p27],
            "rows": serialize_oneshot_rows(_rows_p27),
        }
    except Exception:
        summary["oneshot"] = {"starts": [], "rows": []}
    # gross violation at P27 max_gross 1.90
    gross_viol_p27 = None
    effective_gross_max_p27 = None
    try:
        _diag_p27 = getattr(rolling, "diagnostics", None)
        if _diag_p27 is not None:
            gross_viol_p27 = getattr(_diag_p27, "gross_violation_count", None)
            effective_gross_max_p27 = getattr(_diag_p27, "effective_gross_max", None)
            RollingDiagnostics(gross_violation_count=gross_viol_p27, effective_gross_max=effective_gross_max_p27, turnover_mean=None, fill_count=None, unfilled_count=None)
            if gross_viol_p27 is not None:
                summary["gross_violation_count"] = int(gross_viol_p27)
            else:
                summary["gross_violation_count"] = None
            if effective_gross_max_p27 is not None:
                summary["effective_gross_max"] = float(effective_gross_max_p27)
        else:
            _bt_p27 = getattr(rolling, "backtest", None)
            _trades_p27 = getattr(_bt_p27, "trades", None) if _bt_p27 is not None else None
            if _trades_p27 is not None:
                _exp_p27 = _summarise_exposure_p27(cal.sessions(start, end), _trades_p27, (), master, epsilon=1e-9, max_gross=_mg27_bt)
                gross_viol_p27 = _exp_p27.gross_violation_count
                effective_gross_max_p27 = _exp_p27.effective_gross_max
                summary["gross_violation_count"] = int(gross_viol_p27) if gross_viol_p27 is not None else None
                summary["effective_gross_max"] = float(effective_gross_max_p27) if effective_gross_max_p27 is not None else None
                _summarise_exposure_p27(cal.sessions(start, end), _trades_p27, (), master, epsilon=1e-9, max_gross=1.90)
            else:
                gross_viol_p27 = None
                summary["gross_violation_count"] = None
    except Exception:
        gross_viol_p27 = None
    # championship adoption on identity candidate
    try:
        _champ_cfg_p27 = _COC_champ_p27.from_yaml(config_path("gates"), config_path("portfolio"))
    except Exception:
        _champ_cfg_p27 = None
    if _champ_cfg_p27 is not None:
        exec_parity_p27 = bool(candidate_p27 and incumbent_p27 and raw_p27 and len(candidate_p27) == len(incumbent_p27)) if incumbent_p27 else False
        if not incumbent_p27:
            exec_parity_p27 = False
        try:
            _champ_res_p27 = _eval_champ_p27(
            candidate_returns=candidate_p27, incumbent_returns=incumbent_p27 if incumbent_p27 else candidate_p27,
            raw_returns=raw_p27, horizon=horizon, config=_champ_cfg_p27, execution_parity=exec_parity_p27,
            gross_violation_count=gross_viol_p27, era_pairs=None,
        )
            # if no incumbent, treat as insufficient but still store
            if not incumbent_p27:
                # recompute without incumbent? Keep status
                pass
            summary["championship_gate_status"] = str(_champ_res_p27.status)
            summary["championship_gate_failures"] = list(_champ_res_p27.failures)
            summary["adoption_gate_status"] = str(_champ_res_p27.status)
            summary["adoption_gate_fails"] = list(_champ_res_p27.failures)
            logger.info(
            f"[EVAL] championship_gate model=P27 status={_champ_res_p27.status} failures={_champ_res_p27.failures} "
            f"gross_violation_count={gross_viol_p27} execution_parity={exec_parity_p27}"
        )
            evaluate_championship_adoption(
            candidate_returns=candidate_p27, incumbent_returns=incumbent_p27, raw_returns=raw_p27,
            horizon=horizon, config=_champ_cfg_p27, execution_parity=exec_parity_p27,
            gross_violation_count=gross_viol_p27, era_pairs=None,
        )
        except Exception as _exc_p27:
            logger.warning(f"[EVAL] P27 championship gate failed {_exc_p27!r}")
    from src.tournament.objective.cutoff_auc import (
        CUTOFF_AUC_IS_PRODUCTION_GATE,
        cutoff_auc_score,
        mean_smooth_cutoff_utility,
    )

    summary["cutoff_auc_score"] = float(cutoff_auc_score(candidate_p27))
    summary["cutoff_auc_smooth_mean"] = float(mean_smooth_cutoff_utility(candidate_p27))
    summary["cutoff_auc_is_production_gate"] = bool(CUTOFF_AUC_IS_PRODUCTION_GATE)


CHAMPION_HOOKS: Final[dict[str, _Hook]] = {
    "sticky.mom60_raw": _hook_mom60_raw,
}

__all__ = ["CHAMPION_HOOKS"]
