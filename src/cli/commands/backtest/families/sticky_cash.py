"""Sticky cash-family forensics hooks: mom60_hold, mom60_abs_cash (P4)."""

from __future__ import annotations

import contextlib
import logging
from typing import Final

from src.cli.commands.backtest._core import ForensicsHook as _Hook
from src.cli.commands.backtest._core import _Cell
from src.core.config import config_path

logger = logging.getLogger(__name__)


def _hook_mom60_hold(cell: _Cell) -> None:
    from src.portfolio.constraints import resolve_exposure_limits_for_model as _resolve_exp_p28a
    from src.strategies.registry import STRATEGIES as _BL27_P28A
    from src.tournament.eval_mode import resolve_eval_flags as _ref_p28a
    from src.tournament.objective import ChampionshipObjectiveConfig as _COC_p28a
    from src.tournament.objective import evaluate_championship_adoption as _eval_champ_p28a
    from src.tournament.objective import field_relative_report as _field_rel_p28a
    from src.tournament.simulator import RollingDiagnostics

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
    _path_mode = cell.path_mode
    _shared_cache = cell.shared_cache
    candidate = list(rolling.returns)
    raw = list(rolling.returns)
    # champion P27
    champion_p27 = []
    try:
        p27_model = _BL27_P28A["sticky.mom60_raw"]()
        _p27_flags = _ref_p28a(p27_model, eval_mode)
        _p27_limits = _resolve_exp_p28a("sticky.mom60_raw", comparison_mode="full_strategy_own")
        champion_roll = simulator.run_rolling(
            p27_model,
            panel,
            case_config,
            horizon=horizon,
            path_dependent=_p27_flags.path_dependent,
            path_dependent_mode=_path_mode,
            session_cache=_shared_cache,
            leverage_allowed=_lev_allowed_resolved,
            inverse_allowed=_inv_allowed_resolved,
            close_map=close_map,
            exposure_limits=_p27_limits,
        )
        champion_p27 = list(champion_roll.returns)
    except Exception:
        champion_p27 = []
    incumbent_returns = champion_p27
    # P21 historical anchor for field_relative
    p21_returns = []
    try:
        from src.strategies.registry import STRATEGIES as _BL21_P28A

        p21_m = _BL21_P28A["sticky.impulse_crash"]()
        from src.tournament.eval_mode import resolve_eval_flags as _ref_p21_p28a

        _p21_flags_p28a = _ref_p21_p28a(p21_m, eval_mode)
        from src.portfolio.constraints import resolve_exposure_limits_for_model as _resolve_exp_p21_p28a

        _p21_limits = _resolve_exp_p21_p28a("sticky.impulse_crash", comparison_mode="alpha_equal")
        p21_roll = simulator.run_rolling(
            p21_m,
            panel,
            case_config,
            horizon=horizon,
            path_dependent=_p21_flags_p28a.path_dependent,
            path_dependent_mode=_path_mode,
            session_cache=_shared_cache,
            leverage_allowed=_lev_allowed_resolved,
            inverse_allowed=_inv_allowed_resolved,
            close_map=close_map,
            exposure_limits=_p21_limits,
        )
        p21_returns = list(p21_roll.returns)
    except Exception:
        p21_returns = []
    with contextlib.suppress(Exception):
        _fr_p28a = _field_rel_p28a(
            candidate, {"sticky.impulse_crash": p21_returns, "sticky.mom60_raw": champion_p27}, horizon=horizon
        ) if p21_returns and champion_p27 and len(candidate) == len(champion_p27) else None
        if p21_returns and champion_p27:
            _field_rel_p28a(
            candidate, {"sticky.impulse_crash": p21_returns, "sticky.mom60_raw": champion_p27}, horizon=horizon
            )
        if _fr_p28a is not None:
            summary["field_relative"] = {"win_rate": float(_fr_p28a.win_rate), "top2_rate": float(_fr_p28a.top2_rate)}
    if champion_p27:
        _field_rel_p28a(candidate, {"sticky.impulse_crash": p21_returns, "sticky.mom60_raw": champion_p27}, horizon=horizon)
    # gross diagnostics
    gross_violation_count = None
    effective_gross_max = None
    try:
        _diag_p28a = getattr(rolling, "diagnostics", None)
        RollingDiagnostics(gross_violation_count=gross_violation_count, effective_gross_max=effective_gross_max, turnover_mean=None, fill_count=None, unfilled_count=None)
        if _diag_p28a is not None:
            gross_violation_count = getattr(_diag_p28a, "gross_violation_count", None)
            effective_gross_max = getattr(_diag_p28a, "effective_gross_max", None)
            if gross_violation_count is not None:
                summary["gross_violation_count"] = int(gross_violation_count)
    except Exception:
        gross_violation_count = None
    # championship evaluation
    try:
        _champ_cfg_p28a = _COC_p28a.from_yaml(config_path("gates"), config_path("portfolio"))
    except Exception:
        _champ_cfg_p28a = None
    if _champ_cfg_p28a is not None:
        exec_parity_p28a = bool(candidate and incumbent_returns and raw and len(candidate) == len(incumbent_returns))
        with contextlib.suppress(Exception):
            _champ_res_p28a = _eval_champ_p28a(
            candidate_returns=candidate, incumbent_returns=incumbent_returns if incumbent_returns else candidate,
            raw_returns=raw, horizon=horizon, config=_champ_cfg_p28a, execution_parity=exec_parity_p28a,
            gross_violation_count=gross_violation_count, era_pairs=None,
        )
            summary["championship_gate_status"] = str(_champ_res_p28a.status)
            summary["championship_gate_failures"] = list(_champ_res_p28a.failures)
            summary["adoption_gate_status"] = str(_champ_res_p28a.status)
            summary["adoption_gate_fails"] = list(_champ_res_p28a.failures)
            _eval_champ_p28a(
            candidate_returns=candidate, incumbent_returns=champion_p27, raw_returns=raw,
            horizon=horizon, config=_champ_cfg_p28a, execution_parity=exec_parity_p28a,
            gross_violation_count=gross_violation_count, era_pairs=None,
        )


def _hook_mom60_abs_cash(cell: _Cell) -> None:
    from src.portfolio.constraints import load_p27_exposure_limits as _load_p27_exp_p28b_bt
    from src.portfolio.constraints import resolve_exposure_limits_for_model as _resolve_exp_p28b
    from src.strategies.registry import STRATEGIES as _BL27_P28B
    from src.tournament.eval_mode import resolve_eval_flags as _ref_p28b
    from src.tournament.objective import ChampionshipObjectiveConfig as _COC_p28b
    from src.tournament.objective import evaluate_championship_adoption as _eval_champ_p28b
    from src.tournament.objective import field_relative_report as _field_rel_p28b
    from src.tournament.simulator import RollingDiagnostics

    summary = cell.summary
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
    _path_mode = cell.path_mode
    _shared_cache = cell.shared_cache
    with contextlib.suppress(Exception):
        engine.set_portfolio_exposure_limits(_load_p27_exp_p28b_bt())
    candidate = list(rolling.returns)
    raw = list(rolling.returns)
    champion_p27 = []
    try:
        p27_model = _BL27_P28B["sticky.mom60_raw"]()
        _p27_flags = _ref_p28b(p27_model, eval_mode)
        _p27_limits = _resolve_exp_p28b("sticky.mom60_raw", comparison_mode="full_strategy_own")
        champion_roll = simulator.run_rolling(
            p27_model,
            panel,
            case_config,
            horizon=horizon,
            path_dependent=_p27_flags.path_dependent,
            path_dependent_mode=_path_mode,
            session_cache=_shared_cache,
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
        from src.strategies.registry import STRATEGIES as _BL21_P28B

        p21_m = _BL21_P28B["sticky.impulse_crash"]()
        from src.tournament.eval_mode import resolve_eval_flags as _ref_p21_p28b

        _p21_flags_p28b = _ref_p21_p28b(p21_m, eval_mode)
        from src.portfolio.constraints import resolve_exposure_limits_for_model as _resolve_exp_p21_p28b

        _p21_limits = _resolve_exp_p21_p28b("sticky.impulse_crash", comparison_mode="alpha_equal")
        p21_roll = simulator.run_rolling(
            p21_m,
            panel,
            case_config,
            horizon=horizon,
            path_dependent=_p21_flags_p28b.path_dependent,
            path_dependent_mode=_path_mode,
            session_cache=_shared_cache,
            leverage_allowed=_lev_allowed_resolved,
            inverse_allowed=_inv_allowed_resolved,
            close_map=close_map,
            exposure_limits=_p21_limits,
        )
        p21_returns = list(p21_roll.returns)
    except Exception:
        p21_returns = []
    with contextlib.suppress(Exception):
        _fr_p28b = _field_rel_p28b(
            candidate, {"sticky.impulse_crash": p21_returns, "sticky.mom60_raw": champion_p27}, horizon=horizon
        ) if p21_returns and champion_p27 and len(candidate) == len(champion_p27) else None
        if p21_returns and champion_p27:
            _field_rel_p28b(
            candidate, {"sticky.impulse_crash": p21_returns, "sticky.mom60_raw": champion_p27}, horizon=horizon
            )
        if _fr_p28b is not None:
            summary["field_relative"] = {"win_rate": float(_fr_p28b.win_rate), "top2_rate": float(_fr_p28b.top2_rate)}
    if champion_p27:
        _field_rel_p28b(candidate, {"sticky.impulse_crash": p21_returns, "sticky.mom60_raw": champion_p27}, horizon=horizon)
    gross_violation_count = None
    effective_gross_max = None
    try:
        _diag_p28b = getattr(rolling, "diagnostics", None)
        RollingDiagnostics(gross_violation_count=gross_violation_count, effective_gross_max=effective_gross_max, turnover_mean=None, fill_count=None, unfilled_count=None)
        if _diag_p28b is not None:
            gross_violation_count = getattr(_diag_p28b, "gross_violation_count", None)
            effective_gross_max = getattr(_diag_p28b, "effective_gross_max", None)
            if gross_violation_count is not None:
                summary["gross_violation_count"] = int(gross_violation_count)
    except Exception:
        gross_violation_count = None
    try:
        _champ_cfg_p28b = _COC_p28b.from_yaml(config_path("gates"), config_path("portfolio"))
    except Exception:
        _champ_cfg_p28b = None
    if _champ_cfg_p28b is not None:
        exec_parity_p28b = bool(candidate and incumbent_returns and raw and len(candidate) == len(incumbent_returns))
        with contextlib.suppress(Exception):
            _champ_res_p28b = _eval_champ_p28b(
            candidate_returns=candidate, incumbent_returns=incumbent_returns if incumbent_returns else candidate,
            raw_returns=raw, horizon=horizon, config=_champ_cfg_p28b, execution_parity=exec_parity_p28b,
            gross_violation_count=gross_violation_count, era_pairs=None,
        )
            summary["championship_gate_status"] = str(_champ_res_p28b.status)
            summary["championship_gate_failures"] = list(_champ_res_p28b.failures)
            summary["adoption_gate_status"] = str(_champ_res_p28b.status)
            summary["adoption_gate_fails"] = list(_champ_res_p28b.failures)
            _eval_champ_p28b(
            candidate_returns=candidate, incumbent_returns=champion_p27, raw_returns=raw,
            horizon=horizon, config=_champ_cfg_p28b, execution_parity=exec_parity_p28b,
            gross_violation_count=gross_violation_count, era_pairs=None,
        )


CASH_HOOKS: Final[dict[str, _Hook]] = {
    "sticky.mom60_hold": _hook_mom60_hold,
    "sticky.mom60_abs_cash": _hook_mom60_abs_cash,
}

__all__ = ["CASH_HOOKS"]
