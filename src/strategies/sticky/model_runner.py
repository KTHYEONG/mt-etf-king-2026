# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

import polars as pl

from src.alpha.base import DecisionContext
from src.strategies.sticky.capacity import apply_capacity_filter, cached_filtered_scores, resolve_capacity_capital
from src.universe.instruments import resolve_leverage

from src.strategies.sticky.model_config import (
    DEFAULT_EXCLUDE_NAME_TOKENS,
    StickyLeaderConfig,
    blend_rank_scores,
    cross_section_percentile_ranks,
    logger,
    momentum_horizon,
    name_excluded,
)
from src.strategies.sticky.model_scores import apply_sticky_leader, collapse_plus2_by_family, filter_plus2_scores


class StickyLeaderModel:
    name: str
    config: StickyLeaderConfig
    path_dependent: bool = True
    scores_path_independent: bool = False
    def __init__(self, name: str = "sticky.leader_base", config: StickyLeaderConfig | None = None) -> None:
        self.name = str(name)
        self.config = config if config is not None else StickyLeaderConfig()
        self._held: str | None = None
        self._hold_len: int = 0
        self._runner_ticker: str | None = None
        self._runner_entry_capital: float | None = None
        self._runner_peak_capital: float | None = None
        self._runner_held_sessions: int = 0
        self._runner_armed: bool = False
        self._filtered_scores_by_snapshot: dict[date, dict[str, float]] = {}
    def reset_trackers(self) -> None:
        self._held = None
        self._hold_len = 0
        self._runner_ticker = None
        self._runner_entry_capital = None
        self._runner_peak_capital = None
        self._runner_held_sessions = 0
        self._runner_armed = False
        self._filtered_scores_by_snapshot = {}
    def restore_state(self, held: str | None, hold_len: int) -> None:
        import math as _math
        if held is not None and not isinstance(held, str):
            raise ValueError("held must be str or None")
        try:
            hl = int(hold_len)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"hold_len invalid: {exc}") from exc
        # check finite via float
        try:
            fv = float(hold_len)  # type: ignore[arg-type]
            if not _math.isfinite(fv):
                raise ValueError("hold_len must be finite")
        except Exception as exc:
            raise ValueError(f"hold_len invalid: {exc}") from exc
        if hl < 0:
            raise ValueError("hold_len must be >=0")
        self._held = held
        self._hold_len = int(hl)
    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float] | object:
        decision_date = getattr(context, "decision_date", None)
        if decision_date is None or not isinstance(decision_date, date):
            raise ValueError(f"score requires a valid decision_date key, got {decision_date!r}")
        filtered = cached_filtered_scores(self._filtered_scores_by_snapshot, context.decision_date, snapshot, lambda frame: filter_plus2_scores(frame, self.config))
        try:
            _mfr = float(getattr(self.config, "min_fill_ratio", 0.0) or 0.0)
        except Exception:
            _mfr = 0.0
        if math.isfinite(_mfr) and _mfr > 0 and filtered:
            _cap = resolve_capacity_capital(context)
            try:
                _rules = getattr(context, "rules", None)
                _phi = float(getattr(_rules, "max_order_to_adv", 0.01))
            except Exception:
                _phi = 0.01
            if not math.isfinite(_phi) or _phi <= 0:
                _phi = 0.01
            filtered = apply_capacity_filter(
                filtered, snapshot, capital=_cap, max_order_to_adv=_phi, min_fill_ratio=_mfr
            )
            if not filtered:
                from src.portfolio.intent import CASH_INTENT as _CASH_CAP
                return _CASH_CAP
        try:
            _aux_col = getattr(self.config, "score_aux_col", None)
            _aux_w = float(getattr(self.config, "score_aux_weight", 0.0) or 0.0)
        except Exception:
            _aux_col, _aux_w = None, 0.0
        if isinstance(_aux_col, str) and _aux_col and math.isfinite(_aux_w) and _aux_w > 0:
            if _aux_col not in snapshot.columns:
                filtered = cross_section_percentile_ranks(filtered)
            else:
                _aux_raw = {}
                try:
                    rows = snapshot.iter_rows(named=True)
                except Exception:
                    return {}
                for _row in rows:
                    try:
                        _t = str(_row.get("ticker"))
                        _af = float(_row.get(_aux_col))  # type: ignore[arg-type]
                        if _t in filtered and math.isfinite(_af):
                            _aux_raw[_t] = float(_af)
                    except Exception:
                        continue
                filtered = blend_rank_scores(filtered, _aux_raw, w_primary=1.0 - _aux_w, w_aux=_aux_w)
            if not filtered:
                from src.portfolio.intent import CASH_INTENT as _CASH_AUX
                return _CASH_AUX
        if getattr(self.config, "collapse_family", False):
            try:
                filtered = collapse_plus2_by_family(filtered, snapshot)
            except Exception:
                pass
        # derive held from context.held by (-weight, ticker)
        held: str | None = None
        try:
            held_map = getattr(context, "held", {})
            if isinstance(held_map, Mapping) and len(held_map) > 0:
                # filter to numeric weights?
                items = []
                for k, v in held_map.items():
                    try:
                        items.append((str(k), float(v)))
                    except Exception:
                        continue
                if items:
                    items_sorted = sorted(items, key=lambda kv: (-kv[1], kv[0]))
                    held = str(items_sorted[0][0])
        except Exception:
            held = None
        # update internal hold_len
        if held != self._held:
            self._held = held
            self._hold_len = 1 if held else 0
        elif held is not None:
            self._hold_len += 1
        # else held is None and _held is None -> keep 0
        # P33 confirmed-runner-reversal tracker (model-local, reversible cash exit only)
        _runner_exit = bool(getattr(self.config, "runner_reversal_exit", False))
        _runner_cap: float | None = None
        _runner_mom: float | None = None
        _runner_mc = "mom_5"
        if _runner_exit:
            try:
                _c = float(getattr(context, "capital", float("nan")))
                _runner_cap = float(_c) if math.isfinite(_c) and _c > 0 else None
            except Exception: _runner_cap = None
            try: _runner_mc = str(getattr(self.config, "runner_mom_col", "mom_5"))
            except Exception: _runner_mc = "mom_5"
            try: _hz = int(momentum_horizon(_runner_mc))
            except Exception: _hz = 5
            if held is None or _runner_cap is None or _hz <= 0:
                if held is None:
                    self._runner_ticker = None
                    self._runner_entry_capital = None
                    self._runner_peak_capital = None
                    self._runner_held_sessions = 0
                self._runner_armed = False
            elif getattr(self, "_runner_ticker", None) != held:
                self._runner_ticker = str(held)
                self._runner_entry_capital = float(_runner_cap)
                self._runner_peak_capital = float(_runner_cap)
                self._runner_held_sessions = 1
                self._runner_armed = False
            else:
                self._runner_held_sessions = int(getattr(self, "_runner_held_sessions", 0) or 0) + 1
                try:
                    _pk = float(getattr(self, "_runner_peak_capital", _runner_cap))
                    _pk = float(_runner_cap) if not math.isfinite(_pk) else max(float(_pk), float(_runner_cap))
                    self._runner_peak_capital = float(_pk)
                    _en = float(getattr(self, "_runner_entry_capital", float("nan")))
                    _pn = float(getattr(self, "_runner_peak_capital", float("nan")))
                    self._runner_armed = bool(math.isfinite(_en) and math.isfinite(_pn) and int(getattr(self, "_runner_held_sessions", 0)) >= _hz and _pn > _en)
                except Exception: self._runner_armed = False
            if held is not None:
                try:
                    if isinstance(snapshot, pl.DataFrame) and _runner_mc in snapshot.columns and "ticker" in snapshot.columns:
                        _df = snapshot.filter(pl.col("ticker") == str(held)).head(1)
                        _runner_mom = (
                            lambda _v: (lambda _f: float(_f) if math.isfinite(float(_f)) else None)(
                                float(_v) if _v is not None else float("nan")
                            )
                        )(_df.row(0, named=True).get(_runner_mc)) if _df.height > 0 else None
                except Exception: _runner_mom = None
        if getattr(self.config, "collapse_family", False):
            try:
                _ns = len(filtered)
                _tt = sorted(filtered.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))[0][0] if filtered else ""
                logger.debug(f"[ALGO] ticker={_tt} held={held} n_scores={_ns}")
            except Exception: pass
        sticky = apply_sticky_leader(filtered, held, self.config, self._hold_len)
        from src.strategies.sticky.overlays import (
            apply_abs_mom_cash,
            apply_crash_cash,
            apply_impulse_switch,
            apply_same_leader_hold,
        )
        from src.tournament.championship_regime import abs_mom_rebound_bypass_allowed

        impulsed = apply_impulse_switch(sticky, held, snapshot, self.config)
        crashed = apply_crash_cash(impulsed, held, snapshot, self.config)
        _sleeve_raw = getattr(context, "championship_sleeve", None)
        _sleeve = _sleeve_raw if isinstance(_sleeve_raw, str) else None
        _bypass = abs_mom_rebound_bypass_allowed(sleeve=_sleeve, config_enabled=bool(getattr(self.config, "crash_rebound_abs_mom_bypass", False)))
        abs_gated = apply_abs_mom_cash(crashed, self.config, held=held, rebound_bypass=_bypass)
        out = apply_same_leader_hold(abs_gated, held, bool(getattr(self.config, "same_leader_hold", False)))
        from src.strategies.sticky.model_scores import rebound_leader_scores
        from src.tournament.objective.cutoff_auc import CUTOFF_AUC_IS_PRODUCTION_GATE
        from src.tournament.objective.cutoff_auc import apply_attack_sleeve_route

        _rebound_scores = rebound_leader_scores(snapshot)
        out = apply_attack_sleeve_route(sleeve=_sleeve, mom60_scores=out, rebound_scores=_rebound_scores, production_gate=CUTOFF_AUC_IS_PRODUCTION_GATE)
        if _runner_exit and held is not None and _runner_cap is not None and _runner_mom is not None:
            try:
                _pf = float(getattr(self, "_runner_peak_capital", float("nan")))
                _ef = float(getattr(self, "_runner_entry_capital", float("nan")))
                if bool(getattr(self, "_runner_armed", False)) and math.isfinite(_pf) and math.isfinite(_ef) and _pf > _ef and float(_runner_cap) < _pf and float(_runner_mom) <= 0:
                    from src.portfolio.intent import CASH_INTENT as _CASH_RUNNER
                    return _CASH_RUNNER
            except Exception: pass
        try:
            from collections.abc import Mapping as _Mapping
            from src.portfolio.intent import CASH_INTENT as _CASH_EMPTY
            if isinstance(out, _Mapping) and len(out) == 0:
                return _CASH_EMPTY
        except Exception:
            pass
        return out
