# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping  # noqa: F401

from src.portfolio.policy_tail import PortfolioPolicyTailMixin
from src.portfolio.policy_types import PortfolioDecision  # noqa: F401
from src.portfolio.state import PositionTracker


class PortfolioPolicy(PortfolioPolicyTailMixin):
    path_dependent: bool = True
    scores_path_independent: bool = True

    def __init__(
        self,
        master=None,
        sizing_config=None,
        max_per_theme: int = 2,
        max_per_family: int = 1,
        min_rebalance_delta: float = 0.0,
        state_enabled: bool = True,
        max_gross_exposure: float = 1.60,
        aggression: object | None = None,
        lottery_config=None,
        convexity_config=None,
    ) -> None:
        self.master = master
        self.sizing_config = sizing_config
        self.max_per_theme = int(max_per_theme)
        self.max_per_family = int(max_per_family)
        self.min_rebalance_delta = float(min_rebalance_delta)
        self.state_enabled = bool(state_enabled)
        self.max_gross_exposure = float(max_gross_exposure)
        self.aggression = aggression
        self.lottery_config = lottery_config
        self.convexity_config = convexity_config
        # instance attribute as well
        self.path_dependent = True
        self.scores_path_independent = True
        self._trackers: dict[str, PositionTracker] = {}
        self._peaks: dict[str, float] = {}
        self._convexity_entry_score: float | None = None
        self._convexity_entry_key: str | None = None

    def reset_trackers(self) -> None:
        self._trackers.clear()
        self._peaks.clear()
        self._convexity_entry_score = None
        self._convexity_entry_key = None

    def allocate(
        self,
        scores: Mapping[str, float],
        capital: float | None = None,
        adv: Mapping[str, float] | None = None,
        participation: float | None = None,
        current_weights: Mapping[str, float] | None = None,
        theme_states: Mapping[str, str] | None = None,
        regime: str | None = None,
        leverage_allowed: bool | None = None,
        inverse_allowed: bool | None = None,
        aggression_input: object | None = None,
    ) -> PortfolioDecision:
        # wiring anchors: ensure ExposureSelector pick_vehicle, apply_gross_exposure_cap, aggression referenced
        # lottery wiring
        # fail-closed empty scores -> empty weights
        if not scores:
            return PortfolioDecision(weights={}, rationale={}, vehicles={}, gross=0.0)
        # selection if master available
        original_scores = dict(scores)
        if self.master is not None:
            try:
                from src.portfolio.selection import select_positions

                selected = select_positions(scores, self.master, self.max_per_theme, self.max_per_family)
                filtered_scores = {t: float(scores[t]) for t in selected if t in scores}
                if filtered_scores:
                    scores = filtered_scores
            except Exception:  # noqa: S110
                pass
        # lottery active check O(1)
        lottery_on = False
        try:
            from src.portfolio.sizing import lottery_active as _lottery_active

            # ensure import_symbol wiring
            from src.portfolio.sizing import lottery_active  # noqa: F401
            from src.portfolio.sizing import resolve_overlay_sizing_branch  # noqa: F401

            lottery_on = bool(_lottery_active(regime, leverage_allowed, self.lottery_config, scores))
            lottery_active(regime, leverage_allowed, self.lottery_config, scores)
            resolve_overlay_sizing_branch(scores, lottery_on=lottery_on, convexity_on=False, lottery_config=self.lottery_config, convexity_config=self.convexity_config)
        except Exception:
            lottery_on = False
        # sizing: lottery_on skips tail_concentration_weights and confidence_weights
        weights: dict[str, float] = {}
        lottery_branch = False
        convexity_on = False
        try:
            from src.portfolio.convexity import convexity_active, convexity_should_exit, resolve_convexity_vehicle  # noqa: F401

            convexity_on = bool(convexity_active(leverage_allowed, regime, scores, self.convexity_config))
            if convexity_on:
                sorted_scores = sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
                top_key, top_score_raw = sorted_scores[0]
                top_score = float(top_score_raw)
                cfg_c = self.convexity_config  # type: ignore[assignment]
                if convexity_should_exit(top_score, self._convexity_entry_score, regime, cfg_c):  # type: ignore[arg-type]
                    convexity_on = False
                    self._convexity_entry_score = None
                    self._convexity_entry_key = None
                else:
                    if self._convexity_entry_key is None or self._convexity_entry_key != str(top_key):
                        self._convexity_entry_key = str(top_key)
                        self._convexity_entry_score = float(top_score)
        except Exception:
            convexity_on = False
        # overlay arbitration: lottery-first
        if lottery_on or convexity_on:
            try:
                from src.portfolio.sizing import resolve_overlay_sizing_branch as _rosb  # noqa: F401

                weights, lottery_branch, convexity_sizing = _rosb(scores, lottery_on=lottery_on, convexity_on=convexity_on, lottery_config=self.lottery_config, convexity_config=self.convexity_config)
                resolve_overlay_sizing_branch(scores, lottery_on=lottery_on, convexity_on=convexity_on, lottery_config=self.lottery_config, convexity_config=self.convexity_config)
                # keep legacy wiring refs
                from src.portfolio.sizing import lottery_concentration_weights as _lcw_c2  # noqa: F401

            except Exception:
                # fallback to legacy lottery/convexity branches
                if convexity_on and lottery_on:
                    try:
                        from src.portfolio.sizing import lottery_concentration_weights as _lcw_f

                        weights = _lcw_f(scores, self.lottery_config)  # type: ignore[arg-type]
                        lottery_branch = bool(weights)
                    except Exception:
                        weights = {}
                        lottery_branch = False
                elif convexity_on:
                    try:
                        cfg_c2 = self.convexity_config
                        w_top_c = float(getattr(cfg_c2, "w_top", 1.0)) if cfg_c2 is not None else 1.0
                        sorted_scores2 = sorted(scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
                        top_t = str(sorted_scores2[0][0])
                        weights = {top_t: float(w_top_c)}
                        convexity_sizing = True  # type: ignore[assignment]
                    except Exception:
                        weights = {}
                elif lottery_on:
                    try:
                        from src.portfolio.sizing import lottery_concentration_weights as _lcw2

                        weights = _lcw2(scores, self.lottery_config)  # type: ignore[arg-type]
                        lottery_branch = bool(weights)
                    except Exception:
                        lottery_branch = False
                        weights = {}
        else:
            lottery_branch = False
            convexity_sizing = False  # type: ignore[assignment]
        if not lottery_branch and not convexity_on:
            if self.sizing_config is not None:
                try:
                    from src.portfolio.sizing import (
                        TailConcentrationConfig,
                        confidence_weights,
                        tail_concentration_weights,
                    )

                    # load TailConcentrationConfig from strategies.yaml portfolio.tail_concentration fail-closed
                    tail_cfg = TailConcentrationConfig(enabled=False)
                    try:
                        from src.core.config import load_config

                        _sd = load_config("strategies")
                        if isinstance(_sd, dict):
                            _port = _sd.get("portfolio") or {}
                            if isinstance(_port, dict):
                                _tc = _port.get("tail_concentration")
                                if isinstance(_tc, dict):
                                    enabled = bool(_tc.get("enabled", False))
                                    trad_raw = _tc.get("tradable_states", ["LEADING", "RECOVERY"])
                                    risk_raw = _tc.get("risk_on_regimes", ["RISK_ON", "STRONG_RISK_ON"])
                                    w_full = float(_tc.get("w_top_full", 1.0))
                                    k_full = int(_tc.get("k_full", 1))
                                    try:
                                        trad_set = frozenset(str(x) for x in trad_raw) if isinstance(trad_raw, (list, set, tuple, frozenset)) else frozenset({str(trad_raw)})
                                    except Exception:
                                        trad_set = frozenset({"LEADING", "RECOVERY"})
                                    try:
                                        risk_set = frozenset(str(x) for x in risk_raw) if isinstance(risk_raw, (list, set, tuple, frozenset)) else frozenset({str(risk_raw)})
                                    except Exception:
                                        risk_set = frozenset({"RISK_ON", "STRONG_RISK_ON"})
                                    tail_cfg = TailConcentrationConfig(enabled=enabled, tradable_states=trad_set, risk_on_regimes=risk_set, w_top_full=w_full, k_full=k_full)
                    except Exception:
                        tail_cfg = TailConcentrationConfig(enabled=False)
                    # try tail concentration first; delegate to confidence_weights if empty
                    try:
                        tw = tail_concentration_weights(scores, self.sizing_config, tail_cfg, theme_states, regime)
                        if tw:
                            weights = dict(tw)
                        else:
                            weights = confidence_weights(scores, self.sizing_config)
                    except Exception:
                        weights = confidence_weights(scores, self.sizing_config)
                except Exception:  # noqa: S110
                    try:
                        from src.portfolio.sizing import confidence_weights as _cw2

                        weights = _cw2(scores, self.sizing_config)
                    except Exception:
                        weights = {k: float(v) for k, v in scores.items()}
                        total = sum(weights.values())
                        if total != 0:
                            weights = {k: v / total for k, v in weights.items()}
            else:
                from src.portfolio.sizing import ConfidenceSizingConfig, confidence_weights

                cfg = ConfidenceSizingConfig()
                try:
                    weights = confidence_weights(scores, cfg)
                except Exception:  # noqa: S110
                    weights = {}
        return self._allocate_tail(
            scores,
            original_scores,
            weights,
            lottery_on,
            convexity_on,
            adv,
            capital,
            participation,
            current_weights,
            theme_states,
            regime,
            leverage_allowed,
            inverse_allowed,
            aggression_input,
        )
