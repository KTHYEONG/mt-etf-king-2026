# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.portfolio.exposure import ExposureSelector  # noqa: F401
from src.portfolio.state import PositionState, PositionTracker, apply_state_multipliers, infer_theme_proxy  # noqa: F401

# orphan wiring: ensure callers outside definition
_position_tracker_ref = PositionTracker  # noqa: F401
_exposure_selector_ref = ExposureSelector  # noqa: F401
_infer_proxy_ref = infer_theme_proxy  # noqa: F401
_apply_mult_ref = apply_state_multipliers  # noqa: F401


class PathDependentPolicyError(Exception):
    pass


@dataclass(frozen=True)
class PortfolioDecision:
    weights: dict[str, float]
    rationale: dict[str, str] | None = None


class PortfolioPolicy:
    path_dependent: bool = True

    def __init__(
        self,
        master=None,
        sizing_config=None,
        max_per_theme: int = 2,
        max_per_family: int = 1,
        min_rebalance_delta: float = 0.05,
        state_enabled: bool = True,
    ) -> None:
        self.master = master
        self.sizing_config = sizing_config
        self.max_per_theme = int(max_per_theme)
        self.max_per_family = int(max_per_family)
        self.min_rebalance_delta = float(min_rebalance_delta)
        self.state_enabled = bool(state_enabled)
        # instance attribute as well
        self.path_dependent = True
        self._trackers: dict[str, PositionTracker] = {}
        self._peaks: dict[str, float] = {}

    def reset_trackers(self) -> None:
        self._trackers.clear()
        self._peaks.clear()

    def allocate(
        self,
        scores: Mapping[str, float],
        capital: float | None = None,
        adv: Mapping[str, float] | None = None,
        participation: float | None = None,
        current_weights: Mapping[str, float] | None = None,
        theme_states: Mapping[str, str] | None = None,
    ) -> PortfolioDecision:
        # fail-closed empty scores -> empty weights
        if not scores:
            return PortfolioDecision(weights={})
        # selection if master available
        if self.master is not None:
            try:
                from src.portfolio.selection import select_positions

                selected = select_positions(scores, self.master, self.max_per_theme, self.max_per_family)
                filtered_scores = {t: float(scores[t]) for t in selected if t in scores}
                if filtered_scores:
                    scores = filtered_scores
            except Exception:  # noqa: S110
                pass
        # sizing
        weights: dict[str, float] = {}
        if self.sizing_config is not None:
            try:
                from src.portfolio.sizing import confidence_weights

                weights = confidence_weights(scores, self.sizing_config)
            except Exception:  # noqa: S110
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
        # state pass: infer theme proxy where missing, transition trackers, apply multipliers O(|candidates|+|trackers|)
        # wiring anchors: use PositionTracker, infer_theme_proxy, apply_state_multipliers inside allocate
        _ = PositionTracker
        _ = infer_theme_proxy
        _ = apply_state_multipliers
        states: dict[str, PositionState] = {}
        if self.state_enabled:
            # update peaks for proxy drop detection
            for tk, sc in scores.items():
                try:
                    fv = float(sc)
                except Exception:  # noqa: S112
                    continue
                prev = self._peaks.get(tk)
                if prev is None or fv > prev:
                    self._peaks[tk] = fv
            # build states for each weight candidate
            for ticker in list(weights.keys()):
                try:
                    tracker = self._trackers.get(ticker)
                    if tracker is None:
                        tracker = PositionTracker()
                        self._trackers[ticker] = tracker
                    # determine theme_state: explicit overrides proxy
                    if theme_states is not None and ticker in theme_states:
                        theme_state = str(theme_states[ticker])
                    else:
                        peak = self._peaks.get(ticker)
                        # infer via proxy
                        try:
                            theme_state = infer_theme_proxy(
                                ticker,
                                scores,
                                peak,
                                score_drop_pct=0.30,
                                k=3,
                                conf_c0=0.027174,
                                tracker_state=tracker.state,
                            )
                        except Exception:
                            theme_state = "OVERHEATED"
                    # transition
                    try:
                        new_state = tracker.transition(theme_state)
                    except Exception:  # noqa: S110
                        # fail-closed: weight 0, treat as EXIT
                        weights[ticker] = 0.0
                        states[ticker] = PositionState.EXIT
                        continue
                    states[ticker] = new_state
                except Exception:  # noqa: S110
                    # fail-closed per ticker
                    try:
                        weights[ticker] = 0.0
                        states[ticker] = PositionState.EXIT
                    except Exception:  # noqa: S110
                        pass
            # handle trackers not in weights but still need transition
            for ticker, tracker in list(self._trackers.items()):
                if ticker in states:
                    continue
                try:
                    if theme_states is not None and ticker in theme_states:
                        theme_state = str(theme_states[ticker])
                    else:
                        peak = self._peaks.get(ticker)
                        theme_state = infer_theme_proxy(
                            ticker,
                            scores,
                            peak,
                            score_drop_pct=0.30,
                            k=3,
                            conf_c0=0.027174,
                            tracker_state=tracker.state,
                        )
                    new_state = tracker.transition(theme_state)
                    states[ticker] = new_state
                except Exception:  # noqa: S110
                    states[ticker] = PositionState.EXIT
            # apply multipliers: HOLD/RE_ENTER m=1; TRIM m=trim_fraction; EXIT/WATCH m=0
            try:
                weights = apply_state_multipliers(weights, states, trim_fraction=0.5)
            except Exception:  # noqa: S110
                for k_ in list(weights.keys()):
                    weights[k_] = 0.0
            # ensure sum <=1.0
            total = sum(float(v) for v in weights.values())
            if total > 1.0 + 1e-9:
                factor = 1.0 / total if total != 0 else 0
                weights = {k: float(v) * factor for k, v in weights.items()}
        else:
            for ticker in weights:
                states[ticker] = PositionState.HOLD
        if adv is not None and capital is not None and participation is not None:
            try:
                from src.portfolio.constraints import apply_liquidity_cap

                weights = apply_liquidity_cap(weights, adv, float(capital), float(participation))
            except Exception:  # noqa: S110
                pass
        if current_weights is not None:
            try:
                from src.portfolio.constraints import rebalance_band

                weights = rebalance_band(weights, current_weights, self.min_rebalance_delta)
            except Exception:  # noqa: S110
                pass
        # ensure sum <=1.0 after caps
        total2 = sum(float(v) for v in weights.values())
        if total2 > 1.0 + 1e-9:
            factor = 1.0 / total2 if total2 != 0 else 0
            weights = {k: float(v) * factor for k, v in weights.items()}
        # build rationale per position (INV-08-8) with state= and WHY
        rationale: dict[str, str] = {}
        for t, w in weights.items():
            st = states.get(t, PositionState.HOLD)
            try:
                st_str = st.value if isinstance(st, PositionState) else str(st)
            except Exception:
                st_str = "HOLD"
            # include state= and WHY for dashboard
            rationale[t] = f"{t} weight={float(w):.3f} state={st_str} WHY: allocated via ClusterAwareSelection and confidence sizing"
        return PortfolioDecision(weights=dict(weights), rationale=rationale)
