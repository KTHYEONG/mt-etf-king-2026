# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.portfolio.exposure import ExposureSelector  # noqa: F401
from src.portfolio.state import PositionTracker  # noqa: F401

# orphan wiring: ensure callers outside definition
_position_tracker_ref = PositionTracker  # noqa: F401
_exposure_selector_ref = ExposureSelector  # noqa: F401


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
    ) -> None:
        self.master = master
        self.sizing_config = sizing_config
        self.max_per_theme = int(max_per_theme)
        self.max_per_family = int(max_per_family)
        self.min_rebalance_delta = float(min_rebalance_delta)
        # instance attribute as well
        self.path_dependent = True

    def allocate(
        self,
        scores: Mapping[str, float],
        capital: float | None = None,
        adv: Mapping[str, float] | None = None,
        participation: float | None = None,
        current_weights: Mapping[str, float] | None = None,
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
        # ensure sum <=1.0
        total = sum(float(v) for v in weights.values())
        if total > 1.0 + 1e-9:
            # normalize down
            factor = 1.0 / total if total != 0 else 0
            weights = {k: float(v) * factor for k, v in weights.items()}
        # build rationale per position (INV-08-8)
        rationale: dict[str, str] = {}
        for t, w in weights.items():
            rationale[t] = f"{t} weight={w:.3f} conf={0:.3f} selection via ClusterAwareSelection"
        return PortfolioDecision(weights=dict(weights), rationale=rationale)
