# mypy: ignore-errors
# ruff: noqa: S101
"""Champion tail policy: top-family routing to fillable +2x/+1x vehicles."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from src.portfolio.intent import HOLD_INTENT, PortfolioIntent
from src.portfolio.policy import PortfolioDecision


@dataclass(frozen=True)
class ChampionPolicyConfig:
    max_single_weight: float = 0.80
    max_effective_gross: float = 1.60
    min_cash: float = 0.05
    absolute_momentum_cash: bool = True


class ChampionTailPolicy:
    name: str = "champion.tail_ranker"

    def __init__(self, master=None, config: ChampionPolicyConfig | None = None) -> None:
        self.master = master
        self.config = config or ChampionPolicyConfig()

    def score(self, snapshot: pl.DataFrame, context) -> dict[str, float] | PortfolioIntent:
        if snapshot.height == 0:
            return HOLD_INTENT
        if "source_ticker" not in snapshot.columns:
            return HOLD_INTENT
        scores: dict[str, float] = {}
        score_col = "score" if "score" in snapshot.columns else None
        if score_col is None:
            return HOLD_INTENT
        for row in snapshot.iter_rows(named=True):
            try:
                scores[str(row.get("source_ticker"))] = float(row.get(score_col))
            except Exception:
                return HOLD_INTENT
        if not scores:
            return HOLD_INTENT
        # Absolute-momentum cash predicate on the top source.
        if self.config.absolute_momentum_cash and "mom_60" in snapshot.columns:
            top = max(scores, key=lambda k: (scores[k], k))
            for row in snapshot.iter_rows(named=True):
                if str(row.get("source_ticker")) == top:
                    try:
                        if float(row.get("mom_60")) <= 0:
                            from src.portfolio.intent import CASH_INTENT

                            return CASH_INTENT
                    except Exception:
                        return HOLD_INTENT
        return scores

    def allocate(
        self,
        scores: Mapping[str, float],
        *,
        capital: float | None = None,
        adv: Mapping[str, float] | None = None,
        participation: float | None = None,
        regime: str | None = None,
        leverage_allowed: bool | None = None,
        inverse_allowed: bool | None = None,
        current_weights: Mapping[str, float] | None = None,
    ) -> PortfolioDecision:
        _ = inverse_allowed
        _ = current_weights
        if not scores:
            return PortfolioDecision(weights={}, rationale={}, vehicles={}, gross=0.0)
        # Only the top family is used.
        top_source = max(scores, key=lambda k: (float(scores[k]), str(k)))
        family = str(top_source)
        two_ticker: str | None = None
        one_ticker: str | None = None
        if self.master is not None:
            try:
                attr = self.master.attributes.get(top_source)
                if attr is not None:
                    family = str(getattr(attr, "leverage_family_key", top_source))
            except Exception:
                family = str(top_source)
            try:
                for t, a in self.master.attributes.items():
                    if str(getattr(a, "leverage_family_key", t)) != family:
                        continue
                    if bool(getattr(a, "is_synthetic", False)):
                        continue
                    try:
                        mult = int(getattr(a, "leverage_multiple", 1))
                    except Exception:
                        mult = 1
                    if mult == 2 and two_ticker is None:
                        two_ticker = str(t)
                    if mult == 1 and one_ticker is None:
                        one_ticker = str(t)
            except Exception:  # noqa: S110
                pass
        if one_ticker is None:
            one_ticker = str(top_source)
        cap = float(capital) if capital is not None else 1_000_000_000.0
        part = float(participation) if participation is not None else 0.01
        adv_map = dict(adv) if adv is not None else {}
        target = min(float(self.config.max_single_weight), 1.0 - float(self.config.min_cash))
        if target <= 0:
            return PortfolioDecision(weights={}, rationale={}, vehicles={}, gross=0.0)
        # Gross cap: target * multiple <= max_effective_gross.
        def _fillable(ticker: str, weight: float, mult: int) -> bool:
            if weight * mult > float(self.config.max_effective_gross) + 1e-12:
                return False
            notional = weight * cap
            avail = adv_map.get(ticker)
            if avail is None:
                # Unknown ADV fails closed to +1x only when checking +2x.
                return mult == 1
            try:
                return float(notional) <= float(avail) * float(part) + 1e-9
            except Exception:
                return False

        # Aggressive route: +2x only when rules/regime permit and fully fillable.
        want_two = (
            leverage_allowed is True
            and regime in ("RISK_ON", "STRONG_RISK_ON")
            and two_ticker is not None
        )
        if want_two and _fillable(str(two_ticker), target, 2):
            w = min(target, float(self.config.max_effective_gross) / 2.0)
            return PortfolioDecision(
                weights={str(two_ticker): w},
                rationale={str(two_ticker): "champion-tail aggressive +2x fillable"},
                vehicles={str(top_source): str(two_ticker)},
                gross=w * 2.0,
            )
        # Conservative/demote route: fully fillable +1x.
        if _fillable(str(one_ticker), target, 1):
            return PortfolioDecision(
                weights={str(one_ticker): target},
                rationale={str(one_ticker): "champion-tail +1x fillable"},
                vehicles={str(top_source): str(one_ticker)},
                gross=target,
            )
        return PortfolioDecision(weights={}, rationale={}, vehicles={}, gross=0.0)


__all__ = ["ChampionPolicyConfig", "ChampionTailPolicy"]
