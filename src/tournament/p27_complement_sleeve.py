"""P27 hard-switch complement sleeve (research-only, never a production gate)."""

from __future__ import annotations

import math
from typing import Final, Literal

import polars as pl

from src.alpha.base import DecisionContext
from src.tournament.championship_regime import ChampionshipSleeve

P27_COMPLEMENT_IS_PRODUCTION_GATE: Final[bool] = False

ComplementBook = Literal["attack", "complement"]

__all__ = [
    "P27_COMPLEMENT_IS_PRODUCTION_GATE",
    "ComplementBook",
    "P27ComplementHardSwitchModel",
    "evaluate_complement_tail_non_inferiority",
    "make_p27_complement_switch",
    "may_promote_complement_over_p27",
    "reject_if_capital_split",
    "resolve_complement_active_book",
]


def resolve_complement_active_book(*, sleeve: ChampionshipSleeve | str | None) -> ComplementBook:
    if sleeve == ChampionshipSleeve.LOTTERY_ON:
        return "attack"
    return "complement"


def reject_if_capital_split(complement_share: float, *, eps: float = 1e-12) -> None:
    share = float(complement_share)
    if eps < share < 1.0 - eps:
        raise ValueError(f"capital split forbidden: complement_share={share!r} must be exclusive 0.0 or 1.0")


def evaluate_complement_tail_non_inferiority(*, candidate_p50: float, incumbent_p50: float) -> bool:
    candidate = float(candidate_p50)
    incumbent = float(incumbent_p50)
    if not math.isfinite(candidate) or not math.isfinite(incumbent):
        return False
    return candidate >= incumbent


def may_promote_complement_over_p27(*, candidate_p50: float, incumbent_p50: float) -> bool:
    if not P27_COMPLEMENT_IS_PRODUCTION_GATE:
        return False
    return evaluate_complement_tail_non_inferiority(candidate_p50=candidate_p50, incumbent_p50=incumbent_p50)


class P27ComplementHardSwitchModel:
    path_dependent: bool = True

    def __init__(self, *, attack: object, complement: object, name: str = "sticky.p27_complement_switch") -> None:
        self.attack = attack
        self.complement = complement
        self.name = name

    def reset_trackers(self) -> None:
        attack_reset = getattr(self.attack, "reset_trackers", None)
        if callable(attack_reset):
            attack_reset()
        complement_reset = getattr(self.complement, "reset_trackers", None)
        if callable(complement_reset):
            complement_reset()

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float] | object:
        book = resolve_complement_active_book(sleeve=context.championship_sleeve)
        active = self.attack if book == "attack" else self.complement
        score_fn = getattr(active, "score", None)
        if not callable(score_fn):
            raise TypeError(f"active book missing callable score: {type(active)!r}")
        result: dict[str, float] | object = score_fn(snapshot, context)
        return result


def make_p27_complement_switch() -> P27ComplementHardSwitchModel:
    from src.strategies.factories.baselines import make_baseline_mom20_top1
    from src.strategies.factories.sticky_mom60 import make_sticky_mom60_raw

    return P27ComplementHardSwitchModel(attack=make_sticky_mom60_raw(), complement=make_baseline_mom20_top1())
