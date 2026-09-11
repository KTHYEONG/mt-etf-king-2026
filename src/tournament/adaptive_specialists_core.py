from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from src.tournament.objective_core import CHAMPIONSHIP_THRESHOLDS, TOURNAMENT_SESSIONS

if TYPE_CHECKING:
    from src.portfolio.intent import PortfolioIntent


@dataclass(frozen=True, slots=True)
class SpecialistProposal:
    specialist: str
    ticker: str | None
    target_weight: float
    effective_gross: float
    score_key: tuple[float, float, str]


@dataclass(frozen=True, slots=True)
class AdaptiveSpecialistConfig:
    horizon: int = TOURNAMENT_SESSIONS
    warmup_sessions: int = 80
    max_weight: float = 0.80
    max_effective_gross: float = 1.60
    min_cash: float = 0.05
    reward_bound: float = 0.20
    ruin_floor: float = -0.25
    ruin_limit: float = 0.05
    min_terminal_scenarios: int = 30


_ORDER = ("long_broad", "long_theme", "inverse", "defensive")


def _is_high(conf: object) -> bool:
    return str(getattr(conf, "value", conf)).upper() == "HIGH"


def build_specialist_proposals(frame: pl.DataFrame, config: AdaptiveSpecialistConfig) -> tuple[SpecialistProposal, ...]:
    if frame.height == 0:
        raise ValueError("empty specialist frame")
    rows = list(frame.iter_rows(named=True))
    proposals: list[SpecialistProposal] = []
    for specialist in _ORDER:
        bucket_rows = [r for r in rows if str(r.get("bucket", specialist)) == specialist]
        valid = [
            r
            for r in bucket_rows
            if bool(r.get("eligible", True))
            and _is_high(r.get("confidence", "HIGH"))
            and math.isfinite(float(r.get("mom20", 0.0)))
            and math.isfinite(float(r.get("mom60", 0.0)))
        ]
        ranked = sorted(
            valid,
            key=lambda r: (-float(r.get("mom20", 0.0)), -float(r.get("mom60", 0.0)), str(r.get("ticker", ""))),
        )
        top = ranked[0] if ranked else None
        if top is None:
            proposals.append(SpecialistProposal(specialist, None, 0.0, 0.0, (float("inf"), float("inf"), "")))
            continue
        multiple = int(top.get("source_multiple", 1))
        if multiple == 0:
            raise ValueError("leverage multiple must be non-zero")
        weight = min(float(config.max_weight), float(config.max_effective_gross) / abs(multiple), 0.95)
        gross = abs(multiple) * float(weight)
        mom20 = float(top.get("mom20", 0.0))
        mom60 = float(top.get("mom60", 0.0))
        ticker = str(top.get("ticker", ""))
        proposals.append(SpecialistProposal(specialist, ticker, float(weight), float(gross), (-mom20, -mom60, ticker)))
    return tuple(proposals)


@dataclass(frozen=True, slots=True)
class FixedShareState:
    names: tuple[str, ...]
    probabilities: tuple[float, ...]
    observations: int


def initialize_fixed_share(names: Sequence[str]) -> FixedShareState:
    items = tuple(str(n) for n in names)
    if len(items) < 2:
        raise ValueError("fixed-share requires at least two specialists")
    if len(set(items)) != len(items):
        raise ValueError("duplicate specialist names")
    uniform = tuple(float(1.0 / len(items)) for _ in items)
    return FixedShareState(items, uniform, 0)


def update_fixed_share(
    state: FixedShareState,
    observed_rewards: Mapping[str, float],
    *,
    horizon: int,
    reward_bound: float,
) -> FixedShareState:
    if set(observed_rewards.keys()) != set(state.names):
        raise ValueError("reward keys must match specialist names")
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    if not math.isfinite(float(reward_bound)) or float(reward_bound) <= 0:
        raise ValueError("reward_bound must be positive finite")
    clipped: list[float] = []
    for name in state.names:
        value = float(observed_rewards[name])
        if not math.isfinite(value):
            raise ValueError("non-finite reward")
        bound = float(reward_bound)
        clipped.append(max(-bound, min(bound, value)))
    total = len(state.names)
    eta = math.sqrt(8.0 * math.log(total) / float(horizon)) / float(reward_bound)
    alpha = 1.0 / (float(horizon) - 1.0)
    log_terms = [math.log(p) + eta * r for p, r in zip(state.probabilities, clipped, strict=True)]
    peak = max(log_terms)
    shifted = [math.exp(term - peak) for term in log_terms]
    normalizer = sum(shifted)
    posterior = [(1.0 - alpha) * (w / normalizer) + alpha / total for w in shifted]
    return FixedShareState(state.names, tuple(float(v) for v in posterior), state.observations + 1)


@dataclass(frozen=True, slots=True)
class TournamentWealthState:
    equity: float
    start_equity: float
    sessions_remaining: int
    secured_threshold: float | None


@dataclass(frozen=True, slots=True)
class TerminalActionDistribution:
    action: str
    terminal_returns: tuple[float, ...]
    effective_gross: float


_WEIGHTS = (0.10, 0.25, 0.45, 0.20)


def _exceedance_utility(terminal: Sequence[float]) -> float:
    n = len(terminal)
    total = 0.0
    for threshold, weight in zip(CHAMPIONSHIP_THRESHOLDS, _WEIGHTS, strict=True):
        hits = sum(1.0 for r in terminal if float(r) > threshold)
        total += float(weight) * (hits / float(n))
    return float(total)


def choose_championship_action(
    state: TournamentWealthState,
    actions: Sequence[TerminalActionDistribution],
    config: AdaptiveSpecialistConfig,
) -> str:
    eligible: list[tuple[float, float, float, str]] = []
    for candidate in actions:
        terminal = tuple(float(v) for v in candidate.terminal_returns)
        if len(terminal) < int(config.min_terminal_scenarios):
            continue
        if any(not math.isfinite(v) for v in terminal):
            raise ValueError("non-finite terminal scenario")
        ruin = sum(1.0 for v in terminal if v < float(config.ruin_floor)) / float(len(terminal))
        if ruin > float(config.ruin_limit):
            continue
        if state.secured_threshold is not None:
            retention = sum(1.0 for v in terminal if v >= float(state.secured_threshold)) / float(len(terminal))
            if retention < 0.95:
                continue
        utility = _exceedance_utility(terminal)
        eligible.append((utility, -ruin, -float(candidate.effective_gross), candidate.action))
    if not eligible:
        names = [str(a.action) for a in actions]
        if state.secured_threshold is not None:
            return "CASH" if "CASH" in names else names[0]
        return names[0]
    ranked = sorted(eligible, key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    return str(ranked[0][3])


def _bucket_for(multiple: int, theme: object, family: object) -> str | None:
    # PIT-observable specialist partition using master attributes only.
    teat = str(theme or "").upper()
    if multiple == 2:
        if teat in ("EQUITY_BROAD", "KOSPI200", "KOSDAQ"):
            return "long_broad"
        return "long_theme"
    if multiple in (-1, -2):
        return "inverse"
    if multiple == 1 and teat in ("COMMODITY", "BOND", "CURRENCY"):
        return "defensive"
    return None


def _proposal_intent(proposal: SpecialistProposal) -> PortfolioIntent:
    from src.portfolio.intent import CASH_INTENT, PortfolioIntent

    if (
        proposal.ticker is None
        or not math.isfinite(float(proposal.target_weight))
        or float(proposal.target_weight) <= 0
    ):
        return CASH_INTENT
    return PortfolioIntent(kind="target", weights={str(proposal.ticker): float(proposal.target_weight)})


def _cash_proposals() -> tuple[SpecialistProposal, ...]:
    return tuple(
        SpecialistProposal(specialist, None, 0.0, 0.0, (float("inf"), float("inf"), "")) for specialist in _ORDER
    )


def _historical_terminal_paths(
    proposals_by_idx: Sequence[tuple[SpecialistProposal, ...]],
    opens_by_idx: Sequence[Mapping[str, float]],
    closes_by_idx: Sequence[Mapping[str, float]],
    horizon: int,
    cost_rate: float,
) -> dict[str, list[tuple[int, float]]]:
    """Cache completed specialist continuations for causal controller evidence."""
    paths: dict[str, list[tuple[int, float]]] = {name: [] for name in _ORDER}
    for decision_idx, proposals in enumerate(proposals_by_idx):
        end_idx = decision_idx + horizon
        if end_idx >= len(closes_by_idx):
            continue
        entry_idx = decision_idx + 1
        for proposal in proposals:
            if proposal.ticker is None or proposal.target_weight <= 0:
                continue
            entry = opens_by_idx[entry_idx].get(proposal.ticker)
            terminal = closes_by_idx[end_idx].get(proposal.ticker)
            if entry is None or terminal is None or entry <= 0 or terminal <= 0:
                continue
            gross_return = float(terminal) / float(entry) - 1.0
            net_return = float(proposal.target_weight) * gross_return - float(cost_rate)
            if math.isfinite(net_return):
                paths[proposal.specialist].append((end_idx, net_return))
    return paths


def _terminal_actions(
    *,
    decision_idx: int,
    current_return: float,
    router: FixedShareState,
    historical_paths: Mapping[str, Sequence[tuple[int, float]]],
    gross_by_specialist: Mapping[str, float],
    horizon: int,
    config: AdaptiveSpecialistConfig,
) -> tuple[TerminalActionDistribution, ...]:
    """Build actions from completed, non-overlapping historical paths only."""
    actions: list[TerminalActionDistribution] = []
    specialist_values: dict[str, tuple[float, ...]] = {}
    for name in _ORDER:
        selected: list[float] = []
        last_end = -1
        for end_idx, path_return in sorted(historical_paths.get(name, ()), key=lambda item: item[0]):
            if end_idx > decision_idx or (last_end >= 0 and end_idx - last_end < horizon):
                continue
            selected.append((1.0 + current_return) * (1.0 + float(path_return)) - 1.0)
            last_end = end_idx
        specialist_values[name] = tuple(selected)
        actions.append(
            TerminalActionDistribution(
                name,
                tuple(selected),
                float(gross_by_specialist.get(name, 0.0)),
            )
        )
    mixture: list[float] = []
    for row_idx in range(max((len(values) for values in specialist_values.values()), default=0)):
        value = 0.0
        mass = 0.0
        for name, probability in zip(router.names, router.probabilities, strict=True):
            values = specialist_values.get(name, ())
            if row_idx < len(values):
                value += float(probability) * float(values[row_idx])
                mass += float(probability)
        if mass > 0:
            mixture.append(value / mass)
    router_gross = sum(
        float(probability) * float(gross_by_specialist.get(name, 0.0))
        for name, probability in zip(router.names, router.probabilities, strict=True)
    )
    actions.insert(0, TerminalActionDistribution("ROUTER", tuple(mixture), router_gross))
    actions.append(TerminalActionDistribution("CASH", tuple([current_return] * config.min_terminal_scenarios), 0.0))
    return tuple(actions)
