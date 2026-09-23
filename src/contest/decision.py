"""Weekly rank-objective contest decision.

Replaces the daily champion decision with a weekly recommendation maximizing
P(rank 1) at contest end, estimated with the contest simulator (panel, vectorized
engine, calibrated crowd) against the archived leaderboard. Execution stays
manual (HTS at the next session's open). Fail-closed: stale or partial inputs
yield NO_DATA or NEEDS_CONFIRMATION cards, never a SWITCH.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.contest.crowd import append_explicit_leaders, build_crowd, pin_to_leaderboard
from src.contest.engine import SimState, bootstrap_worlds, rank_metrics, simulate_contest
from src.contest.leaderboard import LeaderboardSnapshot, infer_single_vehicle_holders
from src.contest.panel import VehiclePanel, neutralize_drift
from src.core.calendar import TradingCalendar
from src.core.paths import DataPaths

logger = logging.getLogger(__name__)

_HOLD = "HOLD"
_MIMIC_PREFIX = "MIMIC:"


class ContestAction(StrEnum):
    HOLD = "HOLD"
    SWITCH = "SWITCH"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    NO_DATA = "NO_DATA"
    CONTEST_OVER = "CONTEST_OVER"


@dataclass(frozen=True)
class CandidateScore:
    alias: str
    ticker: str
    p1_mean: float
    p1_min: float
    p2_mean: float
    p10_mean: float
    median_return: float
    p_loss30: float


@dataclass(frozen=True)
class ContestDecision:
    """Weekly recommendation card for the manual HTS order at the next session's open."""

    decision_session: date
    execution_session: date | None
    action: ContestAction
    current_alias: str | None
    target_alias: str | None
    target_ticker: str | None
    target_weight: float
    our_rank: int | None
    our_total_return_pct: float | None
    leaderboard_base_date: date | None
    sessions_remaining: int
    scores: tuple[CandidateScore, ...]
    inferred_leaders: dict[str, tuple[str, float]]
    warnings: tuple[str, ...]


def resolve_current_holding(
    snapshot: LeaderboardSnapshot | None,
    nickname: str,
    vehicle_changes_pct: Mapping[str, float],
    state_alias: str | None,
    tol_pct: float,
    min_weight: float,
) -> tuple[str | None, list[str]]:
    """Our holding at the decision session.

    Primary source is the recorded state. It is cross-checked against the single-vehicle inference from our own
    leaderboard daily return, because execution is manual and may diverge from the recommendation.

    Returns:
        (alias or None, warnings).
        - Inferred and recorded disagree: returns the inferred alias with warning `HOLDING_MISMATCH`.
        - Neither is available: returns None with warning `HOLDING_UNKNOWN`.
    """
    inferred: str | None = None
    if snapshot is not None and nickname:
        hit = infer_single_vehicle_holders(snapshot, vehicle_changes_pct, tol_pct, min_weight).get(nickname)
        inferred = hit[0] if hit is not None else None
    if inferred is not None and state_alias is not None and inferred != state_alias:
        return inferred, ["HOLDING_MISMATCH"]
    if inferred is not None:
        return inferred, []
    if state_alias is not None:
        return state_alias, []
    return None, ["HOLDING_UNKNOWN"]


def realized_equity(
    panel: VehiclePanel,
    alias: str,
    entry_session: date,
    end_session: date,
    entry_equity: float,
    weight: float,
) -> float:
    """Compound a holding's close-to-close legs from `entry_session` (exclusive) to `end_session` (inclusive).

    `entry_equity` is the equity at the `entry_session` close (the state file's recorded entry level). Used when
    our leaderboard entry is missing (outside the top-50): equity from the recorded holding plus entry equity.
    """
    vidx = panel.index(alias)
    rows = [r for r, d in enumerate(panel.dates) if entry_session < d <= end_session]
    return _compound_legs(panel, rows, vidx, weight, float(entry_equity))


def _compound_legs(panel: VehiclePanel, rows: Sequence[int], vidx: int, weight: float, start: float) -> float:
    equity = float(start)
    for r in rows:
        gap = float(panel.gap[r, vidx])
        intra = float(panel.intraday[r, vidx])
        equity *= (1.0 + weight * gap) * (1.0 + weight * intra)
    return equity


def panel_changes_pct(panel: VehiclePanel, session: date) -> dict[str, float]:
    """Close-to-close percent change per vehicle at a session (for single-vehicle inference)."""
    row = panel.row(session)
    out: dict[str, float] = {}
    for j, alias in enumerate(panel.names):
        gap = float(panel.gap[row, j])
        intra = float(panel.intraday[row, j])
        out[alias] = ((1.0 + gap) * (1.0 + intra) - 1.0) * 100.0
    return out


def adv_krw(data_root: Path, ticker: str, session: date, sessions: Sequence[date]) -> float:
    """20-session mean trading value (KRW) for a ticker over sessions <= `session`."""
    lookback = [s for s in sessions if s <= session][-20:]
    if not lookback:
        return 0.0
    silver = DataPaths(root=data_root).silver("etf_daily")
    if not silver.is_file():
        return 0.0
    frame = (
        pl.scan_parquet(silver)
        .filter(pl.col("ticker") == ticker, pl.col("date").is_in(lookback))
        .select("trading_value")
        .collect()
    )
    if frame.height == 0:
        return 0.0
    nums = [float(v) for v in frame.get_column("trading_value").to_list() if v is not None]
    if not nums:
        return 0.0
    return float(sum(nums) / len(nums))


def choose_target(
    scores: Mapping[str, tuple[float, float]], current: str | None, min_gain: float
) -> tuple[str, ContestAction]:
    """Pick argmax mean P1 (tie-broken by mean P2); SWITCH only on a sufficient gain over current."""
    best = max(scores, key=lambda a: (scores[a][0], scores[a][1]))
    if current is None:
        return best, ContestAction.SWITCH
    if best != current and scores[best][0] - scores[current][0] >= min_gain:
        return best, ContestAction.SWITCH
    return current, ContestAction.HOLD


def build_explicit_leaders(
    snapshot: LeaderboardSnapshot,
    inferred: Mapping[str, tuple[str, float]],
    nickname: str,
    panel: VehiclePanel,
) -> tuple[list[tuple[float, int]], list[tuple[int, float, int]], np.ndarray]:
    """Split inferred single-vehicle holders (excluding us) into churn-agent holders, pin entries, and top values."""
    top_values = np.array(
        sorted((1.0 + e.total_return_pct / 100.0 for e in snapshot.entries), reverse=True), dtype=np.float32
    )
    holders: list[tuple[float, int]] = []
    pin_entries: list[tuple[int, float, int]] = []
    order = sorted(snapshot.entries, key=lambda e: e.rank)
    used = 0
    for entry in order:
        hit = inferred.get(entry.user_name)
        if hit is None or entry.user_name == nickname:
            continue
        try:
            vidx = panel.index(hit[0])
        except KeyError:
            continue
        equity = 1.0 + entry.total_return_pct / 100.0
        holders.append((hit[1], vidx))
        pin_entries.append((used, equity, vidx))
        used += 1
    return holders, pin_entries, top_values


def _candidate_vehicles(
    candidates: Sequence[str],
    current: str | None,
    inferred: Mapping[str, tuple[str, float]],
    snapshot: LeaderboardSnapshot,
    nickname: str,
    endgame: bool,
) -> list[str]:
    vehicles = list(candidates)
    if endgame:
        best_rank: int | None = None
        best_alias: str | None = None
        for user, hit in inferred.items():
            if user == nickname:
                continue
            entry = snapshot.entry_for(user)
            if entry is None:
                continue
            if best_rank is None or entry.rank < best_rank:
                best_rank = entry.rank
                best_alias = hit[0]
        if best_alias is not None:
            mimic = f"{_MIMIC_PREFIX}{best_alias}"
            if mimic not in vehicles:
                vehicles.append(mimic)
    return vehicles


def _action_fn(realized_idx: int, vehicle_idx: int, switch_day: int) -> Callable[[int, np.ndarray], np.ndarray]:
    def fn(day: int, held: np.ndarray) -> np.ndarray:
        return np.full_like(held, realized_idx if day < switch_day else vehicle_idx)
    return fn


def decide_week(
    panel: VehiclePanel,
    snapshot: LeaderboardSnapshot | None,
    decision_session: date,
    calendar: TradingCalendar,
    config: Mapping[str, Any],
    state_alias: str | None,
) -> ContestDecision:
    """Estimate P(rank1)/P(top2)/P(top10) for every eligible candidate, assuming it is held from the next open to
    contest end, and choose the recommendation.

    Our standing comes from the leaderboard entry for the nickname. Remaining sessions and realized contest
    sessions come from the trading calendar.

    Raises:
        Nothing for missing data. Missing data yields NO_DATA or NEEDS_CONFIRMATION cards (fail-closed: never a
        SWITCH on stale or partial inputs).
    """
    nickname = str(config.get("nickname", ""))
    start_date = date.fromisoformat(str(config["start_date"]))
    end_date = date.fromisoformat(str(config["end_date"]))
    vehicles = config["vehicles"]
    inference = config.get("inference", {})
    tol_pct = float(inference.get("match_tol_pct", 0.02))
    min_weight = float(inference.get("min_weight", 0.90))
    decision_cfg = config["decision"]
    crowd_cfg = config["crowd"]
    target_weight = float(decision_cfg["target_weight"])
    min_gain = float(decision_cfg["switch_min_p1_gain"])
    endgame_n = int(decision_cfg["endgame_sessions"])
    candidates_cfg = [str(c) for c in decision_cfg["candidates"]]
    for cand in candidates_cfg:
        if cand != _HOLD and not cand.startswith(_MIMIC_PREFIX) and cand not in vehicles:
            raise ValueError(f"unknown candidate vehicle: {cand}")

    if decision_session > end_date:
        current, hold_warnings = resolve_current_holding(snapshot, nickname, {}, state_alias, tol_pct, min_weight)
        entry = snapshot.entry_for(nickname) if snapshot is not None else None
        return ContestDecision(
            decision_session=decision_session, execution_session=None, action=ContestAction.CONTEST_OVER,
            current_alias=current, target_alias=None, target_ticker=None, target_weight=target_weight,
            our_rank=entry.rank if entry else None,
            our_total_return_pct=entry.total_return_pct if entry else None,
            leaderboard_base_date=snapshot.base_date if snapshot else None, sessions_remaining=0,
            scores=(), inferred_leaders={}, warnings=tuple(hold_warnings),
        )

    remaining = [s for s in calendar.sessions(decision_session, end_date) if s > decision_session]
    if not remaining:
        current, hold_warnings = resolve_current_holding(snapshot, nickname, {}, state_alias, tol_pct, min_weight)
        entry = snapshot.entry_for(nickname) if snapshot is not None else None
        return ContestDecision(
            decision_session=decision_session, execution_session=None, action=ContestAction.CONTEST_OVER,
            current_alias=current, target_alias=None, target_weight=target_weight, target_ticker=None,
            our_rank=entry.rank if entry else None,
            our_total_return_pct=entry.total_return_pct if entry else None,
            leaderboard_base_date=snapshot.base_date if snapshot else None, sessions_remaining=0,
            scores=(), inferred_leaders={}, warnings=tuple(hold_warnings),
        )

    if snapshot is None or snapshot.base_date != decision_session:
        changes: dict[str, float] = {}
        if snapshot is not None:
            try:
                changes = panel_changes_pct(panel, snapshot.base_date)
            except ValueError:
                changes = {}
        current, hold_warnings = resolve_current_holding(snapshot, nickname, changes, state_alias, tol_pct, min_weight)
        entry = snapshot.entry_for(nickname) if snapshot is not None else None
        ticker = vehicles[current]["ticker"] if current in vehicles else None
        return ContestDecision(
            decision_session=decision_session, execution_session=None, action=ContestAction.NO_DATA,
            current_alias=current, target_alias=current, target_ticker=ticker, target_weight=target_weight,
            our_rank=entry.rank if entry else None,
            our_total_return_pct=entry.total_return_pct if entry else None,
            leaderboard_base_date=snapshot.base_date if snapshot else None,
            sessions_remaining=len(remaining), scores=(), inferred_leaders={},
            warnings=("LEADERBOARD_STALE", *hold_warnings),
        )

    try:
        decision_row = panel.row(decision_session)
    except ValueError:
        current, hold_warnings = resolve_current_holding(snapshot, nickname, {}, state_alias, tol_pct, min_weight)
        entry = snapshot.entry_for(nickname)
        ticker = vehicles[current]["ticker"] if current in vehicles else None
        return ContestDecision(
            decision_session=decision_session, execution_session=None, action=ContestAction.NO_DATA,
            current_alias=current, target_alias=current, target_ticker=ticker, target_weight=target_weight,
            our_rank=entry.rank if entry else None,
            our_total_return_pct=entry.total_return_pct if entry else None,
            leaderboard_base_date=snapshot.base_date, sessions_remaining=len(remaining),
            scores=(), inferred_leaders={}, warnings=tuple(hold_warnings),
        )

    changes = panel_changes_pct(panel, decision_session)
    inferred = infer_single_vehicle_holders(snapshot, changes, tol_pct, min_weight)
    current, hold_warnings = resolve_current_holding(snapshot, nickname, changes, state_alias, tol_pct, min_weight)
    entry = snapshot.entry_for(nickname)
    warnings: list[str] = list(hold_warnings)

    our_equity: float | None = None
    if entry is not None:
        our_equity = 1.0 + entry.total_return_pct / 100.0
        our_rank: int | None = entry.rank
        our_total: float | None = entry.total_return_pct
    else:
        our_rank, our_total = None, None
        warnings.append("OUTSIDE_TOP50")

    realized = list(calendar.sessions(start_date, decision_session))
    realized_rows = [panel.row(s) for s in realized]
    switch_day = len(realized_rows)
    n_remaining = len(remaining)

    candidate_ids = _candidate_vehicles(
        candidates_cfg, current, inferred, snapshot, nickname, n_remaining <= endgame_n
    )
    min_adv = float(decision_cfg.get("min_adv_krw", 0))
    eligible: dict[str, str] = {}
    for cid in candidate_ids:
        alias = current if cid == _HOLD else cid[len(_MIMIC_PREFIX):] if cid.startswith(_MIMIC_PREFIX) else cid
        if alias is None or alias not in vehicles:
            continue
        if min_adv > 0:
            adv = adv_krw(_data_root(config), str(vehicles[alias].get("ticker", "")), decision_session, realized)
            if adv < min_adv:
                warnings.append(f"ILLIQUID:{alias}")
                continue
        eligible[cid] = alias
    if our_equity is None or not eligible:
        return ContestDecision(
            decision_session=decision_session, execution_session=None, action=ContestAction.NEEDS_CONFIRMATION,
            current_alias=current, target_alias=current,
            target_ticker=vehicles[current]["ticker"] if current in vehicles else None,
            target_weight=target_weight, our_rank=our_rank, our_total_return_pct=our_total,
            leaderboard_base_date=snapshot.base_date, sessions_remaining=n_remaining,
            scores=(), inferred_leaders=dict(inferred),
            warnings=tuple(warnings),
        )

    eras = decision_cfg["eras"]
    profiles = [str(p) for p in decision_cfg["profiles"]]
    current_key = _HOLD if _HOLD in eligible else current
    n_worlds = int(decision_cfg["n_worlds"])
    mean_block = float(decision_cfg["mean_block"])
    seed = int(decision_cfg["seed"])
    pop_auto = crowd_cfg["popularity_auto"]
    pop_non = crowd_cfg["popularity_nonauto"]
    n_participants = int(crowd_cfg["n_participants"])
    f_auto = float(crowd_cfg["f_auto"])
    entry_days = list(crowd_cfg["entry_days"])
    entry_probs = list(crowd_cfg["entry_probs"])
    churn = crowd_cfg["leader_churn"]

    holders, pin_entries, top_values = build_explicit_leaders(snapshot, inferred, nickname, panel)
    per_candidate_p1: dict[str, list[float]] = {cid: [] for cid in eligible}
    per_candidate_p2: dict[str, list[float]] = {cid: [] for cid in eligible}
    per_candidate_p10: dict[str, list[float]] = {cid: [] for cid in eligible}
    per_candidate_ret: dict[str, list[float]] = {cid: [] for cid in eligible}
    per_candidate_loss: dict[str, list[float]] = {cid: [] for cid in eligible}
    n_runs = len(eras) * len(profiles)
    run_idx = 0
    for era_name, era_cfg in eras.items():
        era_start = date.fromisoformat(str(era_cfg["start"]))
        era_panel = panel
        if bool(era_cfg.get("neutral", False)):
            era_panel = neutralize_drift(panel, era_start, decision_session, realized)
        pool = np.arange(panel.row(era_start), decision_row + 1, dtype=np.int64)
        for profile_name in profiles:
            run_idx += 1
            t0 = time.perf_counter()
            run_seed = seed + run_idx
            worlds = bootstrap_worlds(era_panel, realized_rows, pool, n_worlds, n_remaining, mean_block, run_seed)
            profile = crowd_cfg["profiles"][profile_name]
            crowd = build_crowd(
                era_panel, worlds, profile, pop_auto, pop_non, n_participants, f_auto,
                entry_days, entry_probs, run_seed + 1,
            )
            crowd, leader_idx = append_explicit_leaders(
                crowd, holders, int(churn["k"]), float(churn["q"])
            )
            pin_explicit = [(leader_idx[i], pin_entries[i][1], pin_entries[i][2]) for i in range(len(pin_entries))]
            realized_idx = era_panel.index(eligible[current_key]) if current_key in eligible else -1

            def on_day_end(day: int, state: SimState, _pin_day: int = switch_day - 1,
                           _top: np.ndarray = top_values,
                           _expl: list[tuple[int, float, int]] = pin_explicit) -> None:
                if day == _pin_day:
                    pin_to_leaderboard(state, _top, _expl)

            actions = {
                cid: _action_fn(realized_idx, era_panel.index(alias), switch_day)
                for cid, alias in eligible.items()
            }
            sched = {cid: [target_weight] * (switch_day + n_remaining) for cid in eligible}
            result = simulate_contest(era_panel, worlds, crowd, actions, sched, on_day_end, run_seed + 2)
            if realized_idx >= 0:
                realized_base = _compound_legs(era_panel, realized_rows, realized_idx, target_weight, 1.0)
            else:
                realized_base = 1.0
            for cid in eligible:
                path = result.ours[cid]
                pinned = path / realized_base * our_equity
                metrics = rank_metrics(pinned, result.crowd_equity)
                per_candidate_p1[cid].append(metrics["P1"])
                per_candidate_p2[cid].append(metrics["P2"])
                per_candidate_p10[cid].append(metrics["P_TOP10"])
                per_candidate_ret[cid].append(metrics["med_ret"])
                per_candidate_loss[cid].append(metrics["P_loss30"])
            elapsed = time.perf_counter() - t0
            logger.info(f"[PORTFOLIO] contest_weekly config={era_name}/{profile_name} {run_idx}/{n_runs} {elapsed:.1f}s")

    scores = tuple(
        CandidateScore(
            alias=cid, ticker=str(vehicles[eligible[cid]]["ticker"]),
            p1_mean=float(np.mean(per_candidate_p1[cid])), p1_min=float(np.min(per_candidate_p1[cid])),
            p2_mean=float(np.mean(per_candidate_p2[cid])), p10_mean=float(np.mean(per_candidate_p10[cid])),
            median_return=float(np.mean(per_candidate_ret[cid])), p_loss30=float(np.mean(per_candidate_loss[cid])),
        )
        for cid in eligible
    )
    table = {cid: (s.p1_mean, s.p2_mean) for cid, s in zip(eligible, scores, strict=True)}
    target_key, action = choose_target(table, current_key if current_key in table else None, min_gain)
    target_alias = eligible[target_key]
    execution = calendar.next_session(decision_session) if action == ContestAction.SWITCH else None
    return ContestDecision(
        decision_session=decision_session, execution_session=execution, action=action,
        current_alias=current, target_alias=target_alias,
        target_ticker=str(vehicles[target_alias]["ticker"]), target_weight=target_weight,
        our_rank=our_rank, our_total_return_pct=our_total,
        leaderboard_base_date=snapshot.base_date, sessions_remaining=n_remaining,
        scores=scores, inferred_leaders=dict(inferred), warnings=tuple(warnings),
    )


def _data_root(config: Mapping[str, Any]) -> Path:
    from src.core.settings import get_settings

    return Path(get_settings().data_root)


def decision_to_dict(decision: ContestDecision) -> dict[str, Any]:
    """Serialize a decision card to JSON-ready primitives."""
    return {
        "decision_session": decision.decision_session.isoformat(),
        "execution_session": decision.execution_session.isoformat() if decision.execution_session else None,
        "action": str(decision.action.value),
        "current_alias": decision.current_alias,
        "target_alias": decision.target_alias,
        "target_ticker": decision.target_ticker,
        "target_weight": decision.target_weight,
        "our_rank": decision.our_rank,
        "our_total_return_pct": decision.our_total_return_pct,
        "leaderboard_base_date": decision.leaderboard_base_date.isoformat() if decision.leaderboard_base_date else None,
        "sessions_remaining": decision.sessions_remaining,
        "scores": [
            {"alias": s.alias, "ticker": s.ticker, "p1_mean": s.p1_mean, "p1_min": s.p1_min,
             "p2_mean": s.p2_mean, "p10_mean": s.p10_mean, "median_return": s.median_return,
             "p_loss30": s.p_loss30}
            for s in decision.scores
        ],
        "inferred_leaders": {u: [a, w] for u, (a, w) in decision.inferred_leaders.items()},
        "warnings": list(decision.warnings),
    }


def render_decision_markdown(decision: ContestDecision, names: Mapping[str, str]) -> str:
    """Korean decision card: action, from→to (ticker and name), execution date, target weight, our rank and return,
    a candidate table (P1 mean/min, P2, P10, median return, P(loss>30%)), inferred leaders, and warnings."""
    lines = [f"# 주간 콘테스트 결정 ({decision.decision_session.isoformat()})", ""]
    action = decision.action.value
    cur = decision.current_alias or "미확인"
    tgt = decision.target_alias or "미정"
    tick = decision.target_ticker or "-"
    tname = names.get(decision.target_alias, "") if decision.target_alias else ""
    lines.append(f"- 행동: {action}")
    lines.append(f"- 전환: {cur} → {tgt} ({tick}{(' ' + tname) if tname else ''})")
    lines.append(f"- 집행일: {decision.execution_session.isoformat() if decision.execution_session else '없음'} (09:00 HTS)")
    lines.append(f"- 목표 비중: {decision.target_weight:.3f}")
    rank = str(decision.our_rank) if decision.our_rank is not None else "권외"
    ret = f"{decision.our_total_return_pct:.3f}%" if decision.our_total_return_pct is not None else "-"
    lines.append(f"- 우리 순위/수익률: {rank} / {ret} (기준일 {decision.leaderboard_base_date})")
    lines.append(f"- 잔여 세션: {decision.sessions_remaining}")
    lines.append("")
    lines.append("## 후보 점수")
    lines.append("| 후보 | P1 평균 | P1 최소 | P2 | P10 | 중앙값 수익률 | 손실30%확률 |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.extend(
        f"| {s.alias} | {s.p1_mean:.4f} | {s.p1_min:.4f} | {s.p2_mean:.4f} | {s.p10_mean:.4f} "
        f"| {s.median_return:.4f} | {s.p_loss30:.4f} |"
        for s in decision.scores
    )
    if not decision.scores:
        lines.append("| (없음) | - | - | - | - | - | - |")
    lines.append("")
    lines.append("## 추정 리더")
    for user, (alias, weight) in decision.inferred_leaders.items():
        lines.append(f"- {user}: {alias} (비중 {weight:.3f})")
    if not decision.inferred_leaders:
        lines.append("- 없음")
    lines.append("")
    lines.append("## 경고")
    lines.extend(f"- {warning}" for warning in decision.warnings)
    if not decision.warnings:
        lines.append("- 없음")
    return "\n".join(lines) + "\n"
