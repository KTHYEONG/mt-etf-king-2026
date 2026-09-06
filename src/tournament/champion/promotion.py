# mypy: ignore-errors
# ruff: noqa
"""Champion promotion gates (P5 split of champion_eval.py)."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from src.alpha.base import AlphaModel, DecisionContext
from src.alpha.champion_dataset import ChampionDatasetConfig
from src.alpha.champion_dataset import build_family_tail_dataset
from src.alpha.champion_dataset import collect_direct_vehicle_candidates
from src.alpha.champion_dataset import collect_family_candidates
from src.alpha.champion_ranker import ChampionTailRanker, ChampionTrainingError, OosScoreStore, PurgedDateWalkForward, TailGradeObjective, is_valid_champion_artifact
from src.alpha.executable_labels import ExecutableLabelConfig, build_executable_vehicle_labels
from src.alpha.opportunity_hurdle import ExecutableOpportunityModel, fit_executable_hurdle
from src.portfolio.intent import HOLD_INTENT, PortfolioIntent
from src.portfolio.policy import PortfolioDecision
from src.strategies.champion_tail import ChampionPolicyConfig, ChampionTailPolicy
from src.tournament.objective_core import TOURNAMENT_SESSIONS


def _primary_holdings_from_backtest(backtest: object | None, sessions: Sequence[date]) -> list[str | None]:
    by_date: dict[date, str | None] = {}
    if backtest is not None:
        trades = getattr(backtest, "trades", None)
        if trades is not None and getattr(trades, "height", 0) > 0:
            dd_col = "decision_date" if "decision_date" in trades.columns else ("date" if "date" in trades.columns else None)
            w_col = "weight_after" if "weight_after" in trades.columns else ("weight" if "weight" in trades.columns else None)
            t_col = "ticker" if "ticker" in trades.columns else None
            if dd_col and w_col and t_col:
                try:
                    for d in trades[dd_col].unique().to_list():
                        if not isinstance(d, date):
                            continue
                        sub = trades.filter(pl.col(dd_col) == d)
                        best_ticker: str | None = None
                        best_w = -1.0
                        for row in sub.iter_rows(named=True):
                            try:
                                wf = float(row.get(w_col, 0.0))
                            except Exception:
                                continue
                            if wf > best_w:
                                best_w = wf
                                raw_t = row.get(t_col)
                                best_ticker = str(raw_t) if raw_t is not None else None
                        by_date[d] = best_ticker if best_w > 1e-9 else None
                except Exception:
                    by_date = {}
    out: list[str | None] = []
    last: str | None = None
    for s in sessions:
        if s in by_date:
            last = by_date[s]
        out.append(last)
    return out


def _count_effective_discordant(
    *,
    paired_starts: Sequence[date],
    sessions: Sequence[date],
    horizon: int,
    candidate_backtest: object | None,
    incumbent_backtest: object | None,
    candidate_window_holdings: Sequence[Sequence[str | None]] | None = None,
    incumbent_window_holdings: Sequence[Sequence[str | None]] | None = None,
) -> int:
    if not paired_starts or not sessions:
        return 0
    if candidate_window_holdings is not None or incumbent_window_holdings is not None:
        try:
            if candidate_window_holdings is None or incumbent_window_holdings is None:
                return 0
            cand_traces = [tuple(w) for w in candidate_window_holdings]
            inc_traces = [tuple(w) for w in incumbent_window_holdings]
            if len(cand_traces) != len(list(paired_starts)) or len(inc_traces) != len(list(paired_starts)):
                return 0
            h = int(horizon)
            for ct, it in zip(cand_traces, inc_traces, strict=True):
                if len(ct) != h or len(it) != h:
                    return 0
                for v in (*ct, *it):
                    if v is not None and not isinstance(v, str):
                        return 0
            count = 0
            for ct, it in zip(cand_traces, inc_traces, strict=True):
                if any(c != i for c, i in zip(ct, it, strict=True)):
                    count += 1
            return int(count)
        except Exception:  # pragma: no cover - malformed trace container
            return 0
    cand = _primary_holdings_from_backtest(candidate_backtest, sessions)
    inc = _primary_holdings_from_backtest(incumbent_backtest, sessions)
    try:
        mask = discordant_window_mask(cand, inc, int(horizon))
    except Exception:
        return 0
    paired = set(paired_starts)
    count = 0
    for i, s in enumerate(sessions):
        if s not in paired:
            continue
        if i < len(mask) and bool(mask[i]):
            count += 1
    return int(count)


def champion_promotion_status(
    *,
    paired_starts: Sequence[date],
    sessions: Sequence[date],
    horizon: int,
    candidate_backtest: object | None,
    incumbent_backtest: object | None,
    aggressive_status: str,
    conservative_status: str,
    loyo_status: str,
    artifact_integrity: bool,
    candidate_window_holdings: Sequence[Sequence[str | None]] | None = None,
    incumbent_window_holdings: Sequence[Sequence[str | None]] | None = None,
) -> tuple[str, int, int, bool]:
    n_effective_discordant = _count_effective_discordant(
        paired_starts=paired_starts,
        sessions=sessions,
        horizon=horizon,
        candidate_backtest=candidate_backtest,
        incumbent_backtest=incumbent_backtest,
        candidate_window_holdings=candidate_window_holdings,
        incumbent_window_holdings=incumbent_window_holdings,
    )
    try:
        from src.tournament.attainability import load_attainability_config

        _, _, min_effective_discordant = load_attainability_config()
    except Exception:
        min_effective_discordant = 5
    promotion_status = resolve_promotion_status(
        aggressive_status=aggressive_status,
        conservative_status=conservative_status,
        loyo_status=loyo_status,
        artifact_integrity=artifact_integrity,
        n_effective_discordant=int(n_effective_discordant),
        min_effective_discordant=int(min_effective_discordant),
    )
    return (
        str(promotion_status),
        int(n_effective_discordant),
        int(min_effective_discordant),
        bool(promotion_status == "PROMOTE"),
    )


def discordant_window_mask(
    candidate_holdings: Sequence[str | None], incumbent_holdings: Sequence[str | None], horizon: int
) -> tuple[bool, ...]:
    try:
        cand = list(candidate_holdings)
        inc = list(incumbent_holdings)
    except Exception:
        raise ValueError("holdings must be sequences")
    if len(cand) != len(inc):
        raise ValueError(f"mismatched holdings lengths: {len(cand)} != {len(inc)}")
    try:
        h = int(horizon)
    except Exception:
        raise ValueError("horizon must be int")
    if h <= 0:
        raise ValueError("horizon must be positive")
    n = len(cand)
    n_win = n - h + 1
    if n_win <= 0:
        return ()
    out: list[bool] = []
    for i in range(n_win):
        disc = False
        for j in range(i, i + h):
            try:
                if cand[j] != inc[j]:
                    disc = True
                    break
            except Exception:
                continue
        out.append(bool(disc))
    return tuple(out)


def resolve_promotion_status(
    *,
    aggressive_status: str,
    conservative_status: str,
    loyo_status: str,
    artifact_integrity: bool,
    n_effective_discordant: int,
    min_effective_discordant: int,
) -> str:
    try:
        n = int(n_effective_discordant)
    except Exception:
        n = 0
    try:
        m = int(min_effective_discordant)
    except Exception:
        m = 0
    gates_pass = (
        aggressive_status == "PASS"
        and conservative_status == "PASS"
        and loyo_status == "PASS"
        and artifact_integrity is True
    )
    if not gates_pass:
        return "RESEARCH_ONLY"
    if n < m:
        return "INSUFFICIENT_POWER"
    return "PROMOTE"


def is_promotable(
    *,
    aggressive_status: str,
    conservative_status: str,
    loyo_status: str,
    artifact_integrity: bool,
    n_effective_discordant: int = 10**9,
    min_effective_discordant: int = 0,
) -> bool:
    """Promotion requires dual-scenario PASS, LOYO PASS, and artifact integrity."""
    return bool(
        resolve_promotion_status(
            aggressive_status=aggressive_status, conservative_status=conservative_status,
            loyo_status=loyo_status, artifact_integrity=artifact_integrity,
            n_effective_discordant=n_effective_discordant,
            min_effective_discordant=min_effective_discordant,
        )
        == "PROMOTE"
    )
