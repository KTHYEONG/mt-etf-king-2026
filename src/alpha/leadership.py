from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.alpha.base import DecisionContext
from src.alpha.cluster import ClusterResolver, SnapshotIndex, build_snapshot_index
from src.alpha.state import ThemeMetrics as StateMetrics
from src.alpha.state import ThemeState, TransitionConfig, run_state_machine, transition
from src.alpha.theme import ThemePanel, build_theme_panel
from src.universe.instruments import InstrumentMaster


@dataclass(frozen=True)
class SectorScoreWeights:
    rs: float
    accel: float
    breadth: float

    def sector_score(self, rs: float, accel: float, breadth: float) -> float:
        return float(self.rs * rs + self.accel * accel + self.breadth * breadth)

    @classmethod
    def from_yaml(cls, raw: Mapping[str, object]) -> SectorScoreWeights:
        rs_val = raw.get("rs", raw.get("RS", 0.0))
        rs = float(rs_val)  # type: ignore[arg-type]
        accel = float(raw.get("accel", 0.0))  # type: ignore[arg-type]
        breadth = float(raw.get("breadth", 0.0))  # type: ignore[arg-type]
        breakout = float(raw.get("breakout", 0.0) or 0.0)  # type: ignore[arg-type]
        flow = float(raw.get("flow", 0.0) or 0.0)  # type: ignore[arg-type]
        if abs(breakout) > 1e-9 or abs(flow) > 1e-9:
            raise ValueError(f"breakout and flow weights must be 0.0, got breakout={breakout} flow={flow}")
        total = rs + accel + breadth
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0, got {total}")
        return cls(rs=rs, accel=accel, breadth=breadth)


TRADABLE_THEME_STATES: frozenset[ThemeState] = frozenset({ThemeState.LEADING, ThemeState.RECOVERY})


def filter_scores_by_theme_state(
    scores: Mapping[str, float],
    theme_states: Mapping[str, str],
    tradable: frozenset[str] | None = None,
) -> dict[str, float]:
    if not scores:
        return {}
    if tradable is None:
        tradable = frozenset({s.value for s in TRADABLE_THEME_STATES})
    else:
        tradable = frozenset(str(x) for x in tradable)
    result: dict[str, float] = {}
    for ticker, val in scores.items():
        state = theme_states.get(ticker) if isinstance(theme_states, Mapping) else None
        if state is None:
            continue
        state_str = str(state)
        if state_str in tradable:
            result[ticker] = float(val)
    return result


class SectorLeadershipModel:
    name: str = "M07"

    def __init__(
        self,
        master: InstrumentMaster,
        resolver: ClusterResolver,
        weights: SectorScoreWeights,
        transition_config: TransitionConfig,
        history: pl.DataFrame | None = None,
    ) -> None:
        self._master = master
        self._resolver = resolver
        self._weights = weights
        self._config = transition_config
        self._history = history
        self._theme_state: dict[str, ThemeState] = {}
        self._patience: dict[str, int] = {}
        self._metrics_history: dict[str, list[StateMetrics]] = {}
        self._rep_by_theme: dict[str, str] = {}

    def theme_states_by_representative(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for theme, state in self._theme_state.items():
            rep = self._rep_by_theme.get(theme)
            if rep is not None:
                out[rep] = state.value if hasattr(state, "value") else str(state)
        return out

    def replay_theme_state(self, theme: str) -> list[ThemeState]:
        return run_state_machine(self._metrics_history.get(theme, []), self._config, initial=ThemeState.DISCOVERY)

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
        if snapshot.height == 0:
            return {}
        required = ["mom_20", "mom_5", "ma_20", "close"]
        for col in required:
            if col not in snapshot.columns:
                return {}
        if "ticker" not in snapshot.columns:
            return {}
        # validate exactly once per invocation
        self._config.validate()
        # single O(N) index build per score() call
        index: SnapshotIndex = build_snapshot_index(snapshot)
        # fast fail-closed if no ticker has all required non-null values; use index to avoid extra snapshot.filter
        has_complete = False
        for m in index.values():
            if all(m.get(c) is not None for c in required):
                has_complete = True
                break
        if not has_complete and snapshot.height > 0:
            return {}
        decision_date: date = context.decision_date
        try:
            choices = self._resolver.resolve_indexed(index, decision_date)
        except Exception:
            return {}
        if not choices:
            return {}
        history = self._history if self._history is not None else pl.DataFrame()
        try:
            panel: ThemePanel = build_theme_panel(snapshot, history, choices, decision_date, self._weights)
        except Exception:
            return {}
        result: dict[str, float] = {}
        for theme in panel.themes():
            metrics = panel.metrics_for(theme)
            if metrics is None:
                continue
            state_metrics = StateMetrics(
                theme=metrics.theme,
                representative=metrics.representative,
                rs=metrics.rs,
                accel=metrics.accel,
                breadth=metrics.breadth,
                ext=metrics.ext,
                dd=metrics.dd,
            )
            cur_state = self._theme_state.get(theme, ThemeState.DISCOVERY)
            cur_pat = self._patience.get(theme, 0)
            nxt, nxt_pat = transition(cur_state, state_metrics, self._config, patience_counter=cur_pat, validate=False)
            self._theme_state[theme] = nxt
            self._patience[theme] = nxt_pat
            self._rep_by_theme[theme] = metrics.representative
            self._metrics_history.setdefault(theme, []).append(state_metrics)
            score_val = self._weights.sector_score(metrics.rs, metrics.accel, metrics.breadth)
            result[metrics.representative] = float(score_val)
        # INV-13-6: filter via theme state before return
        return filter_scores_by_theme_state(result, self.theme_states_by_representative())
