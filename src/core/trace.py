from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final, Protocol

REJECT_REASONS: Final[frozenset[str]] = frozenset(
    {"FAMILY_DEDUP", "THEME_DEDUP", "TOPK_CUT", "SIZING_DROP", "ADV_CAP", "UNFILLED"}
)

DIAGNOSTIC_FEATURE_COLS: Final[tuple[str, ...]] = (
    "mom_5",
    "mom_20",
    "mom_60",
    "mom_20_rs",
    "rv_20",
    "creation_flow_5",
    "disparity",
)

CANDIDATE_CAP: Final[int] = 200


@dataclass(frozen=True)
class SessionTrace:
    decision_date: date
    n_universe: int = 0
    n_scores: int = 0
    n_selected: int = 0
    n_fills: int = 0
    n_unfilled: int = 0
    n_candidates_written: int = 0
    n_candidates_truncated: int = 0
    dropped_existence: int = 0
    dropped_price: int = 0
    dropped_history: int = 0
    dropped_sponsor: int = 0
    dropped_liquidity: int = 0
    dropped_eligibility: int = 0
    regime: str = ""
    equity: float = 0.0


@dataclass(frozen=True)
class CandidateTrace:
    decision_date: date
    ticker: str
    score: float = 0.0
    rank: int = 0
    selected: bool = False
    reject_reason: str = ""
    weight_raw: float = 0.0
    weight_target: float = 0.0
    weight_after_adv: float = 0.0
    weight_fill: float = 0.0
    # optional diagnostic fields stored as extra dict
    diagnostics: dict[str, float] | None = None


@dataclass(frozen=True)
class GateTrace:
    decision_date: date
    gate: str
    exc_type: str = ""


class TraceSink(Protocol):
    @property
    def enabled(self) -> bool: ...

    def emit_session(self, session: SessionTrace) -> None: ...

    def emit_candidates(self, candidates: list[CandidateTrace]) -> None: ...

    def emit_gate(self, gate: GateTrace) -> None: ...


class NullTraceSink:
    @property
    def enabled(self) -> bool:
        return False

    def emit_session(self, session: SessionTrace) -> None:
        return None

    def emit_candidates(self, candidates: list[CandidateTrace]) -> None:
        return None

    def emit_gate(self, gate: GateTrace) -> None:
        return None


class InMemoryTraceSink:
    def __init__(self) -> None:
        self.sessions: list[SessionTrace] = []
        self.candidates: list[CandidateTrace] = []
        self.gates: list[GateTrace] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit_session(self, session: SessionTrace) -> None:
        self.sessions.append(session)

    def emit_candidates(self, candidates: list[CandidateTrace]) -> None:
        self.candidates.extend(candidates)

    def emit_gate(self, gate: GateTrace) -> None:
        self.gates.append(gate)
