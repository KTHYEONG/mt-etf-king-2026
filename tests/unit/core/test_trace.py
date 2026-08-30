from __future__ import annotations

from datetime import date

import polars as pl

from src.core.trace import CANDIDATE_CAP, REJECT_REASONS, CandidateTrace, GateTrace, InMemoryTraceSink, NullTraceSink, SessionTrace
from src.reporting.trace_store import frames_from_sink


def test_null_trace_sink_enabled_false() -> None:
    sink = NullTraceSink()
    assert sink.enabled is False
    # no-ops do not raise
    sink.emit_session(SessionTrace(decision_date=date(2026, 1, 2)))
    sink.emit_candidates([CandidateTrace(decision_date=date(2026, 1, 2), ticker="069500")])
    sink.emit_gate(GateTrace(decision_date=date(2026, 1, 2), gate="EMPTY_SCORES", exc_type=""))
    mem = InMemoryTraceSink()
    assert mem.enabled is True
    mem.emit_session(SessionTrace(decision_date=date(2026, 1, 2)))
    assert len(mem.sessions) == 1
    assert frozenset({"FAMILY_DEDUP", "THEME_DEDUP", "TOPK_CUT", "SIZING_DROP", "ADV_CAP", "UNFILLED"}) == REJECT_REASONS  # noqa: SIM300
    assert CANDIDATE_CAP == 200


def test_in_memory_trace_sink_records_three_streams() -> None:
    sink = InMemoryTraceSink()
    sink.emit_session(SessionTrace(decision_date=date(2026, 1, 2)))
    sink.emit_session(SessionTrace(decision_date=date(2026, 1, 3)))
    sink.emit_candidates(
        [
            CandidateTrace(decision_date=date(2026, 1, 2), ticker="069500"),
            CandidateTrace(decision_date=date(2026, 1, 2), ticker="114800"),
            CandidateTrace(decision_date=date(2026, 1, 3), ticker="069500"),
        ]
    )
    sink.emit_gate(GateTrace(decision_date=date(2026, 1, 2), gate="EMPTY_SCORES", exc_type=""))
    sessions, candidates, gates = frames_from_sink(sink)
    assert sessions.height == 2
    assert candidates.height == 3
    assert len(gates) == 1
