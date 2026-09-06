"""Facade fidelity for the src tournament simulator rolling split (P5)."""

from __future__ import annotations


def test_simulator_rolling_canonical_home() -> None:
    import src.tournament.simulator as facade
    from src.tournament.simulator_rolling import RollingDiagnostics, RollingResult, TournamentSimulator

    assert facade.TournamentSimulator is TournamentSimulator
    assert facade.RollingDiagnostics is RollingDiagnostics
    assert facade.RollingResult is RollingResult
    assert TournamentSimulator.__module__ == "src.tournament.simulator_rolling"
