"""Facade fidelity for the src reporting tail-forensics split (P5)."""

from __future__ import annotations


def test_tail_miss_canonical_home() -> None:
    import src.reporting.tail_forensics as facade
    from src.reporting.tail_miss import TailMissReport, summarise_tail_miss_windows

    assert facade.TailMissReport is TailMissReport
    assert facade.summarise_tail_miss_windows is summarise_tail_miss_windows
    assert TailMissReport.__module__ == "src.reporting.tail_miss"
