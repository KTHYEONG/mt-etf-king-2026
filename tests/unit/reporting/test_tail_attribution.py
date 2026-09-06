"""Facade fidelity for the src reporting tail-attribution split (P5)."""

from __future__ import annotations


def test_tail_attribution_canonical_home() -> None:
    import src.reporting.tail_forensics as facade
    from src.reporting.tail_attribution import summarise_tail_attribution

    assert facade.summarise_tail_attribution is summarise_tail_attribution
    assert summarise_tail_attribution.__module__ == "src.reporting.tail_attribution"
