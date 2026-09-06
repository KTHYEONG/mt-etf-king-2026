"""Facade fidelity for the src reporting tail-windows split (P5)."""

from __future__ import annotations


def test_tail_windows_canonical_home() -> None:
    import src.reporting.tail_forensics as facade
    from src.reporting.tail_windows import WindowAttribution, select_attribution_windows

    assert facade.WindowAttribution is WindowAttribution
    assert facade.select_attribution_windows is select_attribution_windows
    assert WindowAttribution.__module__ == "src.reporting.tail_windows"
