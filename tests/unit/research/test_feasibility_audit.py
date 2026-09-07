from datetime import date
from pathlib import Path

import polars as pl

from src.research.feasibility_audit import run_feasibility_audit


def test_run_feasibility_audit_tags_b1_and_writes_reconciliation(tmp_path: Path) -> None:
    sessions = [date(2026, 1, d) for d in (2, 5, 6, 7, 8)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["A"] * 5, "open": [100.0, 100.0, 110.0, 120.0, 130.0], "close": [100.0, 101.0, 111.0, 121.0, 131.0]})
    opens = pl.DataFrame({"date": sessions, "ticker": ["A"] * 5, "open": [100.0, 100.0, 110.0, 120.0, 130.0], "eligible": [True] * 5})
    p27 = pl.DataFrame({"window_start": [date(2026, 1, 5), date(2026, 1, 6)], "terminal_return": [0.51, 0.10], "giveback": [0.02, 0.01]})
    b1 = pl.DataFrame({"window_start": [date(2026, 1, 5), date(2026, 1, 6)], "terminal_return": [0.02, 0.03], "giveback": [0.01, 0.01]})
    out = tmp_path / "research"
    result = run_feasibility_audit(
        calendar_sessions=sessions,
        panel=panel,
        horizon=2,
        output_dir=out,
        oracle_opens=opens,
        strategy_windows={"sticky.mom60_raw": p27, "baseline.mom20_top1": b1},
        comparator_gross={"baseline.mom20_top1": {"effective_gross_max": 2.0, "gross_violation_count": 1428}, "sticky.mom60_raw": {"effective_gross_max": 1.9, "gross_violation_count": 0}},
        champion_gross_max=1.9,
        p38_terminal_by_start=None,
        manifest=None,
        enforce_horizon_36=False,
    )
    assert result["b1_role"] == "NON_COMPLIANT_REFERENCE"
    assert result["p38_status"] == "MISSING_ARTIFACT"
    assert (out / "window_population_reconciliation.md").exists()
    assert (out / "2026_championship_feasibility_metrics.json").exists()
    assert (out / "2026_championship_feasibility_tables.csv").exists()
    md = (out / "window_population_reconciliation.md").read_text(encoding="utf-8")
    assert "calendar_naive" in md and "panel_naive" in md and "executable" in md
    assert "Eligible Windows" in md or "eligible_windows" in md or "eligible" in md.lower()
    from src.research.feasibility_metrics import peak_return_from_return_space_giveback

    p27_metrics = result["strategies"]["sticky.mom60_raw"]
    assert abs(peak_return_from_return_space_giveback(0.51, 0.02) - 0.53) < 1e-12
    assert abs(p27_metrics["mean_giveback"] - 0.015) < 1e-12


def test_run_feasibility_audit_enforce_horizon_36_guard(tmp_path: Path) -> None:
    import pytest

    from src.research.decision_population import PopulationError

    sessions = [date(2026, 1, d) for d in (2, 5, 6, 7, 8)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["A"] * 5})
    opens = pl.DataFrame({"date": sessions, "ticker": ["A"] * 5, "open": [100.0] * 5, "eligible": [True] * 5})
    with pytest.raises(PopulationError, match="horizon=36"):
        run_feasibility_audit(
            calendar_sessions=sessions,
            panel=panel,
            horizon=2,
            output_dir=tmp_path,
            oracle_opens=opens,
            strategy_windows={},
            comparator_gross={},
            champion_gross_max=1.9,
            p38_terminal_by_start=None,
            manifest=None,
            enforce_horizon_36=True,
        )
