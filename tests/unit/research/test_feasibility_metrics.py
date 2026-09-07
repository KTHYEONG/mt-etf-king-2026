from datetime import date

import polars as pl
import pytest

from src.research.decision_population import WindowPolicy, build_decision_population
from src.research.feasibility_metrics import AlignError, align_returns_by_entry_date


def test_align_returns_by_entry_date_rejects_length_mismatch() -> None:
    sessions = [date(2026, 1, d) for d in (2, 5, 6, 7, 8)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["X"] * 5})
    pop = build_decision_population(calendar_sessions=sessions, panel=panel, horizon=2, policy=WindowPolicy.EXECUTABLE)
    with pytest.raises(AlignError):
        align_returns_by_entry_date(starts=[sessions[1]], returns=[0.1, 0.2], population=pop, require_complete=True)
    aligned = align_returns_by_entry_date(
        starts=[w.entry_date for w in pop.windows],
        returns=[0.1] * pop.n_windows,
        population=pop,
        require_complete=True,
    )
    assert aligned == (0.1,) * pop.n_windows
    with pytest.raises(AlignError):
        align_returns_by_entry_date(starts=[date(1999, 1, 1)] * pop.n_windows, returns=[0.0] * pop.n_windows, population=pop, require_complete=True)


def test_capture_intersection_uses_both_not_ratio_of_counts() -> None:
    from src.research.feasibility_metrics import CaptureError, capture_intersection

    oracle = (0.60, 0.20, 0.55)
    strategy = (0.51, 0.51, 0.10)
    report = capture_intersection(strategy, oracle, threshold=0.50)
    assert report.n_oracle == 2
    assert report.n_strategy == 2
    assert report.n_both == 1
    assert report.rate == 0.5
    assert report.n_strategy_without_oracle == 1
    with pytest.raises(CaptureError):
        capture_intersection((0.1,), (0.1, 0.2), threshold=0.50)


def test_return_space_gap_does_not_subtract_giveback_ratio() -> None:
    from src.research.feasibility_metrics import peak_return_from_wealth_giveback, return_space_gap

    oracle = (0.80,)
    terminal = (0.40,)
    peak = (0.70,)
    report = return_space_gap(oracle, terminal, peak)
    assert abs(report.mean_selection_timing - 0.10) < 1e-12
    assert abs(report.mean_giveback - 0.30) < 1e-12
    assert abs(report.mean_total - 0.40) < 1e-12
    assert abs(report.mean_selection_timing - (0.40 - 0.30)) > 1e-12 or report.mean_selection_timing == 0.10
    wealth_peak = peak_return_from_wealth_giveback(0.40, 0.30)
    assert abs(wealth_peak - 1.0) < 1e-12
    assert abs(wealth_peak - 0.70) > 1e-6


def test_label_comparator_compliance_tags_gross_violation() -> None:
    from src.research.feasibility_metrics import label_comparator_compliance

    assert label_comparator_compliance(effective_gross_max=2.0, gross_violation_count=1428, champion_gross_max=1.9) == "NON_COMPLIANT_REFERENCE"
    assert label_comparator_compliance(effective_gross_max=1.9, gross_violation_count=0, champion_gross_max=1.9) == "COMPARABLE"
    assert label_comparator_compliance(effective_gross_max=1.8, gross_violation_count=1, champion_gross_max=1.9) == "NON_COMPLIANT_REFERENCE"


def test_population_counts_match_uses_eligible_not_literal() -> None:
    from src.research.feasibility_metrics import population_counts_match

    assert population_counts_match(eligible=2088, evaluated=2088, router_reset=2088, controller=2088 * 36, shadow=2088 * 36 * 4, horizon=36) is True
    assert population_counts_match(eligible=2088, evaluated=2088, router_reset=2088, controller=2090 * 36, shadow=2090 * 36 * 4, horizon=36) is False
    assert population_counts_match(eligible=0, evaluated=0, router_reset=0, controller=0, shadow=0, horizon=36) is False
