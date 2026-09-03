from __future__ import annotations


def test_compute_slice_metrics_basic() -> None:
    from src.tournament.loyo import compute_slice_metrics

    rets = [0.6, 0.1, -0.3, 0.55, -0.1]
    m = compute_slice_metrics("y", rets)
    assert m.n == 5
    assert abs(m.p_gt_50 - 0.4) < 1e-12
    assert abs(m.ruin - 0.2) < 1e-12
    empty = compute_slice_metrics("e", [])
    assert empty.n == 0
    assert empty.p_gt_50 == 0.0
    assert empty.ruin == 0.0


def test_group_returns_by_year_and_era() -> None:
    from datetime import date

    import polars as pl

    from src.tournament.loyo import group_returns_by_era, group_returns_by_year

    windows = pl.DataFrame(
        {
            "window_start": [date(2018, 1, 2), date(2019, 6, 1), date(2020, 3, 1), date(2025, 9, 1)],
            "window_end": [date(2018, 2, 22), date(2019, 7, 20), date(2020, 4, 20), date(2025, 10, 21)],
            "terminal_return": [0.1, 0.2, 0.3, 0.8],
        }
    )
    by_y = group_returns_by_year(windows)
    assert by_y["2018"] == [0.1]
    assert by_y["2019"] == [0.2]
    assert by_y["2025"] == [0.8]
    by_e = group_returns_by_era(windows)
    assert abs(sum(by_e["2018_2019"]) - 0.3) < 1e-12
    assert abs(sum(by_e["2024_2026"]) - 0.8) < 1e-12


def test_concentration_share_2025_2026() -> None:
    from datetime import date

    import polars as pl

    from src.tournament.loyo import concentration_share

    windows = pl.DataFrame(
        {
            "window_start": [date(2020, 1, 2), date(2025, 1, 2), date(2026, 1, 2), date(2026, 2, 1)],
            "terminal_return": [0.6, 0.7, 0.8, 0.1],
        }
    )
    # gt_50: 2020,2025,2026 => 2/3 in 2025-26
    share = concentration_share(windows)
    assert abs(share - (2.0 / 3.0)) < 1e-12
    none = concentration_share(pl.DataFrame({"window_start": [date(2020, 1, 2)], "terminal_return": [0.1]}))
    assert none == 0.0


def test_evaluate_loyo_years_non_inferior() -> None:
    from src.tournament.loyo import evaluate_loyo_years

    cand = {
        "2019": [0.0] * 40,
        "2025": [0.6] * 40,
        "2026": [0.1] * 5,
    }
    inc = {
        "2019": [0.0] * 40,
        "2025": [0.55] * 40,
        "2026": [0.9] * 5,
    }
    rows = evaluate_loyo_years(cand, inc, min_year_n=30)
    years = {r.year: r for r in rows}
    assert "2026" not in years
    assert years["2019"].non_inferior is True
    assert years["2025"].non_inferior is True
    # degrade 2019 p50
    cand2 = {"2019": [-0.1] * 40, "2025": [0.6] * 40}
    inc2 = {"2019": [0.0] * 40, "2025": [0.55] * 40}
    rows2 = {r.year: r for r in evaluate_loyo_years(cand2, inc2, min_year_n=30)}
    assert rows2["2019"].non_inferior is False


def test_evaluate_promotion_robustness_rejects_2025_26_concentration() -> None:
    from datetime import date

    import polars as pl

    from src.tournament.loyo import evaluate_promotion_robustness, write_loyo_report

    years = list(range(2018, 2027))

    def _windows(maker) -> pl.DataFrame:
        rows = []
        for y in years:
            for i in range(40):
                rows.append({"window_start": date(y, 1, 2), "terminal_return": float(maker(y, i))})  # noqa: PERF401
        return pl.DataFrame(rows)

    # incumbent: flat zeros
    inc = _windows(lambda y, i: 0.0)
    # candidate: only 2025-26 have P>50
    cand_conc = _windows(lambda y, i: 0.6 if y >= 2025 else 0.0)
    r1 = evaluate_promotion_robustness(candidate_windows=cand_conc, incumbent_windows=inc, min_year_n=30)
    assert r1.status == "FAIL"
    assert "CONCENTRATION" in r1.failures
    assert r1.concentration_2025_2026 > 0.90

    # balanced: every year has some P>50
    cand_bal = _windows(lambda y, i: 0.6 if i < 4 else 0.0)
    r2 = evaluate_promotion_robustness(candidate_windows=cand_bal, incumbent_windows=inc, min_year_n=30)
    assert r2.concentration_2025_2026 <= 0.90
    assert "CONCENTRATION" not in r2.failures
    assert r2.status in ("PASS", "FAIL")
    from pathlib import Path

    out = write_loyo_report(Path("tmp/loyo_test"), r2)
    assert out.endswith("loyo_report.json")


def test_evaluate_promotion_robustness_diagnostic_without_incumbent() -> None:
    from datetime import date

    import polars as pl

    from src.tournament.loyo import evaluate_promotion_robustness

    windows = pl.DataFrame(
        {
            "window_start": [date(2024, 1, 2), date(2025, 1, 2), date(2026, 1, 2)],
            "terminal_return": [0.1, 0.7, 0.8],
        }
    )
    r = evaluate_promotion_robustness(candidate_windows=windows, incumbent_windows=None)
    assert r.status == "DIAGNOSTIC"
    assert r.full_incumbent is None
    assert "2025" in r.year_metrics
    assert "2024_2026" in r.era_metrics
    assert r.loyo_n_years == 0
