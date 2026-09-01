def test_field_relative_report_strict_win_ties_lose() -> None:
    from src.tournament.objective import field_relative_report

    candidate = [0.50, 0.40, 0.90]
    rivals = {"A": [0.40, 0.40, 0.80], "B": [0.10, 0.50, 0.70]}
    report = field_relative_report(candidate, rivals, horizon=36)
    assert report.n_windows == 3
    assert report.n_agents == 3
    assert report.n_effective == 0
    assert abs(report.win_rate - (2.0 / 3.0)) < 1e-12
    assert abs(report.top2_rate - (2.0 / 3.0)) < 1e-12
    assert abs(report.median_rank_percentile - (1.0 / 3.0)) < 1e-12


def test_field_relative_report_rejects_length_mismatch() -> None:
    import pytest

    from src.tournament.objective import field_relative_report

    with pytest.raises(ValueError, match=r".*"):
        field_relative_report([0.1, 0.2], {"A": [0.1]}, horizon=36)
    with pytest.raises(ValueError, match=r".*"):
        field_relative_report([], {"A": []}, horizon=36)
    with pytest.raises(ValueError, match=r".*"):
        field_relative_report([0.1], {}, horizon=36)
    with pytest.raises(ValueError, match=r".*"):
        field_relative_report([0.1], {"A": [0.1]}, horizon=0)
    with pytest.raises(ValueError, match=r".*"):
        field_relative_report([float("nan")], {"A": [0.1]}, horizon=36)
