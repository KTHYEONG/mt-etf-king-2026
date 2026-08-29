"""SCENARIO-08-01 SCENARIO-08-02 SCENARIO-08-03 SCENARIO-08-04"""
from src.portfolio.sizing import ConfidenceSizingConfig, compute_confidence, confidence_weights


def test_SCENARIO_08_01_compute_confidence() -> None:  # noqa: N802
    assert abs(compute_confidence({"A": 0.10, "B": 0.04}) - 0.06) < 1e-9
    assert compute_confidence({"A": 0.10}) == 0.0
    assert compute_confidence({}) == 0.0


def test_SCENARIO_08_02_monotonic_w1() -> None:  # noqa: N802
    cfg = ConfidenceSizingConfig()
    # scores with conf 0.0, 0.1, 0.5
    scores0 = {"A": 0.5, "B": 0.5, "C": 0.3}
    scores1 = {"A": 0.6, "B": 0.5, "C": 0.3}
    scores2 = {"A": 1.0, "B": 0.5, "C": 0.3}
    w0 = confidence_weights(scores0, cfg)
    w1 = confidence_weights(scores1, cfg)
    w2 = confidence_weights(scores2, cfg)
    assert w2["A"] >= w1["A"] - 1e-9
    assert w1["A"] >= w0["A"] - 1e-9
    # also test fixed scores description via direct conf mapping check using same scores but varying config? we already cover monotonic


def test_SCENARIO_08_03_low_conf_dispersion() -> None:  # noqa: N802
    cfg = ConfidenceSizingConfig()
    # conf <<c0: very small diff
    scores = {"A": 0.03, "B": 0.029, "C": 0.028}
    w = confidence_weights(scores, cfg)
    assert w["A"] <= 0.40 + 1e-6
    assert sum(w.values()) <= 1.0 + 1e-9
    for t in ["A", "B", "C"]:
        assert w[t] >= 0.15 - 1e-9


def test_SCENARIO_08_04_high_conf_concentration() -> None:  # noqa: N802
    cfg = ConfidenceSizingConfig()
    scores = {"A": 1.0, "B": 0.1, "C": 0.05}
    w = confidence_weights(scores, cfg)
    assert w["A"] >= 0.85 - 1e-9


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-01", "SCENARIO-08-02", "SCENARIO-08-03", "SCENARIO-08-04"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-01":
        test_SCENARIO_08_01_compute_confidence()
    elif scenario_id == "SCENARIO-08-02":
        test_SCENARIO_08_02_monotonic_w1()
    elif scenario_id == "SCENARIO-08-03":
        test_SCENARIO_08_03_low_conf_dispersion()
    elif scenario_id == "SCENARIO-08-04":
        test_SCENARIO_08_04_high_conf_concentration()
