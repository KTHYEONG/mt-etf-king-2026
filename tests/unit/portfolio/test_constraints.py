"""SCENARIO-08-05 SCENARIO-08-15 SCENARIO-09-04"""
from src.portfolio.constraints import apply_gross_exposure_cap, apply_liquidity_cap, gross_exposure, rebalance_band


def test_SCENARIO_08_05_liquidity_cap() -> None:  # noqa: N802
    out = apply_liquidity_cap({"T": 0.90}, {"T": 1e10}, capital=1e9, participation=0.01)
    assert out["T"] <= 0.10 + 1e-6


def test_SCENARIO_08_15_rebalance_band() -> None:  # noqa: N802
    out = rebalance_band(target={"A": 0.55}, current={"A": 0.52}, min_delta=0.05)
    assert abs(out["A"] - 0.52) < 1e-9
    # also test that large delta triggers trade
    out2 = rebalance_band(target={"A": 0.60}, current={"A": 0.52}, min_delta=0.05)
    assert abs(out2["A"] - 0.60) < 1e-9


import pytest


def test_SCENARIO_09_04_gross_exposure_cap() -> None:  # noqa: N802
    assert gross_exposure({"A": 0.8}, {"A": 2}) == 1.6
    capped = apply_gross_exposure_cap({"A": 0.9}, {"A": 2}, max_gross=1.60)
    assert gross_exposure(capped, {"A": 2}) <= 1.60 + 1e-9


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-05", "SCENARIO-08-15", "SCENARIO-09-04"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-05":
        test_SCENARIO_08_05_liquidity_cap()
    if scenario_id == "SCENARIO-08-15":
        test_SCENARIO_08_15_rebalance_band()
    if scenario_id == "SCENARIO-09-04":
        test_SCENARIO_09_04_gross_exposure_cap()


def test_rebalance_band_suppresses_only_small_active_resize() -> None:
    from src.portfolio.constraints import rebalance_band

    result = rebalance_band(
        target={"KEEP": 0.55, "ENTER": 0.04},
        current={"KEEP": 0.50, "EXIT": 0.04},
        min_delta=0.10,
    )

    assert result == {"KEEP": 0.50, "ENTER": 0.04}


def test_rebalance_band_rejects_invalid_threshold() -> None:
    import pytest

    from src.portfolio.constraints import rebalance_band

    for threshold in (-0.01, float("nan")):
        with pytest.raises(ValueError, match="min_delta"):
            rebalance_band(target={"A": 0.5}, current={"A": 0.4}, min_delta=threshold)


def test_load_rebalance_threshold_reads_nested_value_and_fails_closed(tmp_path) -> None:
    import pytest

    from src.portfolio.constraints import load_rebalance_threshold

    valid = tmp_path / "valid.yaml"
    valid.write_text("portfolio:\n  rebalance_threshold: 0.10\n", encoding="utf-8")
    assert load_rebalance_threshold(valid) == pytest.approx(0.10)

    missing = tmp_path / "missing.yaml"
    missing.write_text("portfolio: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rebalance_threshold"):
        load_rebalance_threshold(missing)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("portfolio:\n  rebalance_threshold: 1.10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rebalance_threshold"):
        load_rebalance_threshold(invalid)


def test_effective_weight_cap_combines_all_portfolio_limits() -> None:
    from pathlib import Path

    from src.portfolio.constraints import load_effective_weight_cap

    cap = load_effective_weight_cap(Path('configs/portfolio.yaml'), leverage_multiple=2)

    assert cap == 0.80


def test_load_p26_exposure_limits_from_config() -> None:
    from pathlib import Path

    from src.portfolio.constraints import load_p26_exposure_limits, load_portfolio_exposure_limits

    single, gross, cash = load_p26_exposure_limits(Path("configs/strategies.yaml"))
    assert abs(single - 0.95) < 1e-12
    assert abs(gross - 1.90) < 1e-12
    assert abs(cash - 0.05) < 1e-12
    g_single, g_gross, g_cash = load_portfolio_exposure_limits(Path("configs/portfolio.yaml"))
    assert abs(g_single - 0.80) < 1e-12
    assert abs(g_gross - 1.60) < 1e-12
    assert abs(g_cash - 0.05) < 1e-12


def test_load_p26_exposure_limits_fail_closed(tmp_path) -> None:
    from pathlib import Path

    from src.portfolio.constraints import load_p26_exposure_limits, load_portfolio_exposure_limits

    missing = tmp_path / "nope.yaml"
    fallback = load_p26_exposure_limits(missing)
    global_limits = load_portfolio_exposure_limits(Path("configs/portfolio.yaml"))
    assert fallback == global_limits
    bad = tmp_path / "bad.yaml"
    bad.write_text("portfolio:\n  p26:\n    max_single_weight: nan\n    max_gross_exposure: 1.90\n    min_cash: 0.05\n", encoding="utf-8")
    assert load_p26_exposure_limits(bad) == global_limits


def test_apply_portfolio_exposure_limits_collapses_switch_overlap() -> None:
    from src.portfolio.constraints import apply_portfolio_exposure_limits, gross_exposure

    leftover = {"OLD": 0.271665, "NEW": 0.80}
    multiples = {"OLD": 2, "NEW": 2}
    assert gross_exposure(leftover, multiples) > 1.90
    capped = apply_portfolio_exposure_limits(
        leftover,
        multiples,
        max_single_weight=0.95,
        max_gross_exposure=1.90,
        min_cash=0.05,
    )
    assert gross_exposure(capped, multiples) <= 1.90 + 1e-9
    assert sum(capped.values()) <= 0.95 + 1e-9
    tight = apply_portfolio_exposure_limits(
        leftover,
        multiples,
        max_single_weight=0.80,
        max_gross_exposure=1.60,
        min_cash=0.05,
    )
    assert gross_exposure(tight, multiples) <= 1.60 + 1e-9
