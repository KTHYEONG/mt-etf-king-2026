"""SCENARIO-08-07 SCENARIO-08-08 SCENARIO-B2-01 SCENARIO-B2-02"""
from src.portfolio.state import PositionState, apply_state_multipliers, infer_theme_proxy, transition_position


def test_SCENARIO_08_07_cooldown() -> None:  # noqa: N802
    assert transition_position(PositionState.EXIT, "RECOVERY", 1, 3) == PositionState.WATCH
    assert transition_position(PositionState.EXIT, "RECOVERY", 3, 3) == PositionState.RE_ENTER
    assert transition_position(PositionState.WATCH, "RECOVERY", 2, 3) == PositionState.WATCH
    assert transition_position(PositionState.WATCH, "RECOVERY", 3, 3) == PositionState.RE_ENTER


def test_SCENARIO_08_08_theme_mapping() -> None:  # noqa: N802
    assert transition_position(PositionState.HOLD, "LEADING", 0, 3) == PositionState.HOLD
    assert transition_position(PositionState.HOLD, "OVERHEATED", 0, 3) == PositionState.TRIM
    assert transition_position(PositionState.HOLD, "BREAKDOWN", 0, 3) == PositionState.EXIT


def test_SCENARIO_B2_01_apply_state_multipliers() -> None:  # noqa: N802
    # SCENARIO-B2-01: apply_state_multipliers({'A':0.8,'B':0.2},{A:TRIM,B:HOLD},trim_fraction=0.5) == {'A':0.4,'B':0.2}
    w = {"A": 0.8, "B": 0.2}
    states = {"A": PositionState.TRIM, "B": PositionState.HOLD}
    out = apply_state_multipliers(w, states, trim_fraction=0.5)
    assert abs(out["A"] - 0.4) < 1e-9
    assert abs(out["B"] - 0.2) < 1e-9
    # EXIT/WATCH map to 0.0
    out2 = apply_state_multipliers({"X": 0.5, "Y": 0.3}, {"X": PositionState.EXIT, "Y": PositionState.WATCH}, trim_fraction=0.5)
    assert abs(out2["X"] - 0.0) < 1e-9
    assert abs(out2["Y"] - 0.0) < 1e-9
    # RE_ENTER -> 1.0
    out3 = apply_state_multipliers({"R": 0.6}, {"R": PositionState.RE_ENTER}, trim_fraction=0.5)
    assert abs(out3["R"] - 0.6) < 1e-9
    # sum <=1 invariant
    assert sum(out.values()) <= 1.0 + 1e-9


def test_SCENARIO_B2_02_infer_theme_proxy() -> None:  # noqa: N802
    # SCENARIO-B2-02: infer_theme_proxy for rank1 with conf>=c0/2 -> LEADING; score<=peak*(1-0.30) -> BREAKDOWN; EXIT tracker with rank<=3 score>0 -> RECOVERY
    scores_rank1 = {"A": 0.10, "B": 0.05, "C": 0.02}
    # conf = 0.05 >=0.013587 -> LEADING for rank1
    res = infer_theme_proxy("A", scores_rank1, peak_score=0.12, score_drop_pct=0.30, k=3, conf_c0=0.027174, tracker_state=None)
    assert res == "LEADING"
    # score <= peak*(1-0.30) -> BREAKDOWN (A score 0.05, peak 0.10 -> 0.05 <=0.07 true)
    res2 = infer_theme_proxy("B", {"A": 0.12, "B": 0.05}, peak_score=0.10, score_drop_pct=0.30, k=3, conf_c0=0.027174, tracker_state=None)
    # B score 0.05 <=0.07 -> BREAKDOWN
    assert res2 == "BREAKDOWN"
    # EXIT tracker with rank<=3 score>0 -> RECOVERY
    scores_rec = {"A": 0.03, "B": 0.02, "C": 0.01}
    res3 = infer_theme_proxy("B", scores_rec, peak_score=0.05, score_drop_pct=0.30, k=3, conf_c0=0.027174, tracker_state=PositionState.EXIT)
    # B rank2 <=3 score>0 and tracker EXIT -> RECOVERY (ensure not BREAKDOWN because peak check first? 0.02 <=0.035? 0.05*0.7=0.035, 0.02 <=0.035 would be BREAKDOWN first, so choose peak higher to avoid)
    # Use peak not triggering breakdown
    res3b = infer_theme_proxy("B", {"A": 0.05, "B": 0.04, "C": 0.01, "D": 0.005}, peak_score=0.045, score_drop_pct=0.30, k=3, conf_c0=0.027174, tracker_state=PositionState.EXIT)
    assert res3b == "RECOVERY"
    # rank > k -> BREAKDOWN
    scores_tail = {"A": 0.10, "B": 0.09, "C": 0.08, "D": 0.07}
    assert infer_theme_proxy("D", scores_tail, peak_score=0.10, score_drop_pct=0.30, k=3, conf_c0=0.027174, tracker_state=None) == "BREAKDOWN"


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-07", "SCENARIO-08-08", "SCENARIO-B2-01", "SCENARIO-B2-02"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-07":
        test_SCENARIO_08_07_cooldown()
    if scenario_id == "SCENARIO-08-08":
        test_SCENARIO_08_08_theme_mapping()
    if scenario_id == "SCENARIO-B2-01":
        test_SCENARIO_B2_01_apply_state_multipliers()
    if scenario_id == "SCENARIO-B2-02":
        test_SCENARIO_B2_02_infer_theme_proxy()
