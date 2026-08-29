"""SCENARIO-08-13"""
from src.tournament.montecarlo import CompetitorField, rank_interval


def test_SCENARIO_08_13_rank_interval() -> None:  # noqa: N802
    scenarios = {"aggressive": 0.72, "normal": 0.478, "weak": 0.30}
    out = rank_interval(0.5, scenarios, n_competitors=1000)
    assert set(out.keys()) >= {"aggressive", "normal", "weak"}
    for k in ("aggressive", "normal", "weak"):
        lo, hi = out[k]
        assert isinstance(lo, int)  # noqa: PT018
        assert isinstance(hi, int)  # noqa: PT018
        assert lo <= hi  # noqa: PT018
    assert "win_probability" not in out
    # also test CompetitorField returns interval no scalar
    cf = CompetitorField(scenarios)
    out2 = cf.rank_interval(0.5)
    assert "win_probability" not in out2
    for v in out2.values():
        assert isinstance(v, tuple)  # noqa: PT018
        assert len(v) == 2  # noqa: PT018


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-13"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-13":
        test_SCENARIO_08_13_rank_interval()
