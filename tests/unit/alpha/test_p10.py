"""SCENARIO-10-05"""
from datetime import date

import polars as pl
import pytest

from src.alpha.baselines import BASELINES
from src.alpha.base import DecisionContext
from src.portfolio.policy import PortfolioPolicy
from src.portfolio.selection import family_canonical_scores
from src.portfolio.sizing import ConfidenceSizingConfig
from src.tournament.simulator import model_requires_path_dependent
from src.universe.instruments import InstrumentAttributes, InstrumentMaster
from src.universe.taxonomy import Taxonomy


def test_SCENARIO_11_04_scores_path_independent() -> None:  # noqa: N802
    """SCENARIO-11-04"""
    p08 = BASELINES["P08"]()
    p10 = BASELINES["P10"]()
    assert getattr(p08, "scores_path_independent", False) is True
    assert getattr(p10, "scores_path_independent", False) is True
    assert model_requires_path_dependent(p08) is True
    assert model_requires_path_dependent(p10) is True


def test_SCENARIO_10_05_p10_score_and_allocate() -> None:
    assert "P10" in BASELINES
    model = BASELINES["P10"]()
    assert hasattr(model, "score")
    assert hasattr(model, "allocate")
    panel = pl.DataFrame([
        {"date": date(2026, 1, 2), "ticker": "A1", "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
        {"date": date(2026, 1, 2), "ticker": "A2", "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
    ])
    tax = Taxonomy(rules=[])
    master = InstrumentMaster.build(panel, tax, {})
    a = master.attributes["A1"]
    b = master.attributes["A2"]
    fam = a.leverage_family_key
    master2 = InstrumentMaster(
        attributes={
            "A1": a,
            "A2": InstrumentAttributes(
                ticker=b.ticker,
                name=b.name,
                issuer=b.issuer,
                leverage_multiple=2,
                leverage_family_key=fam,
                is_synthetic=False,
                is_hedged=False,
                is_active=True,
                index_key=b.index_key,
                theme=b.theme,
                first_seen=b.first_seen,
                last_seen=b.last_seen,
                left_censored=b.left_censored,
                confidence=b.confidence,
            ),
        },
        panel_start=master.panel_start,
    )
    canonical = family_canonical_scores({"A1": 0.1, "A2": 0.9}, master2)
    assert set(canonical.keys()) == {"A1"}
    snap = pl.DataFrame({
        "ticker": ["A1", "A2"],
        "mom_20": [0.5, 0.9],
        "close": [100, 100],
        "date": [date(2026, 1, 2), date(2026, 1, 2)],
    })
    ctx = DecisionContext(decision_date=date(2026, 1, 2), regime=None, capital=1e9, held={}, rules=None)
    from src.alpha.baselines import TopKMomentum

    b1 = TopKMomentum(horizon=20, name="P10")
    raw = b1.score(snap, ctx)
    filtered = family_canonical_scores(raw, master2)
    assert "A2" not in filtered
    pol = PortfolioPolicy(sizing_config=ConfidenceSizingConfig(), master=master2, state_enabled=False)
    dec = pol.allocate({"A1": 0.9}, regime="RISK_ON", leverage_allowed=True)
    assert dec.vehicles.get("A1") == "A2"
    assert "A2" in dec.weights


@pytest.mark.parametrize("scenario_id", ["SCENARIO-10-05", "SCENARIO-11-04"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:
    if scenario_id == "SCENARIO-10-05":
        test_SCENARIO_10_05_p10_score_and_allocate()
    elif scenario_id == "SCENARIO-11-04":
        test_SCENARIO_11_04_scores_path_independent()
