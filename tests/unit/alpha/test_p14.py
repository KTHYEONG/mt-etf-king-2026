from __future__ import annotations

import inspect
from datetime import date

import polars as pl
import pytest

from src.strategies.registry import STRATEGIES as BASELINES
from src.alpha.base import DecisionContext
from src.alpha.intensity import FamilyIntensityModel
from src.portfolio.selection import family_canonical_scores
from src.universe.instruments import Confidence, InstrumentAttributes, InstrumentMaster
from src.universe.tournament import TournamentRules


def _make_attr(ticker: str, family_key: str, leverage: int) -> InstrumentAttributes:
    d = date(2024, 1, 2)
    return InstrumentAttributes(
        ticker=ticker,
        name=ticker,
        issuer="삼성자산운용",
        leverage_multiple=leverage,
        leverage_family_key=family_key,
        is_synthetic=False,
        is_hedged=False,
        is_active=True,
        index_key="kospi 200",
        theme="EQUITY",
        first_seen=d,
        last_seen=d,
        left_censored=False,
        confidence=Confidence.HIGH,
    )


def test_p14_registered_in_baselines() -> None:
    assert "portfolio.lottery_exposure" in BASELINES
    model = BASELINES["portfolio.lottery_exposure"]()
    assert model.name == "portfolio.lottery_exposure"  # type: ignore[attr-defined]
    assert model.scores_path_independent is True  # type: ignore[attr-defined]
    assert hasattr(model, "score")
    assert hasattr(model, "allocate")
    assert "portfolio.leadership_confidence" in BASELINES
    assert BASELINES["portfolio.leadership_confidence"]().name == "portfolio.leadership_confidence"  # type: ignore[attr-defined]
    # lottery_config enabled is True on the policy
    lc = getattr(model, "lottery_config", None)
    assert lc is not None
    assert lc.enabled is True  # type: ignore[attr-defined]


def test_p14_score_matches_p13_intensity_canonical() -> None:
    attrs = {
        "C1": _make_attr("C1", "FAM", 1),
        "C2": _make_attr("C2", "FAM", 2),
    }
    master = InstrumentMaster(attributes=attrs, panel_start=date(2024, 1, 2))
    snap = pl.DataFrame({"ticker": ["C1", "C2"], "mom_20": [0.10, 0.22]})
    ctx = DecisionContext(
        decision_date=date(2024, 6, 1),
        regime=None,
        capital=1_000_000_000.0,
        held={},
        rules=TournamentRules.from_yaml(__import__("pathlib").Path("configs/tournament.yaml")),
    )
    direct = FamilyIntensityModel(master).score(snap, ctx)
    # family_intensity_scores then family_canonical_scores returns {'C1': approx 0.22}
    filtered = family_canonical_scores(direct, master)
    assert filtered == {"C1": pytest.approx(0.22)}

    # P14 score path must not use filter_scores_by_theme_state
    import src.portfolio.builders_convexity as baselines_mod

    src = inspect.getsource(baselines_mod.make_portfolio_lottery_exposure)  # type: ignore[attr-defined]
    assert "family_intensity_scores" in src or "FamilyIntensityModel" in src
    assert "family_canonical_scores" in src
    assert "filter_scores_by_theme_state" not in src

    # BASELINES['portfolio.leadership_confidence'] still has name P13
    p13_model = BASELINES["portfolio.leadership_confidence"]()
    assert p13_model.name == "portfolio.leadership_confidence"  # type: ignore[attr-defined]
