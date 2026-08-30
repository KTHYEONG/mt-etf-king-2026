from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from src.alpha.base import DecisionContext
from src.alpha.intensity import FamilyIntensityConfig, FamilyIntensityModel, family_intensity_scores
from src.universe.instruments import Confidence, InstrumentAttributes, InstrumentMaster
from src.universe.tournament import TournamentRules


def _make_attr(
    ticker: str,
    family_key: str,
    leverage: int,
    *,
    is_synthetic: bool = False,
    confidence: Confidence = Confidence.HIGH,
    theme: str = "EQUITY",
    index_key: str = "kospi 200",
) -> InstrumentAttributes:
    d = date(2024, 1, 2)
    return InstrumentAttributes(
        ticker=ticker,
        name=ticker,
        issuer="삼성자산운용",
        leverage_multiple=leverage,
        leverage_family_key=family_key,
        is_synthetic=is_synthetic,
        is_hedged=False,
        is_active=True,
        index_key=index_key,
        theme=theme,
        first_seen=d,
        last_seen=d,
        left_censored=False,
        confidence=confidence,
    )


def _master(attrs: dict[str, InstrumentAttributes]) -> InstrumentMaster:
    return InstrumentMaster(attributes=attrs, panel_start=date(2024, 1, 2))


def test_family_intensity_promotes_2x_mom_onto_1x_key() -> None:
    attrs = {
        "C1": _make_attr("C1", "FAM", 1),
        "C2": _make_attr("C2", "FAM", 2),
    }
    master = _master(attrs)
    snapshot = pl.DataFrame({"ticker": ["C1", "C2"], "mom_20": [0.10, 0.22]})
    out = family_intensity_scores(snapshot, master)
    assert set(out.keys()) == {"C1"}
    assert "C2" not in out
    assert math.isclose(out["C1"], 0.22, abs_tol=1e-12)
    assert not math.isclose(out["C1"], 0.10, abs_tol=1e-12)


def test_family_intensity_ranks_b1_family_ahead_of_weaker_1x_rival() -> None:
    attrs = {
        "A1": _make_attr("A1", "FAM_A", 1),
        "A2": _make_attr("A2", "FAM_A", 2),
        "B1": _make_attr("B1", "FAM_B", 1),
    }
    master = _master(attrs)
    snapshot = pl.DataFrame({"ticker": ["A1", "A2", "B1"], "mom_20": [0.05, 0.12, 0.08]})
    out = family_intensity_scores(snapshot, master)
    assert set(out.keys()) == {"A1", "B1"}
    assert out["A1"] == pytest.approx(0.12)
    assert out["B1"] == pytest.approx(0.08)
    # intensity promotes A family ahead; canonical-only would pick B1
    assert out["A1"] > out["B1"]
    best = max(out, key=lambda k: out[k])
    assert best == "A1"


def test_family_intensity_ignores_inverse_and_synthetic() -> None:
    attrs = {
        "F1": _make_attr("F1", "FAM", 1),
        "F2": _make_attr("F2", "FAM", 2, is_synthetic=True),
        "F3": _make_attr("F3", "FAM", -2),
    }
    master = _master(attrs)
    snapshot = pl.DataFrame({"ticker": ["F1", "F2", "F3"], "mom_20": [0.04, 0.90, 0.50]})
    out = family_intensity_scores(snapshot, master)
    assert out == {"F1": pytest.approx(0.04)}
    # second call with exclude_synthetic False should allow F2 to contribute
    cfg = FamilyIntensityConfig(exclude_synthetic=False)
    out2 = family_intensity_scores(snapshot, master, cfg)
    # when allowing synthetic, intensity becomes 0.90
    assert out2["F1"] == pytest.approx(0.90)


def test_family_intensity_skips_low_confidence_source() -> None:
    attrs = {
        "F1": _make_attr("F1", "FAM", 1, confidence=Confidence.HIGH),
        "F2": _make_attr("F2", "FAM", 2, confidence=Confidence.LOW),
    }
    master = _master(attrs)
    snapshot = pl.DataFrame({"ticker": ["F1", "F2"], "mom_20": [0.03, 0.40]})
    out = family_intensity_scores(snapshot, master)
    assert out == {"F1": pytest.approx(0.03)}
    # LOW excluded by default
    assert out["F1"] != pytest.approx(0.40)


def test_family_intensity_no_plus1_emits_min_abs_nonsynthetic() -> None:
    attrs = {
        "X1": _make_attr("X1", "FAM_X", 2, is_synthetic=False),
        "X2": _make_attr("X2", "FAM_X", 2, is_synthetic=True),
    }
    master = _master(attrs)
    snapshot = pl.DataFrame({"ticker": ["X1", "X2"], "mom_20": [0.20, 0.90]})
    out = family_intensity_scores(snapshot, master)
    assert out == {"X1": pytest.approx(0.20)}
    # empty snapshot
    empty = pl.DataFrame({"ticker": [], "mom_20": []})
    assert family_intensity_scores(empty, master) == {}
    # missing mom_20 column
    missing = pl.DataFrame({"ticker": ["X1"], "other": [0.1]})
    assert family_intensity_scores(missing, master) == {}
    # all-null mom
    all_null = pl.DataFrame({"ticker": ["X1", "X2"], "mom_20": [None, None]})
    assert family_intensity_scores(all_null, master) == {}


def test_family_intensity_fail_closed_orphans_and_config() -> None:
    attrs = {
        "C1": _make_attr("C1", "FAM", 1),
        "C2": _make_attr("C2", "FAM", 2),
    }
    master = _master(attrs)
    snapshot = pl.DataFrame({"ticker": ["C1", "C2", "Z"], "mom_20": [0.10, 0.22, 0.50]})
    out = family_intensity_scores(snapshot, master)
    assert "Z" not in out
    assert set(out.keys()) == {"C1"}
    # config defaults
    cfg = FamilyIntensityConfig.from_yaml({})
    assert cfg.mom_col == "mom_20"
    assert cfg.allowed_multiples == (1, 2)
    assert cfg.exclude_inverse is True
    assert cfg.exclude_synthetic is True
    assert cfg.exclude_low_confidence is True
    # empty allowed_multiples raises
    with pytest.raises(ValueError, match="allowed_multiples"):
        FamilyIntensityConfig.from_yaml({"allowed_multiples": []})
    # Model.name == 'M13' and delegation
    model = FamilyIntensityModel(master)
    assert model.name == "M13"
    ctx = DecisionContext(
        decision_date=date(2024, 6, 1),
        regime=None,
        capital=1_000_000_000.0,
        held={},
        rules=TournamentRules.from_yaml(__import__("pathlib").Path("configs/tournament.yaml")),
    )
    snap2 = pl.DataFrame({"ticker": ["C1", "C2"], "mom_20": [0.10, 0.22]})
    direct = family_intensity_scores(snap2, master)
    via_model = model.score(snap2, ctx)
    assert via_model == direct
    # unknown yaml keys ignored
    cfg2 = FamilyIntensityConfig.from_yaml({"mom_col": "mom_20", "unknown_key": 123})
    assert cfg2.mom_col == "mom_20"
