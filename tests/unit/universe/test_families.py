from __future__ import annotations

from src.universe.families import LeverageFamilyIndex, resolve_family_key
from src.universe.instruments import Confidence, InstrumentAttributes


def _attr(ticker: str, family_key: str, lev: int, index_key: str) -> InstrumentAttributes:
    from datetime import date

    return InstrumentAttributes(
        ticker=ticker,
        name=ticker,
        issuer="삼성자산운용",
        leverage_multiple=lev,
        leverage_family_key=family_key,
        is_synthetic=False,
        is_hedged=False,
        is_active=True,
        index_key=index_key,
        theme="KOSPI200",
        first_seen=date(2026, 8, 13),
        last_seen=date(2026, 8, 27),
        left_censored=True,
        confidence=Confidence.HIGH,
    )


def test_scenario_04_09_leverage_family_index() -> None:
    """SCENARIO-04-09"""
    assert resolve_family_key("kospi200") == "kospi200"
    assert resolve_family_key("kospi200_futures", {"kospi200_futures": "kospi200"}) == "kospi200"

    attrs = {
        "069500": _attr("069500", "kospi200", 1, "kospi200"),
        "233740": _attr("233740", "kospi200", 2, "kospi200"),
        "999999": _attr("999999", "other", 1, "other"),
    }
    index = LeverageFamilyIndex.build(attrs)
    family = index.get("kospi200")
    assert family is not None
    assert len(family.members) == 2
    assert family.members[0].leverage_multiple == 1
    assert family.members[1].leverage_multiple == 2
    assert index.get("other") is not None
    assert len(index.get("other").members) == 1  # type: ignore[union-attr]


def test_scenario_04_20_universe_allows_multiple_family_members() -> None:
    """SCENARIO-04-20: INV-17 — universe는 family membership만 제공, 동시 보유 제한은 portfolio 책임."""
    from datetime import date

    from src.universe.provider import UniverseFilters, UniverseMode
    from tests.unit.universe.conftest import build_universe, make_panel, panel_row

    day = date(2026, 8, 27)
    panel = make_panel(
        [
            panel_row(day=day, ticker="069500", name="KODEX 200", index="코스피 200"),
            panel_row(day=day, ticker="233740", name="KODEX 레버리지", index="코스피 200"),
        ]
    )
    universe, master, _ = build_universe(panel, adv_window=1)
    family = master.attributes["069500"].leverage_family_key
    assert master.attributes["233740"].leverage_family_key == family
    filt = UniverseFilters(mode=UniverseMode.STRUCTURAL, warmup_sessions=1, max_order_to_adv=0.05)
    snap = universe.get(day, filt)
    assert set(snap.tickers) == {"069500", "233740"}
