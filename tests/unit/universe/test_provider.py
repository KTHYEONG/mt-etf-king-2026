from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import yaml

from src.universe.instruments import load_sponsor_brand_map
from src.universe.provider import UniverseFilters, UniverseMode
from tests.unit.universe.conftest import build_universe, deploy_filters, make_panel, panel_row


def test_universe_filters_score_max_order_to_adv_loosens_gate() -> None:
    fill_only = UniverseFilters(capital=1_000_000_000, max_position_weight=1.0, max_order_to_adv=0.01)
    assert fill_only.required_adv() == 1e11
    scored = UniverseFilters(
        capital=1_000_000_000,
        max_position_weight=1.0,
        max_order_to_adv=0.01,
        score_max_order_to_adv=0.05,
    )
    assert scored.required_adv() == 2e10
    assert scored.max_order_to_adv == 0.01
    bad = UniverseFilters(capital=1_000_000_000, max_order_to_adv=0.01, score_max_order_to_adv=float("nan"))
    assert bad.required_adv() == 1e11


def test_scenario_04_06_required_adv_liquidity_filter() -> None:
    """SCENARIO-04-06"""
    filt = UniverseFilters(capital=1_000_000_000, max_position_weight=1.0, max_order_to_adv=0.01)
    assert filt.required_adv() == 1e11

    day = date(2026, 8, 27)
    panel = make_panel(
        [
            {
                "date": day,
                "ticker": "HIGH",
                "name": "KODEX 200",
                "underlying_index_name": "코스피 200",
                "is_tradable": True,
                "close": 1000.0,
                "trading_value": 2e11,
            },
            {
                "date": day,
                "ticker": "LOW",
                "name": "KODEX 채권",
                "underlying_index_name": "코스피 200",
                "is_tradable": True,
                "close": 1000.0,
                "trading_value": 1e10,
            },
        ]
    )
    universe, _, _ = build_universe(panel, adv_window=1)
    snap = universe.get(day, filt)
    assert snap.tickers == ("HIGH",)


def test_scenario_04_07_drop_counts_and_sponsor_stage() -> None:
    """SCENARIO-04-07"""
    d0 = date(2026, 8, 13)
    d1 = date(2026, 8, 27)
    panel = make_panel(
        [
            {
                "date": d0,
                "ticker": "ABSENT",
                "name": "KODEX 200",
                "underlying_index_name": "코스피 200",
                "is_tradable": True,
                "close": 1000.0,
                "trading_value": 2e12,
            },
            {
                "date": d1,
                "ticker": "GOOD",
                "name": "KODEX 200",
                "underlying_index_name": "코스피 200",
                "is_tradable": True,
                "close": 1000.0,
                "trading_value": 2e12,
            },
            {
                "date": d1,
                "ticker": "BAD_PRICE",
                "name": "KODEX 채권",
                "underlying_index_name": "코스피 200",
                "is_tradable": False,
                "close": 1000.0,
                "trading_value": 2e12,
            },
            {
                "date": d1,
                "ticker": "BAD_SPONSOR",
                "name": "WON 200",
                "underlying_index_name": "코스피 200",
                "is_tradable": True,
                "close": 1000.0,
                "trading_value": 2e12,
            },
        ]
    )
    universe, _, _ = build_universe(panel, adv_window=1)
    brand_map = load_sponsor_brand_map(Path("configs/sponsor_brands.yaml"))
    sponsor_issuers = tuple(sorted(set(brand_map.values())))
    deploy_filt = UniverseFilters(
        mode=UniverseMode.DEPLOYMENT,
        warmup_sessions=1,
        max_order_to_adv=0.05,
        issuer_whitelist=sponsor_issuers,
    )
    snap = universe.get(d1, deploy_filt)
    present = {"GOOD", "BAD_PRICE", "BAD_SPONSOR"}
    stage_drops = snap.dropped["price"] + snap.dropped["history"] + snap.dropped["sponsor"] + snap.dropped["liquidity"] + snap.dropped["eligibility"]
    assert stage_drops + len(snap.tickers) == len(present)
    assert snap.dropped["existence"] == 1
    assert snap.dropped["price"] == 1
    assert snap.dropped["sponsor"] == 1
    assert snap.tickers == ("GOOD",)

    struct_filt = UniverseFilters(mode=UniverseMode.STRUCTURAL, warmup_sessions=1, max_order_to_adv=0.05)
    struct_snap = universe.get(d1, struct_filt)
    assert struct_snap.dropped["sponsor"] == 0
    assert set(struct_snap.tickers) == {"GOOD", "BAD_SPONSOR"}


def test_scenario_04_08_truncated_panel_and_adv_tradable_only() -> None:
    """SCENARIO-04-08"""
    d0 = date(2026, 8, 13)
    d1 = date(2026, 8, 14)
    d2 = date(2026, 8, 18)
    rows = [
        {
            "date": d0,
            "ticker": "T1",
            "name": "KODEX 200",
            "underlying_index_name": "코스피 200",
            "is_tradable": True,
            "close": 1000.0,
            "trading_value": 1e12,
        },
        {
            "date": d1,
            "ticker": "T1",
            "name": "KODEX 200",
            "underlying_index_name": "코스피 200",
            "is_tradable": False,
            "close": 1000.0,
            "trading_value": 1.0,
        },
        {
            "date": d2,
            "ticker": "T1",
            "name": "KODEX 200",
            "underlying_index_name": "코스피 200",
            "is_tradable": True,
            "close": 1000.0,
            "trading_value": 1e12,
        },
        {
            "date": date(2026, 8, 19),
            "ticker": "T1",
            "name": "KODEX 200",
            "underlying_index_name": "코스피 200",
            "is_tradable": True,
            "close": 1000.0,
            "trading_value": 1.0,
        },
    ]
    full_panel = make_panel(rows)
    truncated = full_panel.filter(pl.col("date") <= d2)
    filt = UniverseFilters(mode=UniverseMode.STRUCTURAL, warmup_sessions=1, max_order_to_adv=0.05, capital=1, max_position_weight=1.0)
    full_uni, _, _ = build_universe(full_panel, adv_window=2)
    trunc_uni, _, _ = build_universe(truncated, adv_window=2)
    assert full_uni.get(d2, filt).tickers == trunc_uni.get(d2, filt).tickers == ("T1",)

    low_panel = make_panel(
        [
            {
                "date": d0,
                "ticker": "T2",
                "name": "KODEX 200",
                "underlying_index_name": "코스피 200",
                "is_tradable": True,
                "close": 1000.0,
                "trading_value": 1e12,
            },
            {
                "date": d1,
                "ticker": "T2",
                "name": "KODEX 200",
                "underlying_index_name": "코스피 200",
                "is_tradable": True,
                "close": 1000.0,
                "trading_value": 1.0,
            },
            {
                "date": d2,
                "ticker": "T2",
                "name": "KODEX 200",
                "underlying_index_name": "코스피 200",
                "is_tradable": True,
                "close": 1000.0,
                "trading_value": 1.0,
            },
        ]
    )
    high_uni, _, _ = build_universe(low_panel, adv_window=2)
    snap = high_uni.get(d2, filt)
    assert snap.tickers == ()


def test_scenario_04_10_deployment_vs_structural_sponsor() -> None:
    """SCENARIO-04-10"""
    day = date(2026, 8, 27)
    panel = make_panel(
        [
            {
                "date": day,
                "ticker": "069500",
                "name": "KODEX 200",
                "underlying_index_name": "코스피 200",
                "is_tradable": True,
                "close": 1000.0,
                "trading_value": 2e12,
            },
            {
                "date": day,
                "ticker": "WON001",
                "name": "WON 200",
                "underlying_index_name": "코스피 200",
                "is_tradable": True,
                "close": 1000.0,
                "trading_value": 2e12,
            },
        ]
    )
    universe, _, _ = build_universe(panel, adv_window=1)
    brand_map = load_sponsor_brand_map(Path("configs/sponsor_brands.yaml"))
    sponsor_issuers = tuple(sorted(set(brand_map.values())))
    with open("configs/universe.yaml", encoding="utf-8") as f:
        uc_raw = yaml.safe_load(f) or {}
    universe_config = uc_raw["universe"] if isinstance(uc_raw, dict) and "universe" in uc_raw else uc_raw

    deploy_filt = UniverseFilters.for_mode(
        UniverseMode.DEPLOYMENT,
        universe_config,
        sponsor_issuers,
        max_order_to_adv=0.05,
        warmup_sessions=1,
    )
    assert deploy_filt.issuer_whitelist is not None
    assert len(deploy_filt.issuer_whitelist) == 10
    deploy_snap = universe.get(day, deploy_filt)
    assert deploy_snap.tickers == ("069500",)

    struct_filt = UniverseFilters.for_mode(
        UniverseMode.STRUCTURAL,
        universe_config,
        sponsor_issuers,
        max_order_to_adv=0.05,
        warmup_sessions=1,
    )
    struct_snap = universe.get(day, struct_filt)
    assert set(struct_snap.tickers) == {"069500", "WON001"}


def test_scenario_04_12_future_listing_excluded_before_first_seen() -> None:
    """SCENARIO-04-12: INV-4 — 미래 상장 종목은 as_of 이전 유니버스에 없음."""
    d0 = date(2026, 8, 13)
    d1 = date(2026, 8, 14)
    panel = make_panel(
        [
            panel_row(day=d0, ticker="EARLY", name="KODEX 200"),
            panel_row(day=d1, ticker="EARLY", name="KODEX 200"),
            panel_row(day=d1, ticker="LATE", name="KODEX 채권"),
        ]
    )
    universe, _, _ = build_universe(panel, adv_window=1)
    filt = UniverseFilters(mode=UniverseMode.STRUCTURAL, warmup_sessions=1, max_order_to_adv=0.05)
    assert universe.get(d0, filt).tickers == ("EARLY",)
    assert set(universe.get(d1, filt).tickers) == {"EARLY", "LATE"}


def test_scenario_04_13_delisted_excluded_after_last_seen() -> None:
    """SCENARIO-04-13: INV-4 — 폐지(패널 소멸) 이후 as_of 유니버스에 없음."""
    d0 = date(2026, 8, 13)
    d1 = date(2026, 8, 14)
    d2 = date(2026, 8, 18)
    panel = make_panel(
        [
            panel_row(day=d0, ticker="GONE", name="KODEX 200"),
            panel_row(day=d1, ticker="GONE", name="KODEX 200"),
            panel_row(day=d2, ticker="STAY", name="KODEX 200"),
        ]
    )
    universe, _, _ = build_universe(panel, adv_window=1)
    filt = UniverseFilters(mode=UniverseMode.STRUCTURAL, warmup_sessions=1, max_order_to_adv=0.05)
    assert universe.get(d1, filt).tickers == ("GONE",)
    assert universe.get(d2, filt).tickers == ("STAY",)


def test_scenario_04_14_adv_ignores_future_trading_value() -> None:
    """SCENARIO-04-14: INV-6 — ADV는 as_of 이후 거래대금을 사용하지 않음."""
    d0 = date(2026, 8, 13)
    d1 = date(2026, 8, 14)
    d2 = date(2026, 8, 18)
    d3 = date(2026, 8, 19)
    low = 1e8
    panel = make_panel(
        [
            panel_row(day=d0, ticker="T", trading_value=low),
            panel_row(day=d1, ticker="T", trading_value=low),
            panel_row(day=d2, ticker="T", trading_value=low),
            panel_row(day=d3, ticker="T", trading_value=1e15),
        ]
    )
    filt = UniverseFilters(
        mode=UniverseMode.STRUCTURAL,
        warmup_sessions=1,
        capital=1_000_000_000,
        max_position_weight=1.0,
        max_order_to_adv=0.01,
    )
    universe, _, _ = build_universe(panel, adv_window=2)
    snap = universe.get(d2, filt)
    assert snap.tickers == ()
    assert snap.dropped["liquidity"] == 1


def test_scenario_04_15_non_sponsor_mapped_issuer_excluded() -> None:
    """SCENARIO-04-15: INV-20 — 매핑됐지만 비후원 운용사는 deployment 제외."""
    day = date(2026, 8, 27)
    brand_map = {"KODEX": "삼성자산운용", "RIVAL": "비후원자산운용"}
    panel = make_panel(
        [
            panel_row(day=day, ticker="069500", name="KODEX 200"),
            panel_row(day=day, ticker="999001", name="RIVAL 200"),
        ]
    )
    universe, _, _ = build_universe(panel, adv_window=1, brand_map=brand_map)
    deploy_filt = deploy_filters(("삼성자산운용",))
    snap = universe.get(day, deploy_filt)
    assert snap.tickers == ("069500",)
    assert snap.dropped["sponsor"] == 1


def test_scenario_04_16_manifest_overrides_sponsor_whitelist() -> None:
    """SCENARIO-04-16: INV-04-10 / INV-20 — manifest가 sponsor 단계에서 whitelist보다 우선."""
    day = date(2026, 8, 27)
    panel = make_panel([panel_row(day=day, ticker="069500", name="KODEX 200")])
    universe, _, _ = build_universe(panel, adv_window=1)
    brand_map = load_sponsor_brand_map(Path("configs/sponsor_brands.yaml"))
    sponsor_issuers = tuple(sorted(set(brand_map.values())))
    filt = deploy_filters(sponsor_issuers, manifest=frozenset({"233740"}))
    snap = universe.get(day, filt)
    assert snap.tickers == ()
    assert snap.dropped["sponsor"] == 1


def test_scenario_04_17_eligibility_uses_as_of_name_not_future() -> None:
    """SCENARIO-04-17: INV-6 — eligibility는 as_of 당일 이름 기준(미래 이름 변경 미반영)."""
    d1 = date(2026, 8, 14)
    d2 = date(2026, 8, 18)
    panel = make_panel(
        [
            panel_row(day=d1, ticker="X", name="ACE 레버리지"),
            panel_row(day=d2, ticker="X", name="KODEX 200"),
        ]
    )
    universe, _, _ = build_universe(panel, adv_window=1)
    filt = UniverseFilters(
        mode=UniverseMode.STRUCTURAL,
        warmup_sessions=1,
        max_order_to_adv=0.05,
        allow_leverage=False,
        allow_inverse=True,
    )
    snap = universe.get(d1, filt)
    assert snap.tickers == ()
    assert snap.dropped["eligibility"] == 1


def test_scenario_04_18_low_confidence_excluded_at_eligibility() -> None:
    """SCENARIO-04-18: INV-19 — LOW confidence 비1배수는 eligibility에서 fail-closed 제외."""
    day = date(2026, 8, 27)
    panel = make_panel([panel_row(day=day, ticker="BAD", name="ACE 인버스 레버리지")])
    universe, _, _ = build_universe(panel, adv_window=1)
    filt = UniverseFilters(mode=UniverseMode.STRUCTURAL, warmup_sessions=1, max_order_to_adv=0.05)
    snap = universe.get(day, filt)
    assert snap.tickers == ()
    assert snap.dropped["eligibility"] == 1
