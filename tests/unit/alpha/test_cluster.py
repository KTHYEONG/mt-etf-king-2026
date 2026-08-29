"""SCENARIO-07-01 SCENARIO-07-02 SCENARIO-07-07"""
from __future__ import annotations

from datetime import date

import polars as pl

from src.alpha.cluster import ClusterResolver, select_representative
from tests.unit.alpha.conftest import make_attr, make_master


def test_SCENARIO_07_01_cluster_l1_dedup() -> None:  # noqa: N802
    """SCENARIO-07-01"""
    attrs = {
        "111": make_attr("111"),
        "222": make_attr("222"),
        "333": make_attr("333"),
    }
    snapshot = pl.DataFrame(
        [
            {"ticker": "111", "trading_value": 1_000_000_000.0},
            {"ticker": "222", "trading_value": 2_000_000_000.0},
            {"ticker": "333", "trading_value": 3_000_000_000.0},
        ]
    )
    resolver = ClusterResolver(make_master(attrs), max_per_theme=2)
    result = resolver.resolve(snapshot, date(2024, 6, 1))
    assert len(result) == 1
    assert all(c.index_key == "kospi 200" for c in result)


def test_SCENARIO_07_02_select_representative_liquidity() -> None:  # noqa: N802
    """SCENARIO-07-02"""
    attrs = {
        "AAA": make_attr("AAA"),
        "BBB": make_attr("BBB"),
    }
    candidates = pl.DataFrame(
        [
            {"ticker": "AAA", "trading_value": 1_000_000_000.0, "tracking_error": 0.01},
            {"ticker": "BBB", "trading_value": 5_000_000_000.0, "tracking_error": 0.01},
        ]
    )
    selected = select_representative(candidates, attrs, lookback=20)
    assert selected == "BBB"


def test_SCENARIO_07_07_exclude_leverage_from_representative() -> None:  # noqa: N802
    """SCENARIO-07-07"""
    attrs = {
        "LEV": make_attr("LEV", leverage=2),
        "SPOT": make_attr("SPOT", leverage=1),
    }
    candidates = pl.DataFrame(
        [
            {"ticker": "LEV", "trading_value": 9_000_000_000.0, "tracking_error": 0.01},
            {"ticker": "SPOT", "trading_value": 1_000_000_000.0, "tracking_error": 0.01},
        ]
    )
    selected = select_representative(candidates, attrs, lookback=20)
    assert selected == "SPOT"
    assert attrs[selected].leverage_multiple == 1
