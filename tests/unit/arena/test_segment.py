from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.arena import segment as seg
from src.arena.panel import Panel, PanelConfig, build_panel
from src.arena.segment import SegmentConfig, SimulationDataError, load_segment_config, simulate_segments
from src.arena.simulate import SimConfig, SimResult, simulate
from src.core.config import ConfigError, load_config

PANEL_CFG = PanelConfig(lock_timeout_s=900.0, nav_divergence_max=0.10)
DAYS = [date(2026, 1, 1) + timedelta(days=i) for i in range(8)]
D0, D1, D2, D3, D4, D5, D6, D7 = DAYS
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
SIM = SimConfig("open", 15.0, 1.0, 2, LEVELS, 1e-9)
COST_BPS = 15.0
CAPITAL = 1000
CLOSES_A = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
OPENS_A = [100.0, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5]
CLOSES_B = [50.0, 50.5, 51.0, 51.5, 52.0, 52.5, 53.0, 53.5]
SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "ticker": pl.String,
    "name": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "trading_value": pl.Int64,
    "nav": pl.Float64,
    "shares_outstanding": pl.Int64,
    "underlying_index_name": pl.String,
}
TARGET_SCHEMA: dict[str, pl.DataType] = {
    "decision_date": pl.Date,
    "leg": pl.String,
    "ticker": pl.String,
    "weight": pl.Float64,
}


def _row(day: date, ticker: str, open_: float, close: float, volume: int, tv: int) -> dict[str, Any]:
    return {
        "date": day,
        "ticker": ticker,
        "name": f"{ticker} fund",
        "open": open_,
        "high": max(open_, close) * 1.01,
        "low": min(open_, close) * 0.99,
        "close": close,
        "volume": volume,
        "trading_value": tv,
        "nav": close,
        "shares_outstanding": 1000,
        "underlying_index_name": "IDX",
    }


def _rows(
    ticker: str,
    closes: Sequence[float | None],
    *,
    opens: Sequence[float] | None = None,
    volumes: Sequence[int] | None = None,
    days: Sequence[date] = DAYS,
    tv: int = 100000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, close in enumerate(closes):
        if close is None:
            continue
        open_ = close if opens is None else opens[i]
        volume = 1000 if volumes is None else volumes[i]
        rows.append(_row(days[i], ticker, open_, close, volume, tv))
    return rows


def _panel(rows: list[dict[str, Any]], days: Sequence[date] = DAYS) -> Panel:
    frame = pl.DataFrame(rows, schema=SCHEMA)
    return build_panel(frame, {r["ticker"] for r in rows}, days[0], days[-1], PANEL_CFG)


def _targets(entries: list[tuple[date, str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decision_date": [e[0] for e in entries],
            "leg": [e[1] for e in entries],
            "ticker": [e[2] for e in entries],
            "weight": [e[3] for e in entries],
        },
        schema=TARGET_SCHEMA,
    )


def _run(
    panel: Panel,
    entries: list[tuple[date, str, str, float]],
    *,
    cost_bps: float = COST_BPS,
    mode: str = "actual",
    drift: Sequence[date] = (),
) -> SimResult:
    return simulate_segments(
        panel,
        _targets(entries),
        SIM,
        cost_bps,
        capital_krw=CAPITAL,
        mode=mode,  # type: ignore[arg-type]
        drift_segments=drift,
    )


def _blocked(res: SimResult, day: date) -> list[tuple[str, str]]:
    rows = res.blocked.filter(pl.col("date") == day).sort("ticker")
    return [(r["ticker"], r["reason"]) for r in rows.iter_rows(named=True)]


def _load_with(monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict[str, Any]], None]) -> SegmentConfig:
    section: dict[str, Any] = dict(load_config("arena")["segment"])
    mutate(section)
    data: Mapping[str, Any] = {"segment": section}
    monkeypatch.setattr(seg, "load_config", lambda name: data)
    return load_segment_config()


def test_config_loads_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_segment_config() == SegmentConfig(cost_bps_per_side=6.5, stress_cost_bps=19.5)

    def _set(**changes: Any) -> Callable[[dict[str, Any]], None]:
        return lambda section: section.update(changes)

    def _drop(key: str) -> Callable[[dict[str, Any]], None]:
        return lambda section: section.pop(key)

    bad: list[Callable[[dict[str, Any]], None]] = [
        _set(extra=1),
        _set(cost_bps_per_side=-1.0),
        _set(cost_bps_per_side=True),
        _set(cost_bps_per_side=float("nan")),
        _set(stress_cost_bps=float("inf")),
        _set(stress_cost_bps=1.0),
        _drop("stress_cost_bps"),
    ]
    for mutate in bad:
        with pytest.raises(ConfigError):
            _load_with(monkeypatch, mutate)

    monkeypatch.setattr(seg, "load_config", lambda name: {"simulate": {}})
    with pytest.raises(ConfigError):
        load_segment_config()


def test_intraday_only_book_earns_only_intraday() -> None:
    panel = _panel(_rows("A", CLOSES_A, opens=OPENS_A) + _rows("B", CLOSES_B))
    res = _run(panel, [(D2, "intraday", "A", 1.0)])
    assert res.sessions[0] == D3
    assert res.gross_returns[0] == pytest.approx(CLOSES_A[3] / OPENS_A[3] - 1.0, rel=1e-12)
    assert res.turnover[0] == pytest.approx(2.0, rel=1e-12)
    assert res.gross_returns[1] == pytest.approx(CLOSES_A[4] / OPENS_A[4] - 1.0, rel=1e-12)
    assert res.weights.filter(pl.col("date") == D3).height == 0


def test_overnight_only_book_earns_only_overnight() -> None:
    panel = _panel(_rows("A", CLOSES_A, opens=OPENS_A) + _rows("B", CLOSES_B))
    res = _run(panel, [(D2, "overnight", "A", 1.0)])
    assert res.gross_returns[0] == pytest.approx(0.0, abs=1e-15)
    assert res.gross_returns[1] == pytest.approx(OPENS_A[4] / CLOSES_A[3] - 1.0, rel=1e-12)
    assert res.weights.filter(pl.col("date") == D3).height == 1
    assert res.weights.filter(pl.col("date") == D3)["weight"][0] == pytest.approx(1.0, rel=1e-12)


def test_same_book_both_legs_equals_daily_hold() -> None:
    panel = _panel(_rows("A", CLOSES_A, opens=OPENS_A) + _rows("B", CLOSES_B))
    seg_res = _run(panel, [(D2, "intraday", "A", 1.0), (D2, "overnight", "A", 1.0)])
    daily = simulate(
        panel,
        pl.DataFrame(
            {"decision_date": [D2], "ticker": ["A"], "weight": [1.0]},
            schema={
                "decision_date": pl.Date,
                "ticker": pl.String,
                "weight": pl.Float64,
            },
        ),
        SIM,
        capital_krw=CAPITAL,
    )
    assert seg_res.sessions == daily.sessions
    np.testing.assert_allclose(seg_res.gross_returns[1:], daily.gross_returns[1:], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(seg_res.net_returns[1:], daily.net_returns[1:], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(seg_res.turnover[1:], 0.0, atol=1e-12)


def test_switch_long_to_inverse_costs_both_sides() -> None:
    closes_i = [50.0, 49.0, 48.5, 48.0, 47.5, 47.0, 46.5, 46.0]
    opens_i = [50.0, 49.5, 48.0, 48.5, 47.0, 47.5, 46.0, 46.5]
    panel = _panel(_rows("L", CLOSES_A, opens=OPENS_A) + _rows("I", closes_i, opens=opens_i))
    entries = [(day, "intraday", "I", 1.0) for day in DAYS[2:6]] + [(day, "overnight", "L", 1.0) for day in DAYS[2:6]]
    res = _run(panel, entries)
    cost = COST_BPS / 10_000.0
    assert res.turnover[1:4] == pytest.approx([4.0, 4.0, 4.0], rel=1e-12)
    growth = (OPENS_A[4] / CLOSES_A[3]) * (closes_i[4] / opens_i[4])
    assert res.costs[1] == pytest.approx(growth - growth * (1.0 - cost * 2) ** 2, rel=1e-12)
    assert res.net_returns[1] == pytest.approx(growth * (1.0 - cost * 2) ** 2 - 1.0, rel=1e-12)


def test_adv_cap_applies_at_both_executions() -> None:
    panel = _panel(_rows("A", CLOSES_A, opens=OPENS_A, tv=100) + _rows("B", CLOSES_B))
    res = _run(panel, [(D2, "intraday", "A", 1.0), (D2, "overnight", "A", 1.0)])
    assert _blocked(res, D3) == [("A", "adv_cap"), ("A", "adv_cap")]
    cap = 100.0 / CAPITAL
    assert res.turnover[0] == pytest.approx(2.0 * cap, rel=1e-2)


def test_untraded_held_ticker_has_no_trade_and_uses_reference_leg() -> None:
    volumes = [1000, 1000, 1000, 1000, 0, 1000, 1000, 1000]
    panel = _panel(_rows("A", CLOSES_A, opens=OPENS_A, volumes=volumes) + _rows("B", CLOSES_B))
    res = _run(
        panel,
        [
            (D2, "intraday", "A", 1.0),
            (D2, "overnight", "A", 1.0),
            (D3, "intraday", "A", 1.0),
            (D3, "overnight", "A", 1.0),
        ],
    )
    assert res.gross_returns[1] == pytest.approx(CLOSES_A[4] / CLOSES_A[3] - 1.0, rel=1e-12)
    assert res.turnover[1] == pytest.approx(0.0, abs=1e-12)
    assert _blocked(res, D4) == []


def test_untraded_holding_exit_is_blocked_not_traded() -> None:
    volumes = [1000, 1000, 1000, 1000, 0, 1000, 1000, 1000]
    panel = _panel(_rows("A", CLOSES_A, opens=OPENS_A, volumes=volumes) + _rows("B", CLOSES_B))
    res = _run(panel, [(D2, "intraday", "A", 1.0), (D2, "overnight", "A", 1.0), (D3, "overnight", "A", 1.0)])
    assert _blocked(res, D4) == [("A", "not_traded")]
    assert res.gross_returns[1] == pytest.approx(CLOSES_A[4] / CLOSES_A[3] - 1.0, rel=1e-12)
    assert res.turnover[1] == pytest.approx(0.0, abs=1e-12)


def test_causality_future_perturbation_invariance() -> None:
    perturbed_a = [*CLOSES_A[:5], 10.0, 9.0, 8.0]
    perturbed_open = [*OPENS_A[:5], 1.0, 2.0, 3.0]
    perturbed_b = [*CLOSES_B[:5], 1.0, 1.0, 1.0]
    base = _panel(_rows("A", CLOSES_A, opens=OPENS_A) + _rows("B", CLOSES_B))
    changed = _panel(
        _rows("A", perturbed_a, opens=perturbed_open, volumes=[1000] * 5 + [1, 1, 1])
        + _rows("B", perturbed_b, volumes=[1000] * 5 + [1, 1, 1])
    )
    entries = [
        (D2, "intraday", "A", 0.5),
        (D2, "overnight", "A", 0.5),
        (D3, "intraday", "B", 1.0),
        (D3, "overnight", "B", 1.0),
    ]
    early = _run(base, entries)
    late = _run(changed, entries)
    np.testing.assert_array_equal(early.net_returns[:2], late.net_returns[:2])
    np.testing.assert_array_equal(early.costs[:2], late.costs[:2])
    np.testing.assert_array_equal(early.gross_returns[:2], late.gross_returns[:2])


def test_drift_neutral_hold_has_zero_mean_drift() -> None:
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(60)]
    rng = np.random.default_rng(5)
    closes = [100.0]
    opens = [100.0]
    for _ in range(59):
        gap = float(rng.normal(0.004, 0.03))
        intraday = 0.8 * gap + float(rng.normal(0.0, 0.02))
        opens.append(closes[-1] * (1.0 + gap))
        closes.append(opens[-1] * (1.0 + intraday))
    panel = _panel(_rows("A", closes, opens=opens, days=days), days=days)
    res = simulate_segments(
        panel,
        _targets([(days[1], "intraday", "A", 1.0), (days[1], "overnight", "A", 1.0)]),
        SIM,
        0.0,
        capital_krw=CAPITAL,
        mode="drift_neutral",
    )
    r_cc = panel.frame.filter(pl.col("date").is_in(list(res.sessions)))["r_cc"].to_numpy()
    np.testing.assert_allclose(res.gross_returns[1:], r_cc[1:] - r_cc.mean(), rtol=0.0, atol=1e-12)

    intraday_only = simulate_segments(
        panel,
        _targets([(days[1], "intraday", "A", 1.0)]),
        SIM,
        0.0,
        capital_krw=CAPITAL,
        mode="drift_neutral",
    )
    assert np.isfinite(intraday_only.gross_returns).all()


def test_drift_neutral_extreme_leg_fails_closed() -> None:
    opens = [100.0, 1.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    panel = _panel(_rows("A", CLOSES_A, opens=opens))
    with pytest.raises(SimulationDataError, match="<= -1"):
        _run(panel, [(D0, "intraday", "A", 1.0), (D0, "overnight", "A", 1.0)], mode="drift_neutral")


def test_held_ticker_delisted_fails_closed() -> None:
    panel = _panel(_rows("A", [*CLOSES_A[:4], None, None, None, None], opens=OPENS_A) + _rows("B", CLOSES_B))
    with pytest.raises(SimulationDataError, match="is held into"):
        _run(panel, [(D2, "intraday", "A", 1.0), (D2, "overnight", "A", 1.0)])


def test_non_positive_multiplier_fails_closed() -> None:
    panel = _panel(_rows("A", CLOSES_A, opens=OPENS_A) + _rows("B", CLOSES_B))
    with pytest.raises(SimulationDataError, match="non-positive"):
        _run(panel, [(D2, "intraday", "A", 1.0)], cost_bps=20_000.0)


def test_no_adv_blocks_new_listing() -> None:
    closes = [None, None, None, None, 104.0, 105.0, 106.0, 107.0]
    opens = [100.0, 100.0, 100.0, 100.0, 103.5, 104.5, 105.5, 106.5]
    panel = _panel(_rows("A", closes, opens=opens) + _rows("B", CLOSES_B))
    res = _run(panel, [(D4, "intraday", "A", 1.0), (D4, "overnight", "A", 1.0)])
    assert _blocked(res, D5) == [("A", "no_adv"), ("A", "no_adv")]


def test_drift_segments_and_dropped_decision() -> None:
    panel = _panel(_rows("A", CLOSES_A, opens=OPENS_A) + _rows("B", CLOSES_B))
    res = _run(panel, [(D2, "intraday", "A", 1.0), (D7, "overnight", "B", 1.0)], drift=(D4,))
    assert res.dropped_decisions == (D7,)
    assert res.mode == "actual"
    segmented = _run(panel, [(D2, "intraday", "A", 1.0)], mode="drift_neutral", drift=(D4,))
    assert np.isfinite(segmented.gross_returns).all()


def test_argument_guards() -> None:
    panel = _panel(_rows("A", CLOSES_A, opens=OPENS_A) + _rows("B", CLOSES_B))
    entries = _targets([(D2, "intraday", "A", 1.0)])
    with pytest.raises(ValueError, match="cost_bps_per_side"):
        simulate_segments(panel, entries, SIM, -1.0, capital_krw=CAPITAL)
    with pytest.raises(ValueError, match="cost_bps_per_side"):
        simulate_segments(panel, entries, SIM, float("nan"), capital_krw=CAPITAL)
    with pytest.raises(ValueError, match="capital_krw"):
        simulate_segments(panel, entries, SIM, 15.0, capital_krw=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        simulate_segments(panel, entries, SIM, 15.0, capital_krw=CAPITAL, drift_segments=(D4, D3))


@pytest.mark.parametrize(
    ("entries", "match"),
    [
        ([(D2, "midday", "A", 1.0)], "targets leg"),
        ([(D2, "intraday", "A", 0.3)], "is not a weight level"),
        ([(D2, "intraday", "A", 0.75), (D2, "intraday", "B", 0.5)], "sum to"),
        ([(D2, "intraday", "A", 0.5), (D2, "intraday", "A", 0.5)], "duplicate"),
        ([(date(2025, 12, 31), "intraday", "A", 1.0)], "panel session"),
        ([(D2, "intraday", "Z", 1.0)], "not in the panel frame"),
        ([], "non-empty"),
    ],
)
def test_bad_targets_rejected(entries: list[tuple[date, str, str, float]], match: str) -> None:
    panel = _panel(_rows("A", CLOSES_A, opens=OPENS_A) + _rows("B", CLOSES_B))
    with pytest.raises(ValueError, match=match):
        simulate_segments(panel, _targets(entries), SIM, 15.0, capital_krw=CAPITAL)


def test_bad_target_schema_and_nulls_rejected() -> None:
    panel = _panel(_rows("A", CLOSES_A, opens=OPENS_A) + _rows("B", CLOSES_B))
    with pytest.raises(ValueError, match="targets column weight"):
        simulate_segments(
            panel,
            pl.DataFrame(
                {"decision_date": [D2], "leg": ["intraday"], "ticker": ["A"], "weight": [1]},
                schema={**TARGET_SCHEMA, "weight": pl.Int64},
            ),
            SIM,
            15.0,
            capital_krw=CAPITAL,
        )
    with pytest.raises(ValueError, match="nulls"):
        simulate_segments(
            panel,
            pl.DataFrame(
                {"decision_date": [D2], "leg": ["intraday"], "ticker": ["A"], "weight": [None]},
                schema=TARGET_SCHEMA,
            ),
            SIM,
            15.0,
            capital_krw=CAPITAL,
        )
    with pytest.raises(ValueError, match="targets leg"):
        simulate_segments(panel, _targets([(D2, "midday", "A", 1.0)]), SIM, 15.0, capital_krw=CAPITAL)
