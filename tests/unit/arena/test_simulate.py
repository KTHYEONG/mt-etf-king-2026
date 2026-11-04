from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.arena import simulate as sim
from src.arena.evaluation import EvalConfig, evaluate
from src.arena.panel import Panel, PanelConfig, build_panel
from src.arena.simulate import SimConfig, SimulationDataError, SimResult, load_sim_config, simulate
from src.core.config import ConfigError, load_config

PANEL_CFG = PanelConfig(lock_timeout_s=900.0, nav_divergence_max=0.10)
DAYS = [date(2026, 1, 1) + timedelta(days=i) for i in range(8)]
D0, D1, D2, D3, D4, D5, D6, D7 = DAYS
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
SIM = SimConfig("open", 15.0, 1.0, 2, LEVELS, 1e-9)
CAPITAL = 1000
COST = 15.0 / 10_000.0
EVAL = EvalConfig(20, 10, 400, 7, 0.95, (1.1, 1.3, 1.5, 2.0, 2.3, 3.0), 60)
CLOSES_A = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
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
    opens: list[float] | None = None,
    volumes: list[int] | None = None,
    days: list[date] = DAYS,
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


def _panel(rows: list[dict[str, Any]], days: list[date] = DAYS) -> Panel:
    frame = pl.DataFrame(rows, schema=SCHEMA)
    return build_panel(frame, {r["ticker"] for r in rows}, days[0], days[-1], PANEL_CFG)


def _targets(entries: list[tuple[date, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decision_date": [e[0] for e in entries],
            "ticker": [e[1] for e in entries],
            "weight": [e[2] for e in entries],
        },
        schema={"decision_date": pl.Date, "ticker": pl.String, "weight": pl.Float64},
    )


def _run(
    panel: Panel,
    entries: list[tuple[date, str, float]],
    cfg: SimConfig = SIM,
    capital: int = CAPITAL,
    mode: sim.ReturnMode = "actual",
) -> SimResult:
    return simulate(panel, _targets(entries), cfg, capital_krw=capital, mode=mode)


def _weight(res: SimResult, day: date, ticker: str) -> float:
    matched = res.weights.filter((pl.col("date") == day) & (pl.col("ticker") == ticker))
    return float(matched["weight"].sum())


def _blocked(res: SimResult, day: date) -> list[tuple[str, str]]:
    rows = res.blocked.filter(pl.col("date") == day).sort("ticker")
    return [(r["ticker"], r["reason"]) for r in rows.iter_rows(named=True)]


def _load_with(monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict[str, Any]], None]) -> SimConfig:
    section: dict[str, Any] = dict(load_config("arena")["simulate"])
    mutate(section)
    data: Mapping[str, Any] = {"simulate": section}
    monkeypatch.setattr(sim, "load_config", lambda name: data)
    return load_sim_config()


def test_sim_config_loads_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_sim_config() == SimConfig("open", 15.0, 0.05, 20, LEVELS, 1e-9)

    def _set(**changes: Any) -> Callable[[dict[str, Any]], None]:
        return lambda section: section.update(changes)

    def _drop(key: str) -> Callable[[dict[str, Any]], None]:
        return lambda section: section.pop(key)

    bad: list[Callable[[dict[str, Any]], None]] = [
        _set(execution="vwap"),
        _set(adv_window=True),
        _set(adv_window=0),
        _set(weight_levels=[0.25, 0.5, 1.0]),
        _set(weight_levels=[0.0, 0.5, 0.25]),
        _set(weight_levels=[0.0, 1.5]),
        _set(weight_levels=[]),
        _set(weight_levels="abc"),
        _set(cost_bps_per_side=-1.0),
        _set(cost_bps_per_side=True),
        _set(participation_max=0.0),
        _set(participation_max=1.5),
        _set(weight_tolerance=0.0),
        _set(extra=1),
        _drop("weight_tolerance"),
    ]
    for mutate in bad:
        with pytest.raises(ConfigError):
            _load_with(monkeypatch, mutate)

    monkeypatch.setattr(sim, "load_config", lambda name: {})
    with pytest.raises(ConfigError):
        load_sim_config()


def test_next_session_execution() -> None:
    opens_a = [100.0, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5]
    panel = _panel(_rows("A", CLOSES_A, opens=opens_a))
    res = _run(panel, [(D2, "A", 1.0)])
    r_id = CLOSES_A[3] / opens_a[3] - 1.0
    assert res.sessions[0] == D3
    assert res.gross_returns[0] == pytest.approx(r_id, rel=1e-12)
    assert res.net_returns[0] == pytest.approx((1.0 + r_id) * (1.0 - COST) - 1.0, rel=1e-12)
    assert res.costs[0] == pytest.approx(COST * (1.0 + r_id), rel=1e-12)


def test_decision_perturbation_invariance() -> None:
    panel = _panel(_rows("A", CLOSES_A) + _rows("B", CLOSES_B))
    common = [(D2, "A", 1.0)]
    first = _run(panel, [*common, (D3, "A", 0.5), (D3, "B", 0.5)])
    second = _run(panel, [*common, (D3, "A", 0.25), (D3, "B", 0.75)])
    np.testing.assert_array_equal(first.net_returns[:1], second.net_returns[:1])
    assert first.net_returns[1] != second.net_returns[1]


def test_future_price_invariance() -> None:
    later_closes_a = [*CLOSES_A[:5], 80.0, 70.0, 60.0]
    later_opens = [*CLOSES_A[:5], 90.0, 60.0, 50.0]
    base = _panel(_rows("A", CLOSES_A) + _rows("B", CLOSES_B))
    changed = _panel(
        _rows("A", later_closes_a, opens=later_opens, volumes=[1000] * 5 + [1, 1, 1], tv=100000) + _rows("B", CLOSES_B)
    )
    entries = [(D2, "A", 1.0), (D3, "A", 0.5), (D3, "B", 0.5)]
    early = _run(base, entries)
    late = _run(changed, entries)
    np.testing.assert_array_equal(early.net_returns[:2], late.net_returns[:2])
    np.testing.assert_array_equal(early.costs[:2], late.costs[:2])
    np.testing.assert_array_equal(early.gross_returns[:2], late.gross_returns[:2])


def test_open_mode_legs_on_no_trade_session() -> None:
    opens_a = [100.0, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5]
    panel = _panel(_rows("A", CLOSES_A, opens=opens_a))
    res = _run(panel, [(D2, "A", 1.0)])
    r_on = opens_a[4] / CLOSES_A[3] - 1.0
    r_id = CLOSES_A[4] / opens_a[4] - 1.0
    assert res.turnover[1] == 0.0
    assert res.net_returns[1] == pytest.approx((1.0 + r_on) * (1.0 + r_id) - 1.0, rel=1e-12)


def test_close_mode_return() -> None:
    panel = _panel(_rows("A", CLOSES_A))
    cfg = SimConfig("close", 15.0, 1.0, 2, LEVELS, 1e-9)
    res = _run(panel, [(D2, "A", 1.0)], cfg=cfg)
    assert res.gross_returns[0] == pytest.approx(0.0, abs=1e-15)
    assert res.net_returns[0] == pytest.approx(-COST, rel=1e-12)
    assert res.net_returns[1] == pytest.approx(CLOSES_A[4] / CLOSES_A[3] - 1.0, rel=1e-12)


def test_cost_identity_on_full_switch() -> None:
    panel = _panel(_rows("A", CLOSES_A) + _rows("B", CLOSES_B))
    res = _run(panel, [(D2, "A", 1.0), (D4, "B", 1.0)])
    assert res.turnover[2] == pytest.approx(2.0, rel=1e-12)
    growth = res.gross_returns[2] + 1.0
    assert res.costs[2] == pytest.approx(2.0 * COST * growth, rel=1e-12)
    assert res.net_returns[2] == pytest.approx(res.gross_returns[2] - res.costs[2], rel=1e-12)


def test_adv_cap_clips_and_reports_without_carry() -> None:
    panel = _panel(_rows("A", CLOSES_A))
    res = _run(panel, [(D2, "A", 1.0)], capital=250_000)
    assert _weight(res, D3, "A") == pytest.approx(0.4, rel=1e-12)
    assert _blocked(res, D3) == [("A", "adv_cap")]
    row = res.blocked.row(0, named=True)
    assert row["wanted"] == pytest.approx(1.0)
    assert row["executed"] == pytest.approx(0.4, rel=1e-12)
    assert res.turnover[1] == 0.0


def test_no_adv_blocks_new_listing() -> None:
    closes_b: list[float | None] = [None, None, None, 50.0, 50.5, 51.0, 51.5, 52.0]
    panel = _panel(_rows("A", CLOSES_A) + _rows("B", closes_b))
    res = _run(panel, [(D3, "B", 1.0)])
    assert _blocked(res, D4) == [("B", "no_adv")]
    assert _weight(res, D4, "B") == 0.0


def test_not_listed_order_is_blocked() -> None:
    closes_b: list[float | None] = [None, None, None, None, 50.0, 50.5, 51.0, 51.5]
    panel = _panel(_rows("A", CLOSES_A) + _rows("B", closes_b))
    res = _run(panel, [(D2, "B", 1.0)])
    assert _blocked(res, D3) == [("B", "not_listed")]


def test_untradable_held_position_marked_by_ref() -> None:
    volumes = [1000, 1000, 1000, 1000, 1000, 0, 1000, 1000]
    panel = _panel(_rows("A", CLOSES_A, volumes=volumes))
    res = _run(panel, [(D2, "A", 1.0), (D4, "A", 0.5)])
    assert res.net_returns[2] == pytest.approx(CLOSES_A[5] / CLOSES_A[4] - 1.0, rel=1e-12)
    assert _blocked(res, D5) == [("A", "not_traded")]
    assert _weight(res, D5, "A") == pytest.approx(1.0, rel=1e-12)


def test_held_ticker_delisted_fails_closed() -> None:
    closes_a: list[float | None] = [100.0, 101.0, 102.0, 103.0, 104.0, None, None, None]
    panel = _panel(_rows("A", closes_a) + _rows("B", CLOSES_B))
    with pytest.raises(SimulationDataError, match=f"A is held into {D5}"):
        _run(panel, [(D2, "A", 1.0)])


def test_sells_before_buys_and_cash_scaling() -> None:
    volumes = [1000, 1000, 1000, 1000, 1000, 0, 1000, 1000]
    panel = _panel(_rows("A", CLOSES_A, volumes=volumes) + _rows("B", CLOSES_B))
    res = _run(panel, [(D2, "A", 1.0), (D4, "A", 0.5), (D4, "B", 0.5)])
    assert _blocked(res, D5) == [("A", "not_traded"), ("B", "cash")]
    sums = res.weights.group_by("date").agg(pl.col("weight").sum())
    assert sums.filter(pl.col("weight") > 1.0 + 1e-9).height == 0
    assert _weight(res, D5, "B") == 0.0


def test_sell_then_buy_within_room_needs_no_cash_row() -> None:
    panel = _panel(_rows("A", CLOSES_A) + _rows("B", CLOSES_B))
    res = _run(panel, [(D2, "A", 1.0), (D4, "A", 0.5), (D4, "B", 0.5)])
    assert _blocked(res, D5) == []
    assert _weight(res, D5, "B") == pytest.approx(0.5, rel=1e-12)


def test_drift_neutral_kills_beta() -> None:
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(120)]
    returns = [0.02 + 0.01 * (-1.0) ** k for k in range(1, 120)]
    returns[0] = sum(returns[1:]) / (len(returns) - 1)
    closes = [100.0]
    for r in returns:
        closes.append(closes[-1] * (1.0 + r))
    panel = _panel(_rows("A", closes, days=days), days=days)
    cfg = SimConfig("close", 15.0, 1.0, 1, LEVELS, 1e-9)
    neutral = _run(panel, [(days[0], "A", 1.0)], cfg=cfg, mode="drift_neutral")
    actual = _run(panel, [(days[0], "A", 1.0)], cfg=cfg)
    assert abs(float(neutral.gross_returns.mean())) < 1e-12
    assert float(actual.gross_returns.mean()) > 0.0
    report = evaluate("s", "h", neutral.sessions, neutral.net_returns, neutral.costs, EVAL, 1)
    assert report.edge.p_value >= 0.4


def test_drift_neutral_extreme_leg_fails_closed() -> None:
    closes: list[float | None] = [100.0, 10.0, 100.0, 10.0, 100.0, 10.0, 100.0, 10.0]
    panel = _panel(_rows("A", closes))
    with pytest.raises(SimulationDataError, match="leg return"):
        _run(panel, [(D2, "A", 1.0)], mode="drift_neutral")


def test_non_positive_multiplier_fails_closed() -> None:
    panel = _panel(_rows("A", CLOSES_A))
    cfg = SimConfig("open", 10_000.0, 1.0, 2, LEVELS, 1e-9)
    with pytest.raises(SimulationDataError, match="multiplier"):
        _run(panel, [(D2, "A", 1.0)], cfg=cfg)


@pytest.mark.parametrize("weight", [0.3, 0.2])
def test_weights_must_be_levels(weight: float) -> None:
    panel = _panel(_rows("A", CLOSES_A))
    with pytest.raises(ValueError, match="weight level"):
        _run(panel, [(D2, "A", weight)])


def test_weight_within_tolerance_is_snapped() -> None:
    panel = _panel(_rows("A", CLOSES_A))
    res = _run(panel, [(D2, "A", 0.25 + 1e-12)])
    assert _weight(res, D3, "A") == pytest.approx(0.25, abs=1e-12)


def test_bad_targets_rejected() -> None:
    panel = _panel(_rows("A", CLOSES_A) + _rows("B", CLOSES_B))
    with pytest.raises(ValueError, match="decision_date must be"):
        _run(panel, [(date(2030, 1, 1), "A", 1.0)])
    with pytest.raises(ValueError, match="duplicate"):
        _run(panel, [(D2, "A", 0.5), (D2, "A", 0.5)])
    with pytest.raises(ValueError, match="sum to"):
        _run(panel, [(D2, "A", 0.75), (D2, "B", 0.5)])
    with pytest.raises(ValueError, match="not in the panel frame"):
        _run(panel, [(D2, "Z", 1.0)])
    with pytest.raises(ValueError, match="must be non-empty"):
        _run(panel, [])
    with pytest.raises(ValueError, match="capital_krw"):
        _run(panel, [(D2, "A", 1.0)], capital=0)


def test_bad_target_schema_and_nulls_rejected() -> None:
    panel = _panel(_rows("A", CLOSES_A))
    missing = _targets([(D2, "A", 1.0)]).drop("weight")
    with pytest.raises(ValueError, match="weight"):
        simulate(panel, missing, SIM, capital_krw=CAPITAL)
    nulled = pl.DataFrame(
        {"decision_date": [D2], "ticker": ["A"], "weight": [None]},
        schema={"decision_date": pl.Date, "ticker": pl.String, "weight": pl.Float64},
    )
    with pytest.raises(ValueError, match="nulls"):
        simulate(panel, nulled, SIM, capital_krw=CAPITAL)


def test_last_session_decision_dropped() -> None:
    panel = _panel(_rows("A", CLOSES_A))
    base = _run(panel, [(D2, "A", 1.0)])
    with_last = _run(panel, [(D2, "A", 1.0), (D7, "A", 0.5)])
    assert with_last.dropped_decisions == (D7,)
    assert base.dropped_decisions == ()
    np.testing.assert_array_equal(base.net_returns, with_last.net_returns)


def test_evaluate_accepts_output() -> None:
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(300)]
    rng = np.random.default_rng(3)
    closes_a = list(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 300))))
    closes_b = list(50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, 300))))
    panel = _panel(_rows("A", closes_a, days=days) + _rows("B", closes_b, days=days), days=days)
    patterns: list[list[tuple[str, float]]] = [[("A", 1.0)], [("A", 0.5), ("B", 0.5)], [("B", 1.0)]]
    entries: list[tuple[date, str, float]] = []
    for idx in range(2, 300, 5):
        entries.extend((days[idx], ticker, w) for ticker, w in patterns[(idx // 5) % 3])
    res = simulate(panel, _targets(entries), SIM, capital_krw=CAPITAL)
    assert np.isfinite(res.net_returns).all()
    report = evaluate("s", "h", res.sessions, res.net_returns, res.costs, EVAL, 1)
    assert report.edge.n_sessions == len(res.sessions)


def test_close_execution_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _load_with(monkeypatch, lambda section: section.update(execution="close"))
    assert loaded.execution == "close"


def test_drift_neutral_open_hold_has_no_leg_covariance_drift() -> None:
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
    cfg = SimConfig("open", 0.0, 1.0, 1, LEVELS, 1e-9)
    neutral = _run(panel, [(days[1], "A", 1.0)], cfg=cfg, mode="drift_neutral")
    r_cc = panel.frame.filter(pl.col("date").is_in(list(neutral.sessions)))["r_cc"].to_numpy()
    held = 1.0 + neutral.gross_returns[1:]
    assert np.allclose(held, 1.0 + r_cc[1:] - r_cc.mean(), rtol=0.0, atol=1e-12)
