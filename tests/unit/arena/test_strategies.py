import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.arena import strategies as sg
from src.arena.panel import Panel, PanelConfig, build_panel
from src.arena.registry import config_digest
from src.arena.research import check_causality, load_research_config
from src.arena.simulate import load_sim_config, simulate
from src.arena.strategies import (
    Brake,
    CoinFlip,
    Committee,
    FlowContrarian,
    HoldLong,
    Rotation,
    StrategiesConfig,
    TrendSwitch,
    baselines,
    candidate_grid,
    load_strategies_config,
    strategy_tickers,
)
from src.core.config import ConfigError, load_config

PANEL_CFG = PanelConfig(lock_timeout_s=900.0, nav_divergence_max=0.10)
BASE = load_strategies_config()
RESEARCH = load_research_config()
SIM = load_sim_config()
CASH = "CASH"
KOSPI_SIGNAL, KOSPI_LONG, KOSPI_INV = "069500", "122630", "252670"
KOSDAQ_SIGNAL, KOSDAQ_LONG, KOSDAQ_INV = "229200", "233740", "251340"
SEMIS_SIGNAL, SEMIS_LONG = "091160", "494310"
SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date(),
    "ticker": pl.String(),
    "name": pl.String(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Int64(),
    "trading_value": pl.Int64(),
    "nav": pl.Float64(),
    "shares_outstanding": pl.Int64(),
    "underlying_index_name": pl.String(),
}
# Scenario fixtures: small windows so each regime fits in a few dozen sessions.
SMALL = replace(
    BASE,
    horizons=(2, 3),
    vol_window=3,
    er_window=4,
    er_min=0.4,
    band=0.05,
    min_listed_sessions=4,
    z_window=10,
    z_threshold=1.0,
    flow_window=(2,),
    rotation_hold=(5,),
    switch_sleeves=("kospi200",),
    chop_filter=(False, True),
    sleeves={
        "kospi200": (KOSPI_SIGNAL, KOSPI_LONG),
        "kosdaq150": (KOSDAQ_SIGNAL, KOSDAQ_LONG),
        "semis": (SEMIS_SIGNAL, SEMIS_LONG),
    },
    inverse={"kospi200": KOSPI_INV, "kosdaq150": KOSDAQ_INV},
)
_REMOVE = object()


def _bdays(n: int) -> list[date]:
    days: list[date] = []
    day = date(2024, 1, 1)
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _panel(
    closes: Mapping[str, Sequence[float | None]],
    shares: Mapping[str, Sequence[int]] | None = None,
    tickers: Sequence[str] | None = None,
) -> Panel:
    days = _bdays(len(next(iter(closes.values()))))
    rows: list[dict[str, Any]] = []
    for ticker, series in closes.items():
        for i, (day, close) in enumerate(zip(days, series, strict=True)):
            if close is None:
                continue
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "name": f"{ticker} fund",
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000,
                    "trading_value": 10**9,
                    "nav": close,
                    "shares_outstanding": 1000 if shares is None else shares[ticker][i],
                    "underlying_index_name": "IDX",
                }
            )
    frame = pl.DataFrame(rows, schema=SCHEMA)
    return build_panel(frame, list(tickers or closes), days[0], days[-1], PANEL_CFG)


def _wave(mean: float, amp: float, n: int) -> list[float]:
    return [mean + (amp if i % 2 == 0 else -amp) for i in range(n)]


def _closes(returns: Sequence[float], start: float = 100.0) -> list[float]:
    return [float(v) for v in start * np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))]


def _flat(n: int, value: float = 100.0) -> list[float]:
    return [value] * n


def _held(targets: pl.DataFrame, sessions: Sequence[date]) -> list[str | None]:
    """Forward-fill decisions: a 1.0 row holds its ticker, a 0.0 row holds CASH, before the first decision None."""
    by_day = {day: (ticker, weight) for day, ticker, weight in targets.rows()}
    held: list[str | None] = []
    current: str | None = None
    for day in sessions:
        if day in by_day:
            ticker, weight = by_day[day]
            current = ticker if weight > 0.0 else CASH
        held.append(current)
    return held


def _schema_ok(targets: pl.DataFrame) -> None:
    assert targets.schema == {"decision_date": pl.Date, "ticker": pl.String, "weight": pl.Float64}
    assert set(targets["weight"].to_list()) <= {0.0, 1.0}


def test_loaded_config_documents_the_shipped_values() -> None:
    assert BASE.horizons == (10, 20, 40, 60, 120)
    assert (BASE.vol_window, BASE.er_window, BASE.band, BASE.er_min) == (60, 20, 0.10, 0.30)
    assert BASE.sleeves["kospi200"] == ("069500", "122630")
    assert BASE.sleeves["skhynix"] == ("0193T0", "0193T0")
    assert dict(BASE.inverse) == {
        "kospi200": "252670",
        "kosdaq150": "251340",
        "skhynix": "0197X0",
        "samsung": "0193L0",
    }
    assert BASE.min_listed_sessions == 20
    assert BASE.switch_sleeves == ("kospi200", "kosdaq150", "skhynix", "samsung")
    assert BASE.chop_filter == (False, True)
    assert BASE.rotation_hold == (5, 10)
    assert BASE.flow_window == (5, 20)
    assert (BASE.z_window, BASE.z_threshold) == (60, 1.0)
    assert BASE.committee is True
    assert BASE.down_streak == 3
    assert BASE.baseline_hold == ("122630", "0193T0", "0193W0")
    assert BASE.coin_pair == ("122630", "252670")
    assert BASE.coin_seed == 20261105


def test_candidate_grid_has_fourteen_unique_ids_in_documented_order() -> None:
    ids = [strategy.strategy_id for strategy in candidate_grid(BASE)]
    assert ids == [
        "trend_switch:kospi200:plain",
        "trend_switch:kospi200:chop",
        "trend_switch:kosdaq150:plain",
        "trend_switch:kosdaq150:chop",
        "trend_switch:skhynix:plain",
        "trend_switch:skhynix:chop",
        "trend_switch:samsung:plain",
        "trend_switch:samsung:chop",
        "rotation:hold5",
        "rotation:hold10",
        "flow_contrarian:w5",
        "flow_contrarian:w20",
        "committee:trend+flow",
        "brake:kospi200:down3",
    ]
    assert len(set(ids)) == 14
    assert len({config_digest(s.params) for s in candidate_grid(BASE)}) == 14
    assert [s.strategy_id for s in baselines(BASE)] == [
        "hold:122630",
        "hold:0193T0",
        "hold:0193W0",
        "coin:122630-252670:20261105",
    ]
    assert "committee:trend+flow" not in [s.strategy_id for s in candidate_grid(replace(BASE, committee=False))]


def _mutated(section: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def apply(data: dict[str, Any]) -> None:
        node = data["strategies"][section]
        if value is _REMOVE:
            del node[key]
        else:
            node[key] = value

    return apply


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("trend", "horizons", [10, 20, 20, 60, 120]),
        ("trend", "horizons", [60, 40, 20, 10, 120]),
        ("trend", "horizons", [True]),
        ("trend", "horizons", []),
        ("trend", "vol_window", 1),
        ("trend", "band", -0.1),
        ("trend", "er_window", 0),
        ("trend", "er_min", 1.5),
        ("trend", "er_min", True),
        ("trend", "er_min", _REMOVE),
        ("sleeves", "kospi200", ["069500", "12263"]),
        ("sleeves", "kospi200", ["069500"]),
        ("sleeves", "kospi200", ["069500", "lowercase"]),
        ("sleeves", "kospi200", "069500"),
        ("inverse", "kospi200", "25267"),
        ("inverse", "kosdaq150", _REMOVE),
        ("trend_switch", "sleeves", ["kosdaq150", "kosdaq150"]),
        ("trend_switch", "sleeves", []),
        ("trend_switch", "sleeves", [""]),
        ("trend_switch", "sleeves", ["nosuchsleeve"]),
        ("trend_switch", "chop_filter", [False, False]),
        ("trend_switch", "chop_filter", [1]),
        ("trend_switch", "chop_filter", []),
        ("rotation", "hold", [True]),
        ("rotation", "hold", [0]),
        ("rotation", "hold", [5, 5]),
        ("rotation", "hold", []),
        ("flow_contrarian", "flow_window", []),
        ("flow_contrarian", "z_window", 1),
        ("flow_contrarian", "z_threshold", 0.0),
        ("flow_contrarian", "z_threshold", False),
        ("committee", "enabled", "yes"),
        ("brake", "down_streak", 0),
        ("baselines", "hold", []),
        ("baselines", "hold", ["12263"]),
        ("baselines", "coin_pair", ["122630"]),
        ("baselines", "coin_seed", -1),
        ("baselines", "coin_seed", True),
    ],
)
def test_config_violations_raise(monkeypatch: pytest.MonkeyPatch, section: str, key: str, value: Any) -> None:
    data = copy.deepcopy(dict(load_config("arena")))
    _mutated(section, key, value)(data)
    monkeypatch.setattr(sg, "load_config", lambda name: data)
    with pytest.raises(ConfigError):
        load_strategies_config()


def _config_with(monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict[str, Any]], None]) -> None:
    data = copy.deepcopy(dict(load_config("arena")))
    mutate(data)
    monkeypatch.setattr(sg, "load_config", lambda name: data)


def _drop_kospi(data: dict[str, Any]) -> None:
    del data["strategies"]["sleeves"]["kospi200"]
    del data["strategies"]["inverse"]["kospi200"]


def test_config_structure_violations_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sg, "load_config", lambda name: {"other": {}})
    with pytest.raises(ConfigError, match="strategies mapping"):
        load_strategies_config()

    cases: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("keys must be exactly", lambda d: d["strategies"].update(extra=1)),
        ("non-empty mapping", lambda d: d["strategies"].update(sleeves={})),
        ("inverse must be a mapping", lambda d: d["strategies"].update(inverse=["252670"])),
        ("keys must be sleeves", lambda d: d["strategies"]["inverse"].update(nosuch="252670")),
        ("must both define kospi200", lambda d: _drop_kospi(d)),
        ("non-empty strings", lambda d: d["strategies"]["sleeves"].update({"": ["069500", "122630"]})),
        ("non-empty strings", lambda d: d["strategies"]["sleeves"].update({1: ["069500", "122630"]})),
        ("must be in sleeves and inverse", lambda d: d["strategies"]["inverse"].pop("kosdaq150")),
    ]
    for match, mutate in cases:
        _config_with(monkeypatch, mutate)
        with pytest.raises(ConfigError, match=match):
            load_strategies_config()


def test_config_switch_sleeve_missing_from_inverse_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _config_with(monkeypatch, lambda d: d["strategies"]["inverse"].pop("kosdaq150"))
    with pytest.raises(ConfigError, match="kosdaq150"):
        load_strategies_config()


def test_config_mapping_with_one_switch_sleeve_builds_the_expected_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    _config_with(monkeypatch, lambda d: d["strategies"]["trend_switch"].update(sleeves=["kospi200"]))
    cfg = load_strategies_config()
    assert cfg.switch_sleeves == ("kospi200",)
    assert [s.strategy_id for s in candidate_grid(cfg)][:2] == [
        "trend_switch:kospi200:plain",
        "trend_switch:kospi200:chop",
    ]


def test_trend_switch_long_cash_inverse_with_chop_filter() -> None:
    rise = _wave(0.02, 0.005, 15)
    chop = _wave(0.01, 0.03, 12)
    fall = _wave(-0.03, 0.005, 15)
    sig = _closes([0.0, *rise, *chop, *fall])
    n = len(sig)
    panel = _panel({KOSPI_SIGNAL: sig, KOSPI_LONG: _flat(n), KOSPI_INV: _flat(n)})
    days = list(panel.sessions)

    plain = TrendSwitch("kospi200", False, SMALL).targets(panel)
    _schema_ok(plain)
    held_plain = _held(plain, days)
    assert held_plain[:3] == [None, None, None]
    assert held_plain[3:28] == [KOSPI_LONG] * 25
    assert held_plain[28:] == [KOSPI_INV] * (n - 28)
    assert plain.height == 2

    chopped = TrendSwitch("kospi200", True, SMALL).targets(panel)
    _schema_ok(chopped)
    held_chop = _held(chopped, days)
    assert held_chop[3:19] == [KOSPI_LONG] * 16
    assert held_chop[19:30] == [CASH] * 11
    assert held_chop[30:] == [KOSPI_INV] * (n - 30)
    assert chopped.height == 3
    assert chopped.row(1) == (days[19], KOSPI_LONG, 0.0)


def test_rotation_picks_strongest_eligible_and_repicks_on_grid() -> None:
    n = 120
    kospi_r = [0.0, *_wave(0.0, 0.001, 59), *_wave(-0.03, 0.005, 30), *_wave(0.0, 0.001, 30)]
    kosdaq_r = [0.0, *_wave(0.015, 0.02, 59), *_wave(0.0, 0.001, 60)]
    semis_r = [0.0, *_wave(0.03, 0.02, 59), *_wave(0.0, 0.001, 60)]
    kospi, kosdaq, semis = _closes(kospi_r), _closes(kosdaq_r), _closes(semis_r)
    late = [value if day >= 50 else None for day, value in enumerate(semis)]
    panel = _panel(
        {
            KOSPI_SIGNAL: kospi,
            KOSPI_LONG: _flat(n),
            KOSDAQ_SIGNAL: kosdaq,
            KOSDAQ_LONG: kosdaq,
            KOSDAQ_INV: _flat(n),
            SEMIS_SIGNAL: semis,
            SEMIS_LONG: late,
            KOSPI_INV: _flat(n),
        }
    )
    days = list(panel.sessions)
    targets = Rotation(5, replace(SMALL, band=0.5)).targets(panel)
    _schema_ok(targets)
    held = _held(targets, days)

    rebalances = {3 + 5 * m for m in range(0, 24)}
    assert {days.index(day) for day in targets["decision_date"].to_list()} <= rebalances
    assert held[23] == KOSDAQ_LONG
    assert held[48] == KOSDAQ_LONG  # semis is strongest here but its long leg is not yet listed for 4 sessions
    assert held[53] == SEMIS_LONG
    assert held[68] == KOSPI_INV
    assert held[98] == CASH
    assert held[24] == KOSDAQ_LONG and held[54] == SEMIS_LONG and held[69] == KOSPI_INV
    assert held[:3] == [None, None, None]


def test_flow_contrarian_fades_surges_and_holds_between_triggers() -> None:
    n = 80
    lev = [1000] * n
    inv = [500] * n
    for i in range(30, n):
        lev[i] = 2000
    for i in range(60, n):
        lev[i] = 0
    panel = _panel(
        {KOSPI_LONG: _flat(n, 10.0), KOSPI_INV: _flat(n, 20.0)},
        shares={KOSPI_LONG: lev, KOSPI_INV: inv},
    )
    days = list(panel.sessions)
    targets = FlowContrarian(2, SMALL).targets(panel)
    _schema_ok(targets)
    assert targets.rows() == [(days[30], KOSPI_INV, 1.0), (days[60], KOSPI_LONG, 1.0)]
    held = _held(targets, days)
    assert held[31:60] == [KOSPI_INV] * 29
    assert all(h in (KOSPI_LONG, None) for h in held[61:])
    assert held[-1] == KOSPI_LONG


def _committee_panel() -> tuple[Panel, Sequence[date]]:
    n = 80
    kospi = _closes([0.0, *_wave(0.02, 0.005, 49), *_wave(-0.03, 0.005, 30)])
    lev = [1000 + (1 if i % 4 in (0, 1) else -1) * 0 for i in range(n)]
    noise = [0, 1, 2, 1, 0, -1, -2, -1] * 10
    lev = [1000 + noise[i] for i in range(n)]
    for i in range(20, n):
        lev[i] += 1000
    for i in range(35, n):
        lev[i] -= 2000
    for i in range(70, n):
        lev[i] += 1000
    inv = [500] * n
    panel = _panel(
        {KOSPI_SIGNAL: kospi, KOSPI_LONG: _flat(n, 10.0), KOSPI_INV: _flat(n, 20.0)},
        shares={KOSPI_SIGNAL: [1000] * n, KOSPI_LONG: lev, KOSPI_INV: inv},
    )
    return panel, list(panel.sessions)


def test_committee_follows_the_agreed_votes_and_is_cash_on_disagreement() -> None:
    panel, days = _committee_panel()
    cfg = replace(SMALL, z_threshold=2.0)
    committee = Committee(cfg).holdings(panel)
    trend = TrendSwitch("kospi200", False, cfg).holdings(panel)
    flow = FlowContrarian(2, cfg).holdings(panel)
    assert len(committee) == len(days)
    expected: list[str | None] = []
    for t, f in zip(trend, flow, strict=True):
        if t is None or f is None:
            expected.append(None)
            continue
        votes = sum(1 if h == KOSPI_LONG else -1 if h == KOSPI_INV else 0 for h in (t, f))
        expected.append(KOSPI_LONG if votes > 0 else KOSPI_INV if votes < 0 else CASH)
    assert committee == expected
    assert any(h == KOSPI_LONG for h in committee)
    assert any(h == CASH for h in committee)
    assert any(h == KOSPI_INV for h in committee)
    assert committee[0] is None and committee[-1] is not None

    targets = Committee(cfg).targets(panel)
    _schema_ok(targets)
    for held, want in zip(_held(targets, days), expected, strict=True):
        if want is not None:
            assert held == want


def test_brake_forces_cash_during_a_down_streak_while_trend_wants_long() -> None:
    returns = [0.0, *_wave(0.02, 0.01, 30), -0.01, -0.01, -0.01, *_wave(0.02, 0.01, 17)]
    sig = _closes(returns)
    n = len(sig)
    panel = _panel({KOSPI_SIGNAL: sig, KOSPI_LONG: _flat(n), KOSPI_INV: _flat(n)})
    cfg = replace(SMALL, horizons=(5, 10), vol_window=10)
    trend = TrendSwitch("kospi200", False, cfg).holdings(panel)
    brake = Brake(cfg).holdings(panel)

    run = 0
    expected: list[str | None] = []
    for t, r in zip(trend, returns, strict=True):
        run = run + 1 if r < 0.0 else 0
        expected.append(CASH if t == KOSPI_LONG and run >= cfg.down_streak else t)
    assert brake == expected
    forced = [i for i, (t, b) in enumerate(zip(trend, brake, strict=True)) if t == KOSPI_LONG and b == CASH]
    assert forced, "scenario must force at least one cash session during the streak"
    targets = Brake(cfg).targets(panel)
    _schema_ok(targets)
    assert _held(targets, list(panel.sessions))[forced[0]] == CASH


def test_hold_long_emits_one_decision_once_listed() -> None:
    n = 12
    panel = _panel({"122630": [None] * 5 + _flat(n - 5), "252670": _flat(n)})
    targets = HoldLong("122630", 3).targets(panel)
    _schema_ok(targets)
    assert targets.rows() == [(panel.sessions[7], "122630", 1.0)]


def test_coin_flip_is_deterministic_per_seed_and_matches_its_draws() -> None:
    n = 40
    panel = _panel({KOSPI_LONG: _flat(n), KOSPI_INV: _flat(n, 50.0)})
    days = list(panel.sessions)
    rng = np.random.default_rng(3)
    draws = [rng.random() for _ in days]
    holdings = [KOSPI_LONG if u < 0.5 else KOSPI_INV for u in draws]
    expected = [
        (day, ticker, 1.0)
        for i, (day, ticker) in enumerate(zip(days, holdings, strict=True))
        if i == 0 or holdings[i - 1] != ticker
    ]
    first = CoinFlip(KOSPI_LONG, KOSPI_INV, 3, 1).targets(panel)
    _schema_ok(first)
    assert first.rows() == expected
    assert CoinFlip(KOSPI_LONG, KOSPI_INV, 3, 1).targets(panel).rows() == first.rows()
    assert CoinFlip(KOSPI_LONG, KOSPI_INV, 4, 1).targets(panel).rows() != first.rows()


def test_coin_flip_waits_until_both_legs_have_history() -> None:
    n = 12
    panel = _panel({KOSPI_LONG: _flat(n), KOSPI_INV: [None] * 4 + _flat(n - 4)})
    targets = CoinFlip(KOSPI_LONG, KOSPI_INV, 9, 2).targets(panel)
    assert targets["decision_date"].min() == panel.sessions[5]


def test_strategy_tickers_union_and_unknown_type_rejected() -> None:
    assert strategy_tickers([FlowContrarian(5, BASE)]) == frozenset({KOSPI_LONG, KOSPI_INV})
    assert strategy_tickers([CoinFlip("122630", "252670", 1, 1)]) == frozenset({"122630", "252670"})
    assert strategy_tickers([HoldLong("0193T0", 20)]) == frozenset({"0193T0"})
    assert strategy_tickers([Committee(BASE)]) == frozenset({KOSPI_SIGNAL, KOSPI_LONG, KOSPI_INV})

    class Unknown:
        family = "unknown"
        strategy_id = "unknown"
        params: Mapping[str, Any] = {}

        def targets(self, panel: Panel) -> pl.DataFrame:
            raise NotImplementedError

    with pytest.raises(TypeError, match="unknown strategy type"):
        strategy_tickers([Unknown()])  # type: ignore[list-item]


def test_all_candidates_and_baselines_are_causal_and_simulable() -> None:
    n = 260
    rng = np.random.default_rng(20261105)
    tickers = sorted(
        {t for pair in BASE.sleeves.values() for t in pair}
        | set(BASE.inverse.values())
        | set(BASE.baseline_hold)
        | set(BASE.coin_pair)
    )
    closes: dict[str, list[float | None]] = {}
    shares: dict[str, list[int]] = {}
    for ticker in tickers:
        path: list[float | None] = list(_closes(rng.normal(0.0005, 0.01, n).tolist()))
        if ticker == "0193T0":
            path = [None] * 40 + path[40:]
        if ticker == "418660":
            path = [None] * 100 + path[100:]
        closes[ticker] = path
        shares[ticker] = [int(v) for v in np.maximum(1000 + np.cumsum(rng.integers(-40, 41, n)), 1)]
    panel = _panel(closes, shares=shares)
    assert strategy_tickers((*candidate_grid(BASE), *baselines(BASE))) == frozenset(tickers)
    for strategy in (*candidate_grid(BASE), *baselines(BASE)):
        targets = check_causality(strategy, panel, RESEARCH)
        _schema_ok(targets)
        result = simulate(panel, targets, SIM, capital_krw=10**9)
        assert len(result.sessions) > 0
    assert [type(s) for s in candidate_grid(BASE)].count(Rotation) == 2
