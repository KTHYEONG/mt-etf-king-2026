import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import polars as pl
import pytest

from src.arena import research as rs
from src.arena.evaluation import EvalConfig, EvaluationReport, YearStat, evaluate
from src.arena.panel import Panel, PanelConfig, build_panel
from src.arena.registry import count_trials
from src.arena.research import (
    REGIMES,
    CausalityError,
    Evidence,
    GateVerdict,
    Regime,
    ResearchConfig,
    check_causality,
    gate,
    gather_evidence,
    load_research_config,
    regime_labels,
    select,
    truncate_panel,
)
from src.arena.prize_line import PrizeLineConfig
from src.arena.segment import SegmentConfig
from src.arena.simulate import SimConfig, simulate
from src.core.config import ConfigError, load_config

PANEL_CFG = PanelConfig(lock_timeout_s=900.0, nav_divergence_max=0.10)
SIM = SimConfig("open", 3.0, 1.0, 2, (0.0, 0.25, 0.5, 0.75, 1.0), 1e-9)
EVAL = EvalConfig(10, 10, 200, 7, 0.95, (1.1, 1.3, 1.5, 2.0, 2.3, 3.0), 40)
CAPITAL = 1_000_000
NOW = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)
REF = "069500"
BASE = load_research_config()
_BASE_RESEARCH: dict[str, Any] = dict(load_config("arena")["research"])
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


def _bdays(n: int) -> list[date]:
    days: list[date] = []
    day = date(2024, 1, 1)
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _rows(ticker: str, returns: Sequence[float], days: Sequence[date]) -> list[dict[str, Any]]:
    closes = 100.0 * np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
    return [
        {
            "date": day,
            "ticker": ticker,
            "name": f"{ticker} fund",
            "open": float(close),
            "high": float(close) * 1.01,
            "low": float(close) * 0.99,
            "close": float(close),
            "volume": 1000,
            "trading_value": 10**9,
            "nav": float(close),
            "shares_outstanding": 1000,
            "underlying_index_name": "IDX",
        }
        for day, close in zip(days, closes, strict=True)
    ]


def _panel(series: Mapping[str, Sequence[float]], *, start: date | None = None, end: date | None = None) -> Panel:
    n = len(next(iter(series.values())))
    days = _bdays(n)
    rows: list[dict[str, Any]] = []
    for ticker, returns in series.items():
        rows.extend(_rows(ticker, returns, days))
    frame = pl.DataFrame(rows, schema=SCHEMA)
    return build_panel(frame, list(series), start or days[0], end or days[-1], PANEL_CFG)


def _random(n: int, seed: int) -> list[float]:
    return np.random.default_rng(seed).normal(0.0005, 0.01, n).tolist()


@dataclass(frozen=True)
class Toy:
    strategy_id: str
    family: str
    kind: Literal["ma", "lookahead", "flat"] = "ma"
    window: int = 3

    @property
    def params(self) -> Mapping[str, Any]:
        return {"kind": self.kind, "window": self.window}

    def targets(self, panel: Panel) -> pl.DataFrame:
        ordered = panel.frame.filter(pl.col("listed")).sort(["ticker", "date"])
        close = pl.col("close_ref")
        if self.kind == "lookahead":
            signal = close.shift(-1).over("ticker") > close
        elif self.kind == "ma":
            average = sum(close.shift(i).over("ticker") for i in range(self.window)) / self.window
            signal = close > average
        else:
            signal = pl.lit(False)
        return ordered.select(
            pl.col("date").alias("decision_date"),
            "ticker",
            pl.when(signal.fill_null(False)).then(pl.lit(0.5)).otherwise(pl.lit(0.0)).alias("weight"),
        )


def _cfg(**changes: Any) -> ResearchConfig:
    return replace(BASE, **changes)


def _gather(
    strategy: Toy,
    panel: Panel,
    cfg: ResearchConfig,
    registry: Path,
    *,
    track: str = "arena_t",
    trial_tags: Mapping[str, str] | None = None,
) -> Evidence:
    return gather_evidence(
        strategy,
        panel,
        SIM,
        EVAL,
        cfg,
        track=track,
        capital_krw=CAPITAL,
        registry=registry,
        recorded_at=NOW,
        trial_tags=trial_tags,
    )


def _registry_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_with(monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict[str, Any]], None]) -> ResearchConfig:
    section: dict[str, Any] = dict(_BASE_RESEARCH)
    mutate(section)
    data: Mapping[str, Any] = {"research": section}
    monkeypatch.setattr(rs, "load_config", lambda name: data)
    return load_research_config()


def _base_report(strategy_id: str = "s1") -> EvaluationReport:
    days = _bdays(260)
    net = np.random.default_rng(3).normal(0.002, 0.01, 260)
    return evaluate(strategy_id, "h", days, net, np.full(260, 0.001), EVAL, 1)


def _passing(strategy_id: str = "s1", family: str = "F", *, full_mean: float = 0.002) -> Evidence:
    base = _base_report(strategy_id)
    years = (YearStat(2025, 130, full_mean), YearStat(2026, 130, full_mean))
    full = replace(base, edge=replace(base.edge, mean_gross=full_mean), by_year=years)
    stress = replace(
        base,
        exceedance=replace(base.exceedance, lift=tuple(0.1 for _ in base.exceedance.thresholds)),
    )
    edge = replace(base.edge, p_value_adjusted=0.01)
    return Evidence(
        strategy_id=strategy_id,
        family=family,
        discovery_dn=replace(base, edge=edge),
        holdout_dn=replace(base, edge=edge),
        full_dn=full,
        holdout_stress=stress,
        regimes={},
        missing=(),
    )


BAND = (1.3, 2.0)


def _full(evidence: Evidence) -> EvaluationReport:
    assert evidence.full_dn is not None
    return evidence.full_dn


def test_config_loads_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_research_config() == ResearchConfig(
        discovery_start=date(2018, 1, 2),
        holdout_start=date(2025, 1, 2),
        regime_reference="069500",
        regime_window=20,
        regime_edges=(0.15, 0.25, 0.40),
        regime_min_sessions=120,
        stress_cost_bps=9.0,
        gate_alpha=0.05,
        gate_min_plateau_share=0.5,
        causality_cuts=3,
        annualization=252,
    )

    def _set(**changes: Any) -> Callable[[dict[str, Any]], None]:
        return lambda section: section.update(changes)

    def _drop(key: str) -> Callable[[dict[str, Any]], None]:
        return lambda section: section.pop(key)

    bad_mutations: list[Callable[[dict[str, Any]], None]] = [
        _set(extra=1),
        _drop("gate_alpha"),
        _set(discovery_start=date(2025, 1, 2)),
        _set(holdout_start=datetime(2025, 1, 2)),
        _set(holdout_start="2025-01-02"),
        _set(regime_reference=""),
        _set(regime_reference=69500),
        _set(regime_window=1),
        _set(regime_window=20.0),
        _set(regime_min_sessions=0),
        _set(regime_min_sessions=True),
        _set(regime_edges=[0.15, 0.25]),
        _set(regime_edges=[0.15, 0.25, 0.20]),
        _set(regime_edges=[0.0, 0.25, 0.40]),
        _set(regime_edges=[0.15, 0.25, "x"]),
        _set(regime_edges="0.15"),
        _set(stress_cost_bps=-1.0),
        _set(stress_cost_bps=True),
        _set(gate_alpha=0.0),
        _set(gate_alpha=1.0),
        _set(gate_min_plateau_share=1.5),
        _set(gate_min_plateau_share=-0.1),
        _set(causality_cuts=True),
        _set(causality_cuts=0),
        _set(annualization=0),
    ]
    for mutate in bad_mutations:
        with pytest.raises(ConfigError):
            _load_with(monkeypatch, mutate)

    assert _load_with(monkeypatch, _set(stress_cost_bps=45, gate_alpha=0.1)).stress_cost_bps == 45.0

    monkeypatch.setattr(rs, "load_config", lambda name: {"research": "x"})
    with pytest.raises(ConfigError):
        load_research_config()
    monkeypatch.setattr(rs, "load_config", lambda name: {})
    with pytest.raises(ConfigError):
        load_research_config()


def test_regime_labels_are_causal_and_bucketed() -> None:
    amplitudes = [0.005, 0.0123, 0.02, 0.04]
    seg = 40
    returns = [a * (-1) ** t for a in amplitudes for t in range(seg)]
    labels = regime_labels(_panel({REF: returns}), BASE)["regime"].to_list()
    window = BASE.regime_window
    assert labels[:window] == [None] * window
    assert [labels[j * seg + 25] for j in range(len(amplitudes))] == ["low", "mid", "high", "extreme"]


def test_regime_spike_is_invisible_before_it_and_visible_after() -> None:
    calm = [0.005 * (-1) ** t for t in range(80)]
    spiked = list(calm)
    spiked[40] = 0.2
    before = regime_labels(_panel({REF: calm}), BASE)["regime"].to_list()
    after = regime_labels(_panel({REF: spiked}), BASE)["regime"].to_list()
    assert after[40] == before[40] == "low"
    assert after[41] == "extreme"
    assert after[60] == "extreme"
    assert before[60] == "low"


def test_regime_labels_absent_reference() -> None:
    panel = _panel({"AAA": _random(60, 1)})
    labels = regime_labels(panel, BASE)
    assert labels.height == len(panel.sessions)
    assert labels["regime"].null_count() == labels.height


def test_truncate_panel_bounds() -> None:
    days = _bdays(30)
    panel = build_panel(
        pl.DataFrame(_rows("AAA", _random(30, 2), days), schema=SCHEMA),
        ["AAA", "ZZZ"],
        days[0],
        days[-1],
        PANEL_CFG,
    )
    cut = days[10]
    truncated = truncate_panel(panel, cut)
    assert truncated.sessions == tuple(days[:11])
    assert truncated.frame["date"].max() <= cut
    assert truncated.start == panel.start
    assert truncated.missing_tickers == ("ZZZ",)
    with pytest.raises(ValueError, match="outside"):
        truncate_panel(panel, days[0] - timedelta(days=1))
    with pytest.raises(ValueError, match="outside"):
        truncate_panel(panel, days[-1] + timedelta(days=1))


def test_causality_check_passes_causal_strategy() -> None:
    panel = _panel({"AAA": _random(60, 4), REF: _random(60, 5)})
    strategy = Toy("ma", "F")
    result = check_causality(strategy, panel, BASE)
    assert result.equals(strategy.targets(panel))


def test_causality_check_catches_lookahead() -> None:
    panel = _panel({"AAA": [0.01] * 60, REF: [0.01] * 60})
    with pytest.raises(CausalityError, match="look"):
        check_causality(Toy("look", "F", kind="lookahead"), panel, BASE)


def test_causality_check_rejects_empty_panel() -> None:
    days = _bdays(10)
    frame = pl.DataFrame(_rows("AAA", _random(10, 6), days), schema=SCHEMA)
    empty = build_panel(frame, ["AAA"], days[-1] + timedelta(days=1), days[-1] + timedelta(days=5), PANEL_CFG)
    with pytest.raises(ValueError, match="no sessions"):
        check_causality(Toy("ma", "F"), empty, BASE)


def test_evidence_registers_every_view(tmp_path: Path) -> None:
    panel = _panel({"AAA": _random(240, 7), REF: _random(240, 8)})
    cfg = _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[120], regime_min_sessions=5)
    registry = tmp_path / "registry.jsonl"
    evidence = _gather(Toy("s1", "F1"), panel, cfg, registry)
    assert evidence.discovery_dn is not None
    assert evidence.holdout_dn is not None
    assert evidence.full_dn is not None
    assert evidence.holdout_stress is not None
    assert evidence.regimes
    expected = {
        "arena_t/discovery_dn",
        "arena_t/holdout_dn",
        "arena_t/full_dn",
        "arena_t/holdout_stress",
        *(f"arena_t/regime_{r}" for r in evidence.regimes),
    }
    lines = _registry_lines(registry)
    assert len(lines) == len(expected)
    assert {line["strategy_family"] for line in lines} == expected


def test_holdout_haircut_counts_all_strategies(tmp_path: Path) -> None:
    panel = _panel({"AAA": _random(240, 9), REF: _random(240, 10)})
    cfg = _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[120])
    registry = tmp_path / "registry.jsonl"
    first = _gather(Toy("s1", "F1"), panel, cfg, registry)
    second = _gather(Toy("s2", "F2", window=2), panel, cfg, registry)
    assert first.holdout_dn is not None and first.holdout_dn.edge.n_trials == 1
    assert second.holdout_dn is not None and second.holdout_dn.edge.n_trials == 2
    rerun = _gather(Toy("s1", "F1"), panel, cfg, registry)
    assert rerun.holdout_dn is not None and rerun.holdout_dn.edge.n_trials == 2


def test_views_use_the_right_sessions(tmp_path: Path) -> None:
    panel = _panel({"AAA": _random(240, 11), REF: _random(240, 12)})
    cfg = _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[120])
    evidence = _gather(Toy("s1", "F1"), panel, cfg, tmp_path / "registry.jsonl")
    assert evidence.discovery_dn is not None
    assert evidence.holdout_dn is not None
    assert evidence.full_dn is not None
    assert evidence.discovery_dn.last_session < cfg.holdout_start
    assert evidence.holdout_dn.first_session >= cfg.holdout_start
    assert (
        evidence.full_dn.edge.n_sessions == evidence.discovery_dn.edge.n_sessions + evidence.holdout_dn.edge.n_sessions
    )

    strategy = Toy("s1", "F1")
    targets = strategy.targets(panel)
    base = simulate(panel, targets, SIM, capital_krw=CAPITAL, mode="actual")
    stressed = simulate(
        panel, targets, replace(SIM, cost_bps_per_side=cfg.stress_cost_bps), capital_krw=CAPITAL, mode="actual"
    )
    on_holdout = np.array([day >= cfg.holdout_start for day in base.sessions])
    assert base.sessions == stressed.sessions
    assert np.all(stressed.costs[on_holdout] >= base.costs[on_holdout])


def test_short_view_becomes_missing(tmp_path: Path) -> None:
    panel = _panel({"AAA": _random(240, 13), REF: _random(240, 14)})
    cfg = _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[-10])
    evidence = _gather(Toy("s1", "F1"), panel, cfg, tmp_path / "registry.jsonl")
    assert evidence.holdout_dn is None
    assert evidence.holdout_stress is None
    assert any(entry.startswith("holdout_dn:") for entry in evidence.missing)


def test_degenerate_views_become_missing(tmp_path: Path) -> None:
    panel = _panel({"AAA": _random(240, 15), REF: _random(240, 16)})
    cfg = _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[120], regime_min_sessions=5)
    evidence = _gather(Toy("flat", "F1", kind="flat"), panel, cfg, tmp_path / "registry.jsonl")
    assert evidence.discovery_dn is None
    assert evidence.full_dn is None
    assert evidence.holdout_stress is None
    assert dict(evidence.regimes) == {}
    assert any(entry.startswith("full_dn:") for entry in evidence.missing)


def test_short_regimes_are_skipped(tmp_path: Path) -> None:
    panel = _panel({"AAA": _random(240, 17), REF: _random(240, 18)})
    cfg = _cfg(
        discovery_start=panel.sessions[0],
        holdout_start=panel.sessions[120],
        regime_min_sessions=10_000,
    )
    evidence = _gather(Toy("s1", "F1"), panel, cfg, tmp_path / "registry.jsonl")
    assert dict(evidence.regimes) == {}
    for regime in REGIMES:
        assert any(entry.startswith(f"regime_{regime}:") for entry in evidence.missing)


def test_gate_all_pass_and_fail_closed() -> None:
    evidence = _passing()
    verdict = gate(evidence, BASE, BAND, [evidence])
    assert verdict.passed
    assert verdict.lift_at_band == pytest.approx((0.1, 0.1))
    assert all(c.passed for c in verdict.criteria)

    full = _full(evidence)
    one_year = replace(
        evidence,
        full_dn=replace(full, by_year=(YearStat(2025, 130, 0.01), YearStat(2026, 130, -0.001))),
    )
    loo = next(c for c in gate(one_year, BASE, BAND, [one_year]).criteria if c.name == "leave_one_year_out")
    assert not loo.passed
    assert loo.value == pytest.approx(-0.001)

    no_holdout = replace(evidence, holdout_dn=None)
    verdict = gate(no_holdout, BASE, BAND, [no_holdout])
    edge = next(c for c in verdict.criteria if c.name == "holdout_edge")
    assert edge.value is None and not edge.passed
    assert not verdict.passed

    no_discovery = replace(evidence, discovery_dn=None)
    assert not gate(no_discovery, BASE, BAND, [no_discovery]).passed

    no_stress = replace(evidence, holdout_stress=None)
    verdict = gate(no_stress, BASE, BAND, [no_stress])
    assert verdict.lift_at_band is None
    assert all(c.value is None for c in verdict.criteria if c.name.startswith("lift_"))
    assert not verdict.passed

    no_full = replace(evidence, full_dn=None)
    verdict = gate(no_full, BASE, BAND, [no_full])
    assert next(c for c in verdict.criteria if c.name == "leave_one_year_out").value is None

    single_year = replace(evidence, full_dn=replace(full, by_year=(YearStat(2025, 260, 0.01),)))
    loo_single = next(
        c for c in gate(single_year, BASE, BAND, [single_year]).criteria if c.name == "leave_one_year_out"
    )
    assert loo_single.value is None and not loo_single.passed


def test_gate_requires_band_on_grid() -> None:
    evidence = _passing()
    with pytest.raises(ValueError, match="exceedance grid"):
        gate(evidence, BASE, (1.2, 2.0), [evidence])


def test_gate_plateau_rejects_knife_edge() -> None:
    evaluated = _passing("s1", "F", full_mean=0.002)
    lone = [evaluated] + [_passing(f"s{i}", "F", full_mean=-0.001) for i in range(2, 5)]
    share = next(c for c in gate(evaluated, BASE, BAND, lone).criteria if c.name == "plateau")
    assert share.value == pytest.approx(0.25)
    assert not share.passed

    half = [evaluated, _passing("s2", "F", full_mean=0.002), _passing("s3", "F", full_mean=-0.001)]
    half.append(replace(_passing("s4", "F"), full_dn=None))
    share = next(c for c in gate(evaluated, BASE, BAND, half).criteria if c.name == "plateau")
    assert share.value == pytest.approx(0.5)
    assert share.passed

    other_family = [evaluated, _passing("s9", "G", full_mean=0.002)]
    share = next(c for c in gate(evaluated, BASE, BAND, other_family).criteria if c.name == "plateau")
    assert share.value == pytest.approx(1.0)

    with pytest.raises(ValueError, match="must include"):
        gate(evaluated, BASE, BAND, [_passing("s2", "F")])


def _verdict(strategy_id: str, passed: bool, lift: float | None) -> GateVerdict:
    return GateVerdict(strategy_id, passed, (), None if lift is None else (lift, lift + 0.01))


def test_select_prefers_validated_then_lift() -> None:
    chosen = select([_verdict("A", True, 0.02), _verdict("B", False, 0.10)])
    assert chosen is not None
    assert (chosen.strategy_id, chosen.label, chosen.lift_at_band_lo) == ("A", "VALIDATED", 0.02)

    fallback = select([_verdict("A", False, None), _verdict("B", False, 0.10), _verdict("C", False, 0.05)])
    assert fallback is not None
    assert (fallback.strategy_id, fallback.label, fallback.lift_at_band_lo) == ("B", "UNVALIDATED", 0.10)

    assert select([_verdict("A", True, None), _verdict("B", False, None)]) is None
    assert select([]) is None

    tie = select([_verdict("b", True, 0.05), _verdict("a", True, 0.05)])
    assert tie is not None and tie.strategy_id == "a"


@dataclass(frozen=True)
class _Hold:
    strategy_id: str = "hold"
    family: str = "H"

    @property
    def params(self) -> Mapping[str, Any]:
        return {"kind": "hold"}

    def targets(self, panel: Panel) -> pl.DataFrame:
        return pl.DataFrame(
            {"decision_date": [panel.sessions[5]], "ticker": ["AAA"], "weight": [1.0]},
            schema={"decision_date": pl.Date, "ticker": pl.String, "weight": pl.Float64},
        )


def test_drift_is_removed_within_each_window(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    returns = np.concatenate([rng.normal(0.0, 0.01, 100), rng.normal(0.01, 0.01, 100)]).tolist()
    panel = _panel({"AAA": returns, REF: _random(200, 3)})
    cfg = _cfg(holdout_start=panel.sessions[100], regime_min_sessions=1000)
    evidence = gather_evidence(
        _Hold(), panel, SIM, EVAL, cfg, track="t", capital_krw=CAPITAL, registry=tmp_path / "r.jsonl", recorded_at=NOW
    )
    assert evidence.holdout_dn is not None
    assert evidence.full_dn is not None
    assert abs(evidence.holdout_dn.edge.mean_gross) < 1e-12
    assert evidence.holdout_dn.edge.p_value > 0.4


def test_regime_views_use_regime_floor_not_eval_floor(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    calm = rng.normal(0.0, 0.002, 150)
    wild = rng.normal(0.0, 0.05, 50)
    panel = _panel({"AAA": _random(200, 5), REF: np.concatenate([calm, wild]).tolist()})
    cfg = _cfg(holdout_start=panel.sessions[100], regime_min_sessions=30, regime_window=5)
    strict_eval = replace(EVAL, min_sessions=100)
    evidence = gather_evidence(
        Toy("s", "F"),
        panel,
        SIM,
        strict_eval,
        cfg,
        track="t",
        capital_krw=CAPITAL,
        registry=tmp_path / "r.jsonl",
        recorded_at=NOW,
    )
    assert "extreme" in evidence.regimes
    assert evidence.regimes["extreme"].edge.n_sessions < strict_eval.min_sessions


def test_empty_tags_keep_digests(tmp_path: Path) -> None:
    panel = _panel({"AAA": _random(240, 19), REF: _random(240, 20)})
    cfg = _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[120], regime_min_sessions=5)
    strategy = Toy("s1", "F1")
    plain = tmp_path / "plain.jsonl"
    none_tags = tmp_path / "none.jsonl"
    empty_tags = tmp_path / "empty.jsonl"
    _gather(strategy, panel, cfg, plain)
    _gather(strategy, panel, cfg, none_tags, trial_tags=None)
    _gather(strategy, panel, cfg, empty_tags, trial_tags={})
    expected = _registry_lines(plain)
    assert _registry_lines(none_tags) == expected
    assert _registry_lines(empty_tags) == expected


def test_tags_create_new_trials(tmp_path: Path) -> None:
    panel = _panel({"AAA": _random(240, 21), REF: _random(240, 22)})
    cfg = _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[120])
    registry = tmp_path / "registry.jsonl"
    _gather(Toy("s1", "F1"), panel, cfg, registry)
    _gather(Toy("s1", "F1"), panel, cfg, registry, trial_tags={"panel": "x"})
    assert count_trials(registry, "arena_t/holdout_dn") == 2


@pytest.mark.parametrize("tags", [{"view": "x"}, {"strategy": "x"}, {"mode": "x"}, {"cost_bps": "1"}, {"kind": "x"}])
def test_colliding_tag_rejected(tmp_path: Path, tags: Mapping[str, str]) -> None:
    panel = _panel({"AAA": _random(240, 23), REF: _random(240, 24)})
    cfg = _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[120])
    registry = tmp_path / "registry.jsonl"
    with pytest.raises(ValueError, match="collide"):
        _gather(Toy("s1", "F1"), panel, cfg, registry, trial_tags=tags)
    assert not registry.exists()


class _Silent:
    strategy_id = "silent"
    family = "F_silent"
    params: Mapping[str, Any] = {"kind": "silent"}

    def targets(self, panel: Panel) -> pl.DataFrame:
        return pl.DataFrame(schema={"decision_date": pl.Date, "ticker": pl.String, "weight": pl.Float64})


def test_strategy_without_decisions_has_no_evidence(tmp_path: Path) -> None:
    panel = _panel({"AAA": _random(300, 21), REF: _random(300, 22)})
    registry = tmp_path / "registry.jsonl"
    evidence = gather_evidence(
        _Silent(),
        panel,
        SIM,
        EVAL,
        _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[150]),
        track="arena_t",
        capital_krw=CAPITAL,
        registry=registry,
        recorded_at=NOW,
    )
    assert (evidence.discovery_dn, evidence.holdout_dn, evidence.full_dn, evidence.holdout_stress) == (None,) * 4
    assert evidence.regimes == {}
    assert evidence.missing and evidence.missing[0].startswith("all views:")
    assert not registry.exists()


@dataclass(frozen=True)
class SegToy(Toy):
    execution_model: ClassVar[Literal["segment"]] = "segment"

    def targets(self, panel: Panel) -> pl.DataFrame:
        daily = super().targets(panel)
        intraday = daily.with_columns(pl.lit("intraday").alias("leg"))
        overnight = daily.with_columns(pl.lit("overnight").alias("leg"), pl.lit(0.0).alias("weight"))
        return pl.concat([intraday, overnight])


SEG_COST = SegmentConfig(cost_bps_per_side=6.5, stress_cost_bps=19.5)


def _segment_panel(series: Mapping[str, Sequence[float]]) -> Panel:
    n = len(next(iter(series.values())))
    days = _bdays(n)
    rows: list[dict[str, Any]] = []
    for ticker, returns in series.items():
        rows.extend(_rows(ticker, returns, days))
    frame = pl.DataFrame(rows, schema=SCHEMA).with_columns(
        pl.col("close").shift(1).over("ticker").fill_null(pl.col("close")).alias("open")
    )
    return build_panel(frame, list(series), days[0], days[-1], PANEL_CFG)


def test_segment_strategy_uses_segment_costs_per_view(tmp_path: Path) -> None:
    panel = _segment_panel({"AAA": _random(240, 31), REF: _random(240, 32)})
    cfg = _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[120], regime_min_sessions=5)
    registry = tmp_path / "registry.jsonl"
    evidence = gather_evidence(
        SegToy("seg", "F1"),
        panel,
        SIM,
        EVAL,
        cfg,
        track="arena_t",
        capital_krw=CAPITAL,
        registry=registry,
        recorded_at=NOW,
        segment_cfg=SEG_COST,
    )
    assert evidence.family == "F1"
    lines = _registry_lines(registry)
    assert lines
    for line in lines:
        view = line["strategy_family"].split("/")[1]
        stressed = view == "holdout_stress" or view.startswith("regime_")
        expected = SEG_COST.stress_cost_bps if stressed else SEG_COST.cost_bps_per_side
        assert line["params"]["cost_bps"] == expected


def test_segment_strategy_requires_segment_config(tmp_path: Path) -> None:
    panel = _segment_panel({"AAA": _random(240, 33), REF: _random(240, 34)})
    cfg = _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[120])
    registry = tmp_path / "registry.jsonl"
    with pytest.raises(ValueError, match="needs a segment config"):
        _gather(SegToy("seg", "F1"), panel, cfg, registry)
    assert not registry.exists()


def test_daily_registry_lines_ignore_segment_config(tmp_path: Path) -> None:
    panel = _panel({"AAA": _random(240, 35), REF: _random(240, 36)})
    cfg = _cfg(discovery_start=panel.sessions[0], holdout_start=panel.sessions[120], regime_min_sessions=5)
    plain = tmp_path / "plain.jsonl"
    with_segment = tmp_path / "with_segment.jsonl"
    _gather(Toy("s1", "F1"), panel, cfg, plain)
    gather_evidence(
        Toy("s1", "F1"),
        panel,
        SIM,
        EVAL,
        cfg,
        track="arena_t",
        capital_krw=CAPITAL,
        registry=with_segment,
        recorded_at=NOW,
        segment_cfg=SEG_COST,
    )
    assert _registry_lines(with_segment) == _registry_lines(plain)


def test_gather_evidence_vol_windows_optional(tmp_path: Path) -> None:
    panel = _panel({"AAA": _random(400, 5)})
    cfg = _cfg()
    plain = _gather(Toy("s1", "F1"), panel, cfg, tmp_path / "plain.jsonl")
    assert plain.vol_windows == ()
    prize = PrizeLineConfig(reference="AAA", lookback=20, ks=(0.9, 1.3), twin_draws=5, seed=3)
    evidence = gather_evidence(
        Toy("s1", "F1"),
        panel,
        SIM,
        EVAL,
        cfg,
        track="arena_t",
        capital_krw=CAPITAL,
        registry=tmp_path / "prize.jsonl",
        recorded_at=NOW,
        prize_cfg=prize,
    )
    periods = tuple(view.period for view in evidence.vol_windows)
    assert periods
    assert periods == tuple(p for p in ("discovery", "holdout") if p in periods)
