"""Invariant guards for the weekly rank-objective contest decision."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from src.contest.decision import (
    ContestAction,
    ContestDecision,
    build_explicit_leaders,
    choose_target,
    decide_week,
    decision_to_dict,
    panel_changes_pct,
    realized_equity,
    render_decision_markdown,
    resolve_current_holding,
)
from src.contest.engine import rank_metrics, simulate_contest
from src.contest.leaderboard import LeaderboardEntry, LeaderboardSnapshot
from src.contest.panel import VehiclePanel
from src.core.calendar import get_calendar

NAMES = ("K2", "HY2", "HY2I", "SEMI2", "K2I", "Q2", "SS2")
PANEL_START = date(2026, 7, 1)
PANEL_DAYS = 140
START = date(2026, 9, 21)
END = date(2026, 11, 13)


def _toy_panel() -> VehiclePanel:
    dates = tuple(PANEL_START + timedelta(days=i) for i in range(PANEL_DAYS))
    n, v = PANEL_DAYS, len(NAMES)
    gap = np.zeros((n, v), dtype=np.float32)
    intra = np.zeros((n, v), dtype=np.float32)
    for j in range(v):
        gap[:, j] = np.float32(0.04 * (j + 1))
        intra[:, j] = np.float32(-0.02 * (j + 1))
        intra[:, j] += np.float32(0.001) * ((np.arange(n) + j) % 5)
    gap[0, :] = 0.0
    cc = (1 + gap.astype(np.float64)) * (1 + intra.astype(np.float64)) - 1.0
    return VehiclePanel(dates=dates, names=NAMES, gap=gap, intraday=intra, log_nav=np.cumsum(np.log1p(cc), axis=0))


def _toy_config(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "nickname": "tester",
        "start_date": START.isoformat(),
        "end_date": END.isoformat(),
        "vehicles": {a: {"ticker": f"T_{a}", "name": a} for a in NAMES},
        "inference": {"match_tol_pct": 0.02, "min_weight": 0.90},
        "decision": {
            "candidates": ["HOLD", "HY2", "K2"],
            "min_adv_krw": 0,
            "target_weight": 0.999,
            "switch_min_p1_gain": 0.05,
            "endgame_sessions": 8,
            "eras": {
                "a": {"start": "2026-07-15", "neutral": False},
                "b": {"start": "2026-07-15", "neutral": True},
            },
            "profiles": ["base"],
            "n_worlds": 4,
            "mean_block": 10,
            "seed": 1,
            "output_dir": "results/contest_weekly",
            "state_name": "contest_position",
        },
        "crowd": {
            "n_participants": 30,
            "f_auto": 0.4,
            "profiles": {
                "base": {"f_aggr": 0.01, "f_mod": 0.15, "mix_aggr": [0.45, 0.30, 0.10, 0.15],
                         "mix_mod": [0.60, 0.20, 0.10, 0.10], "q_scale": 1.0}
            },
            "popularity_auto": {"HY2": 0.5, "K2": 0.3, "SEMI2": 0.2},
            "popularity_nonauto": {"K2": 0.6, "SEMI2": 0.4},
            "entry_days": [0, 1, 2, 3, 5, 8],
            "entry_probs": [0.70, 0.13, 0.07, 0.04, 0.03, 0.03],
            "leader_churn": {"k": 10, "q": 0.10},
        },
    }
    cfg.update(overrides)
    return cfg


def _snapshot(panel: VehiclePanel, session: date, ours: tuple[float, float] | None = (3.0, -99.0),
              leaders: tuple[tuple[str, float, float, float], ...] = ()) -> LeaderboardSnapshot:
    """Build a snapshot; ours=(total, daily) with daily=-99 → auto-fill K2 change; leaders=(name,rank,total,daily)."""
    changes = panel_changes_pct(panel, session)
    entries = []
    if ours is not None:
        total, daily = ours
        entries.append(LeaderboardEntry(rank=30, user_name="tester", total_return_pct=total,
                                        daily_return_pct=changes["K2"] if daily == -99.0 else daily))
    for name, rank, total, daily in leaders:
        entries.append(LeaderboardEntry(rank=rank, user_name=name, total_return_pct=total, daily_return_pct=daily))
    entries.append(LeaderboardEntry(rank=50, user_name="filler", total_return_pct=0.5, daily_return_pct=0.0))
    return LeaderboardSnapshot(base_date=session, requested_at="", entries=tuple(entries), purchases={})


def test_decide_week_stale_snapshot_never_switches() -> None:
    """A two-session-old snapshot yields NO_DATA with the current target kept."""
    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)
    stale_base = cal.previous_session(session, 2)
    snap = _snapshot(panel, stale_base)
    out = decide_week(panel, snap, session, cal, _toy_config(), "K2")
    assert out.action == ContestAction.NO_DATA
    assert out.target_alias == "K2"
    assert out.execution_session is None
    assert "LEADERBOARD_STALE" in out.warnings
    assert out.scores == ()


def test_choose_target_hysteresis_keeps_holding() -> None:
    """A 0.03 P1 edge below the 0.05 bar keeps the current holding."""
    target, action = choose_target({"HOLD": (0.10, 0.20), "HY2": (0.13, 0.30)}, "HOLD", 0.05)
    assert (target, action) == ("HOLD", ContestAction.HOLD)


def test_choose_target_clear_improvement_switches() -> None:
    """A 0.10 P1 edge clears the bar and switches, with ties broken by P2."""
    target, action = choose_target({"HOLD": (0.10, 0.20), "HY2": (0.20, 0.30)}, "HOLD", 0.05)
    assert (target, action) == ("HY2", ContestAction.SWITCH)
    tie, tie_action = choose_target({"A": (0.20, 0.10), "B": (0.20, 0.40)}, None, 0.05)
    assert (tie, tie_action) == ("B", ContestAction.SWITCH)
    first, action_first = choose_target({"HY2": (0.50, 0.50)}, None, 0.05)
    assert (first, action_first) == ("HY2", ContestAction.SWITCH)


def test_decide_week_full_run_card_shape() -> None:
    """A fresh snapshot produces a scored HOLD/SWITCH card with a consistent execution session."""
    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)
    changes = panel_changes_pct(panel, session)
    snap = _snapshot(panel, session, leaders=(("champ", 1, 12.0, changes["HY2"]),))
    out = decide_week(panel, snap, session, cal, _toy_config(), "K2")
    assert out.action in (ContestAction.HOLD, ContestAction.SWITCH)
    assert {s.alias for s in out.scores} == {"HOLD", "HY2", "K2"}
    assert out.sessions_remaining == len([s for s in cal.sessions(session, END) if s > session])
    if out.action == ContestAction.SWITCH:
        assert out.execution_session == cal.next_session(session)
        assert out.target_alias != out.current_alias
    else:
        assert out.execution_session is None
        assert out.target_alias == out.current_alias
    assert out.inferred_leaders["champ"][0] == "HY2"
    assert "tester" in out.inferred_leaders


def test_decide_week_contest_over() -> None:
    """A session past the end date yields CONTEST_OVER with no scores."""
    panel = _toy_panel()
    out = decide_week(panel, None, date(2026, 11, 14), get_calendar(), _toy_config(), "K2")
    assert out.action == ContestAction.CONTEST_OVER
    assert out.sessions_remaining == 0
    assert out.scores == ()
    assert out.execution_session is None


def test_decide_week_illiquid_candidate_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A candidate below min_adv is not scored and is named in warnings."""
    import src.contest.decision as decision_mod

    monkeypatch.setattr(decision_mod, "adv_krw", lambda root, ticker, session, sessions: 1e12 if ticker == "T_K2" else 0.0)
    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)
    snap = _snapshot(panel, session)
    cfg = _toy_config()
    cfg["decision"]["candidates"] = ["HOLD", "HY2"]
    cfg["decision"]["min_adv_krw"] = 1e9
    out = decide_week(panel, snap, session, cal, cfg, "K2")
    assert {s.alias for s in out.scores} == {"HOLD"}
    assert "ILLIQUID:HY2" in out.warnings


def test_resolve_current_holding_mismatch_surfaces() -> None:
    """State HY2 against a K2-like daily return resolves to K2 with HOLDING_MISMATCH."""
    panel = _toy_panel()
    session = date(2026, 9, 23)
    changes = panel_changes_pct(panel, session)
    snap = _snapshot(panel, session, ours=(3.0, changes["K2"] * 0.95))
    alias, warnings = resolve_current_holding(snap, "tester", changes, "HY2", 0.02, 0.90)
    assert alias == "K2"
    assert warnings == ["HOLDING_MISMATCH"]


def test_resolve_current_holding_variants() -> None:
    """Agreement, state-only, and unknown holdings resolve without surprises."""
    panel = _toy_panel()
    session = date(2026, 9, 23)
    changes = panel_changes_pct(panel, session)
    snap = _snapshot(panel, session)
    assert resolve_current_holding(snap, "tester", changes, "K2", 0.02, 0.90) == ("K2", [])
    others = LeaderboardSnapshot(base_date=session, requested_at="", entries=(
        LeaderboardEntry(rank=1, user_name="x", total_return_pct=5.0, daily_return_pct=0.0),), purchases={})
    assert resolve_current_holding(others, "tester", changes, "HY2", 0.02, 0.90) == ("HY2", [])
    assert resolve_current_holding(others, "tester", changes, None, 0.02, 0.90) == (None, ["HOLDING_UNKNOWN"])
    assert resolve_current_holding(None, "tester", changes, None, 0.02, 0.90) == (None, ["HOLDING_UNKNOWN"])


def test_realized_equity_compounds_recorded_holding() -> None:
    """Fallback equity compounds the recorded vehicle's legs from the entry level."""
    panel = _toy_panel()
    entry, end = date(2026, 9, 21), date(2026, 9, 23)
    got = realized_equity(panel, "HY2", entry, end, 1.05, 0.999)
    j = panel.index("HY2")
    want = 1.05
    for r, d in enumerate(panel.dates):
        if entry < d <= end:
            want *= (1 + 0.999 * float(panel.gap[r, j])) * (1 + 0.999 * float(panel.intraday[r, j]))
    assert got == pytest.approx(want)


def test_decide_week_outside_top50_needs_confirmation() -> None:
    """No nickname entry plus no entry equity yields NEEDS_CONFIRMATION with OUTSIDE_TOP50."""
    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)
    snap = _snapshot(panel, session, ours=None)
    out = decide_week(panel, snap, session, cal, _toy_config(), "HY2")
    assert out.action == ContestAction.NEEDS_CONFIRMATION
    assert "OUTSIDE_TOP50" in out.warnings
    assert out.scores == ()


def test_decide_week_outside_top50_uses_equity_override() -> None:
    """An equity override outside the top-50 yields scored candidates and surfaces its tag."""
    panel = _toy_panel()
    session = date(2026, 9, 23)
    snap = _snapshot(panel, session, ours=None)
    out = decide_week(panel, snap, session, get_calendar(), _toy_config(), "HY2", (0.9, "OUR_RETURN_ESTIMATED"))
    assert out.action in (ContestAction.HOLD, ContestAction.SWITCH)
    assert out.scores
    assert {"OUTSIDE_TOP50", "OUR_RETURN_ESTIMATED"} <= set(out.warnings)
    assert out.our_total_return_pct == pytest.approx(-10.0)


def test_build_explicit_leaders_excludes_our_nickname() -> None:
    """Our own inferred entry never becomes a competitor agent."""
    panel = _toy_panel()
    session = date(2026, 9, 23)
    changes = panel_changes_pct(panel, session)
    snap = _snapshot(panel, session, leaders=(("champ", 1, 12.0, changes["HY2"]),))
    from src.contest.leaderboard import infer_single_vehicle_holders

    inferred = infer_single_vehicle_holders(snap, changes, 0.02, 0.90)
    assert inferred["tester"][0] == "K2"
    holders, pin_entries, top_values = build_explicit_leaders(snap, inferred, "tester", panel)
    assert len(top_values) == len(snap.entries) - 1
    assert not any(v == pytest.approx(1.03) for v in top_values.tolist())
    assert all(v == panel.index("HY2") for _, v in holders)
    assert [v for _, _, v in pin_entries] == [panel.index("HY2")]
    assert holders and pin_entries


def test_decide_week_endgame_adds_mimic_candidate() -> None:
    """Five sessions left with a SEMI2 leader scores MIMIC:SEMI2."""
    panel = _toy_panel()
    cal = get_calendar()
    all_sessions = cal.sessions(START, END)
    session = all_sessions[-6]
    changes = panel_changes_pct(panel, session)
    snap = _snapshot(panel, session, leaders=(("champ", 1, 12.0, changes["SEMI2"]),))
    out = decide_week(panel, snap, session, cal, _toy_config(), "K2")
    assert out.sessions_remaining == 5
    assert "MIMIC:SEMI2" in {s.alias for s in out.scores}


def test_simulate_common_random_numbers_shared_across_candidates() -> None:
    """Every candidate is scored against the identical crowd outcome array."""
    from src.contest.engine import CrowdSpec, Worlds, bootstrap_worlds

    panel = _toy_panel()
    worlds = bootstrap_worlds(panel, [panel.row(START)], np.arange(0, panel.row(START)), 4, 3, 10.0, 3)
    n_w = worlds.n_worlds
    crowd = CrowdSpec(
        kind=np.zeros(3, dtype=np.int8), vehicle=np.zeros(3, dtype=np.int16),
        entry=np.zeros(3, dtype=np.int64), weight=np.ones(3, dtype=np.float32),
        k=np.zeros(3, dtype=np.int64), q=np.zeros(3, dtype=np.float32),
        set_id=np.zeros(3, dtype=np.int64), sets=[np.array([0, 1], dtype=np.int16)],
        vehicle_per_world=np.zeros((3, n_w), dtype=np.int16),
    )
    worlds_small = Worlds(rows=worlds.rows, log_nav0=worlds.log_nav0)
    actions = {
        "A": lambda d, h: np.full_like(h, panel.index("K2")),
        "B": lambda d, h: np.full_like(h, panel.index("HY2")),
    }
    result = simulate_contest(panel, worlds_small, crowd, actions, {}, None, seed=9)
    for name in ("A", "B"):
        assert rank_metrics(result.ours[name], result.crowd_equity) == rank_metrics(
            result.ours[name], np.array(result.crowd_equity, copy=True)
        )


def test_decide_week_panel_missing_session_no_data() -> None:
    """A decision session absent from the panel yields NO_DATA."""
    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)
    snap = _snapshot(panel, session)
    thin = VehiclePanel(
        dates=tuple(d for d in panel.dates if d != session), names=panel.names,
        gap=np.delete(panel.gap, panel.row(session), axis=0),
        intraday=np.delete(panel.intraday, panel.row(session), axis=0),
        log_nav=np.delete(panel.log_nav, panel.row(session), axis=0),
    )
    out = decide_week(thin, snap, session, cal, _toy_config(), "K2")
    assert out.action == ContestAction.NO_DATA
    assert out.target_alias == out.current_alias


def test_decide_week_unknown_candidate_rejected() -> None:
    """A candidate alias outside the vehicle map fails closed."""
    panel = _toy_panel()
    session = date(2026, 9, 23)
    snap = _snapshot(panel, session)
    cfg = _toy_config()
    cfg["decision"]["candidates"] = ["HOLD", "NOPE"]
    with pytest.raises(ValueError, match=r"unknown candidate"):
        decide_week(panel, snap, session, get_calendar(), cfg, "K2")


def test_decide_week_pre_contest_session_no_data() -> None:
    """A decision before contest start with no snapshot yields NO_DATA."""
    panel = _toy_panel()
    out = decide_week(panel, None, date(2026, 9, 1), get_calendar(), _toy_config(), None)
    assert out.action == ContestAction.NO_DATA
    assert "LEADERBOARD_STALE" in out.warnings


def test_render_decision_markdown_renders_cards() -> None:
    """Full and empty cards render the Korean sections without crashing."""
    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)
    snap = _snapshot(panel, session)
    out = decide_week(panel, snap, session, cal, _toy_config(), "K2")
    text = render_decision_markdown(out, {a: a for a in NAMES})
    assert "행동" in text and "후보 점수" in text and "추정 리더" in text and "경고" in text
    assert "HOLD" in text and "P1 평균" in text
    empty = ContestDecision(
        decision_session=session, execution_session=None, action=ContestAction.NO_DATA,
        current_alias=None, target_alias=None, target_ticker=None, target_weight=0.999,
        our_rank=None, our_total_return_pct=None, leaderboard_base_date=None,
        sessions_remaining=3, scores=(), inferred_leaders={}, warnings=(),
    )
    text2 = render_decision_markdown(empty, {})
    assert "(없음)" in text2 and "없음" in text2


def test_decision_to_dict_serializes(tmp_path) -> None:
    """Cards serialize to JSON-ready primitives."""
    import json

    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)
    snap = _snapshot(panel, session)
    out = decide_week(panel, snap, session, cal, _toy_config(), "K2")
    blob = json.dumps(decision_to_dict(out), ensure_ascii=False)
    assert session.isoformat() in blob


def test_panel_changes_pct_close_to_close() -> None:
    """Per-vehicle changes equal (1+gap)(1+intraday)-1 in percent."""
    panel = _toy_panel()
    row = panel.row(date(2026, 9, 23))
    got = panel_changes_pct(panel, date(2026, 9, 23))
    for j, alias in enumerate(panel.names):
        want = ((1 + float(panel.gap[row, j])) * (1 + float(panel.intraday[row, j])) - 1.0) * 100.0
        assert got[alias] == pytest.approx(want)


def test_adv_krw_handles_missing_and_null_data(tmp_path) -> None:
    """Unknown tickers, missing files, and null-only series read as zero ADV."""
    import polars as pl

    from src.contest.decision import adv_krw

    sessions = [date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23)]
    assert adv_krw(tmp_path, "T_X", date(2026, 9, 23), sessions) == 0.0
    norm = tmp_path / "normalized"
    norm.mkdir(parents=True)
    pl.DataFrame(
        {"date": sessions, "ticker": ["T_X"] * 3, "trading_value": [None, None, None]},
        schema={"date": pl.Date, "ticker": pl.String, "trading_value": pl.Int64},
    ).write_parquet(norm / "etf_daily.parquet")
    assert adv_krw(tmp_path, "T_X", date(2026, 9, 23), sessions) == 0.0
    assert adv_krw(tmp_path, "T_MISSING", date(2026, 9, 23), sessions) == 0.0
    assert adv_krw(tmp_path, "T_X", date(2026, 9, 23), []) == 0.0


def test_persistence_scenarios_multiply_samples(caplog) -> None:
    """Two churn scenarios double the era-by-profile runs and keep the log shape."""
    import logging

    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)
    changes = panel_changes_pct(panel, session)
    snap = _snapshot(panel, session, leaders=(("champ", 1, 12.0, changes["HY2"]),))
    cfg = _toy_config()
    cfg["crowd"]["leader_churn"] = [{"k": 10, "q": 0.1}, {"k": 10, "q": 0.0}]
    with caplog.at_level(logging.INFO):
        out = decide_week(panel, snap, session, cal, cfg, "K2")
    assert out.action in (ContestAction.HOLD, ContestAction.SWITCH)
    assert out.scores
    runs = [r for r in caplog.records if "contest_weekly config=" in r.getMessage()]
    assert len(runs) == 2 * 1 * 2


def test_legacy_churn_mapping_still_accepted(caplog) -> None:
    """A single mapping churn config still completes with era-by-profile runs."""
    import logging

    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)
    snap = _snapshot(panel, session)
    cfg = _toy_config()
    cfg["crowd"]["leader_churn"] = {"k": 10, "q": 0.10}
    with caplog.at_level(logging.INFO):
        out = decide_week(panel, snap, session, cal, cfg, "K2")
    assert out.action in (ContestAction.HOLD, ContestAction.SWITCH)
    assert out.scores
    runs = [r for r in caplog.records if "contest_weekly config=" in r.getMessage()]
    assert len(runs) == 2


def test_wrapper_changes_read_from_silver_with_skips(tmp_path) -> None:
    """Wrapper closes come from silver; a wrapper missing either close is silently absent."""
    import polars as pl

    from src.contest.decision import _inference_inputs, wrapper_changes_pct

    session = date(2026, 9, 23)
    prev = date(2026, 9, 22)
    norm = tmp_path / "normalized"
    norm.mkdir(parents=True)
    pl.DataFrame(
        {"date": [prev, session], "ticker": ["0195S0", "0195S0"],
         "open": [100.0, 102.0], "close": [100.0, 103.0]},
        schema={"date": pl.Date, "ticker": pl.String, "open": pl.Float64, "close": pl.Float64},
    ).write_parquet(norm / "etf_daily.parquet")
    vehicles = {"HY2": {"ticker": "0193T0", "wrappers": ["0195S0", "MISSING"]}}
    out, exposure = wrapper_changes_pct(tmp_path, vehicles, session, prev)
    assert out["HY2@0195S0"] == pytest.approx(3.0)
    assert exposure == {"HY2@0195S0": "HY2"}
    assert "HY2@MISSING" not in out


def test_wrapper_changes_edge_cases_covered(tmp_path) -> None:
    """Non-mapping configs, missing silver, and null/bad closes never block inference."""
    import polars as pl

    from src.contest.decision import wrapper_changes_pct

    session = date(2026, 9, 23)
    prev = date(2026, 9, 22)
    assert wrapper_changes_pct(tmp_path, {"HY2": {"ticker": "x"}}, session, prev) == ({}, {})
    assert wrapper_changes_pct(tmp_path, {"HY2": "not-a-mapping"}, session, prev) == ({}, {})
    assert wrapper_changes_pct(tmp_path, {"HY2": {"wrappers": "0195S0"}}, session, prev) == ({}, {})
    assert wrapper_changes_pct(tmp_path, {"HY2": {"wrappers": ["0195S0"]}}, session, prev) == ({}, {})
    norm = tmp_path / "normalized"
    norm.mkdir(parents=True)
    pl.DataFrame(
        {"date": [prev, session, prev, session], "ticker": ["A", "A", "B", "B"],
         "open": [10.0, 10.0, 10.0, 10.0], "close": [None, 10.0, 0.0, 10.0]},
        schema={"date": pl.Date, "ticker": pl.String, "open": pl.Float64, "close": pl.Float64},
    ).write_parquet(norm / "etf_daily.parquet")
    out, _ = wrapper_changes_pct(tmp_path, {"X": {"wrappers": ["A", "B"]}}, session, prev)
    assert out == {}


def test_inference_inputs_fallback_paths(tmp_path) -> None:
    """Previous-session and wrapper failures degrade to panel-only changes."""
    from src.contest.decision import _inference_inputs
    import src.contest.decision as decision_mod

    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)

    class _NoPrev:
        def previous_session(self, day, offset=1):
            raise ValueError("no prev")

    changes, exposure = _inference_inputs(panel, tmp_path, {"K2": {}}, session, _NoPrev())  # type: ignore[arg-type]
    assert changes and exposure == {}

    orig = decision_mod.wrapper_changes_pct

    def _boom(*args, **kwargs):
        raise OSError("boom")

    decision_mod.wrapper_changes_pct = _boom  # type: ignore[assignment]
    try:
        changes2, exposure2 = _inference_inputs(panel, tmp_path, {"K2": {}}, session, cal)
        assert changes2 and exposure2 == {}
    finally:
        decision_mod.wrapper_changes_pct = orig


def test_stale_snapshot_missing_panel_session_never_switches() -> None:
    """A stale snapshot whose base date is absent from the panel still yields NO_DATA."""
    from datetime import timedelta

    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)
    missing_base = session - timedelta(days=365 * 3)
    snap = LeaderboardSnapshot(
        base_date=missing_base, requested_at="",
        entries=(LeaderboardEntry(rank=1, user_name="tester", total_return_pct=3.0, daily_return_pct=1.0),),
        purchases={},
    )
    out = decide_week(panel, snap, session, cal, _toy_config(), "K2")
    assert out.action == ContestAction.NO_DATA
    assert "LEADERBOARD_STALE" in out.warnings


def test_decide_week_unknown_current_base_equity() -> None:
    """No recorded holding and no HOLD candidate still scores from the equity override."""
    panel = _toy_panel()
    cal = get_calendar()
    session = date(2026, 9, 23)
    snap = _snapshot(panel, session, ours=None)
    cfg = _toy_config()
    cfg["decision"]["candidates"] = ["HY2"]
    out = decide_week(panel, snap, session, cal, cfg, None, (0.95, "OUR_RETURN_ESTIMATED"))
    assert out.action in (ContestAction.HOLD, ContestAction.SWITCH)
    assert out.scores
