"""Invariant guards for the daily shadow card (leader-mirror policy)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.contest.daily import (
    DailyShadowCard,
    GateVerdict,
    QuoteBar,
    ShadowAction,
    StreakHolder,
    archive_quotes,
    build_daily_card,
    card_metric_row,
    daily_to_dict,
    evaluate_gates,
    fetch_quotes,
    infer_streak_holders,
    load_quotes,
    mirror_target,
    render_daily_markdown,
)
from src.contest.leaderboard import LeaderboardEntry, LeaderboardSnapshot, archive_leaderboard
from src.core.calendar import get_calendar

CAL = get_calendar()
SESS = CAL.sessions(date(2026, 9, 21), date(2026, 10, 20))

NICK = "tester"

HY2_K = ("0193T0", 1000.0, 1027.27)
HY2_T = ("0195S0", 1000.0, 1026.96)
HY2I_Q = ("0197X0", 1000.0, 975.0)
Q2_Q = ("233740", 1000.0, 1032.0)


def _shadow_cfg(**over: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": True,
        "start_session": SESS[0].isoformat(),
        "gate_session": SESS[4].isoformat(),
        "adopt_from_session": SESS[6].isoformat() if len(SESS) > 6 else SESS[-1].isoformat(),
        "output_dir": "results/contest_daily",
        "quotes_dir": "contest/quotes",
        "yahoo_suffix": ".KS",
        "timeout_s": 20,
        "long_alias": "HY2",
        "inverse_alias": "HY2I",
        "exposures": {"HY2": ["0193T0", "0195S0"], "HY2I": ["0197X0"], "Q2": ["233740"]},
        "min_streak": 2,
        "review_streak": 3,
        "top_n": 20,
        "gates": {
            "g1_min_identified_share": 0.5,
            "g1_window": 5,
            "g1_min_days": 3,
            "g2_max_daily_churn": 0.15,
            "g3_max_no_data_days": 2,
        },
    }
    cfg.update(over)
    return cfg


def _config(shadow: dict[str, Any]) -> dict[str, Any]:
    return {
        "nickname": NICK,
        "start_date": "2026-09-21",
        "end_date": "2026-11-13",
        "inference": {"match_tol_pct": 0.02, "min_weight": 0.90},
        "shadow": shadow,
    }


def _entry(rank: int, name: str, total: float, daily: float) -> LeaderboardEntry:
    return LeaderboardEntry(rank=rank, user_name=name, total_return_pct=total, daily_return_pct=daily)


def _snap(day: date, entries: list[LeaderboardEntry]) -> LeaderboardSnapshot:
    return LeaderboardSnapshot(base_date=day, requested_at="", entries=tuple(entries), purchases={})


def _put_snapshot(root: Path, day: date, rows: list[tuple[int, str, float, float]]) -> None:
    archive_leaderboard(
        {
            "etfRankTotal": {
                "baseDt": day.strftime("%Y%m%d"),
                "reqDateTime": day.strftime("%Y%m%d") + "160500",
                "data": [
                    {"rank": r, "userName": u, "totalReturnRate": t, "dailyReturnRate": d}
                    for r, u, t, d in rows
                ],
            }
        },
        root,
    )


def _bar(ticker: str, prev: float, close: float) -> QuoteBar:
    return QuoteBar(ticker=ticker, prev_close=prev, open=prev, close=close)


def _put_quotes(root: Path, day: date, bars: list[QuoteBar]) -> None:
    archive_quotes(root, day, {bar.ticker: bar for bar in bars})


def _std_bars() -> list[QuoteBar]:
    return [
        _bar(ticker, prev, close)
        for ticker, prev, close in (HY2_K, HY2_T, HY2I_Q, Q2_Q)
    ]


def test_mirror_target_holds_side_opposite_higher_leader() -> None:
    """A higher long leader at 1.101 vs inverse at 1.078 mirrors into the inverse alias."""
    snap = _snap(
        SESS[2],
        [_entry(1, "long_user", 10.1, 2.714), _entry(2, "inv_user", 7.8, -2.49),
         _entry(50, "floor", 1.7, 0.0), _entry(30, NICK, 3.0, 0.5)],
    )
    holders = {
        "long_user": StreakHolder("long_user", "HY2", 0.99, 2),
        "inv_user": StreakHolder("inv_user", "HY2I", 0.99, 3),
    }
    target, long_leader, inverse_leader = mirror_target(snap, holders, NICK, "HY2", "HY2I", 2)
    assert target == "HY2I"
    assert long_leader is not None and long_leader[0] == "long_user"
    assert long_leader[1] == pytest.approx(1.101)
    assert inverse_leader is not None and inverse_leader[0] == "inv_user"


def test_mirror_target_unidentified_side_valued_at_floor() -> None:
    """With only an inverse holder at 1.02 above the 1.017 floor, the mirror holds long."""
    snap = _snap(SESS[2], [_entry(1, "inv_user", 2.0, -2.49), _entry(50, "floor", 1.7, 0.0)])
    holders = {"inv_user": StreakHolder("inv_user", "HY2I", 0.99, 2)}
    target, long_leader, inverse_leader = mirror_target(snap, holders, NICK, "HY2", "HY2I", 2)
    assert target == "HY2"
    assert long_leader is None
    assert inverse_leader is not None and inverse_leader[1] == pytest.approx(1.02)


def test_mirror_target_ignores_short_streak_and_unknown_users() -> None:
    """Streak-1 holders, our own entry, and unknown users never become rival leaders."""
    snap = _snap(
        SESS[2],
        [_entry(1, NICK, 20.0, 2.714), _entry(2, "other", 5.0, 2.714), _entry(50, "floor", 1.7, 0.0)],
    )
    holders = {
        "flash": StreakHolder("flash", "HY2", 0.99, 1),
        NICK: StreakHolder(NICK, "HY2", 0.99, 5),
        "ghost": StreakHolder("ghost", "HY2", 0.99, 9),
        "other": StreakHolder("other", "HY2", 0.95, 2),
    }
    target, long_leader, _ = mirror_target(snap, holders, NICK, "HY2", "HY2I", 2)
    assert target == "HY2I"
    assert long_leader is not None and long_leader[0] == "other"


def _streak_history(
    days: list[date], dailies: list[float], ambiguous_middle: bool
) -> list[tuple[LeaderboardSnapshot, dict[str, float], dict[str, str]]]:
    changes_base = {"HY2@0193T0": 2.727, "HY2@0195S0": 2.696}
    exposure_of = {"HY2@0193T0": "HY2", "HY2@0195S0": "HY2"}
    history = []
    for i, day in enumerate(days):
        changes = dict(changes_base)
        expo = dict(exposure_of)
        if ambiguous_middle and i == 1:
            changes["Q2@233740"] = 3.026
            expo["Q2@233740"] = "Q2"
        snap = _snap(
            day,
            [_entry(1, "holder", 5.0, dailies[i]), _entry(30, NICK, 3.0, 0.1)],
        )
        history.append((snap, changes, expo))
    return history


def test_infer_streak_holders_counts_consecutive_exposures() -> None:
    """Three HY2 sessions (via either wrapper) give streak 3; an ambiguous middle resets to 1."""
    days = [SESS[0], SESS[1], SESS[2]]
    holders = infer_streak_holders(_streak_history(days, [2.72, 2.70, 2.714], False), 0.02, 0.90)
    assert holders["holder"].streak == 3
    assert holders["holder"].exposure == "HY2"
    broken = infer_streak_holders(_streak_history(days, [2.72, 2.714, 2.714], True), 0.02, 0.90)
    assert broken["holder"].streak == 1
    assert infer_streak_holders([], 0.02, 0.90) == {}


def test_archive_quotes_write_once_and_load_roundtrip(tmp_path: Path) -> None:
    """The first capture wins; later reruns return the same path with original content."""
    root = tmp_path / "quotes"
    first = {"0193T0": _bar("0193T0", 1000.0, 1027.27)}
    second = {"0193T0": _bar("0193T0", 1.0, 2.0)}
    path1 = archive_quotes(root, SESS[2], first)
    path2 = archive_quotes(root, SESS[2], second)
    assert path1 == path2
    loaded = load_quotes(root, SESS[2])
    assert loaded is not None and loaded["0193T0"].close == pytest.approx(1027.27)
    assert load_quotes(root, SESS[3]) is None


def test_load_quotes_absent_or_corrupt_returns_none(tmp_path: Path) -> None:
    """Missing files, non-JSON bodies, and misshapen records all read as None."""
    root = tmp_path / "quotes"
    assert load_quotes(root, SESS[2]) is None
    bad = root / f"{SESS[2].strftime('%Y%m%d')}.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not json", encoding="utf-8")
    assert load_quotes(root, SESS[2]) is None
    bad.write_text('{"session": "2026-09-29", "bars": [1, 2]}', encoding="utf-8")
    assert load_quotes(root, SESS[2]) is None


def test_archive_quotes_write_failure_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed atomic write leaves no partial file behind."""

    def _boom(self: object, *args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _boom)
    with pytest.raises(OSError, match="disk full"):
        archive_quotes(tmp_path / "quotes", SESS[2], {"T": _bar("T", 1.0, 2.0)})
    assert list((tmp_path / "quotes").iterdir()) == []


def _yahoo_payload(rows: list[tuple[date, float | None, float | None]]) -> dict[str, Any]:
    def _ts(day: date) -> int:
        return int(datetime(day.year, day.month, day.day, 12, 0, tzinfo=UTC).timestamp())

    return {
        "chart": {
            "result": [
                {
                    "timestamp": [_ts(d) for d, _, _ in rows],
                    "indicators": {
                        "quote": [
                            {"open": [o for _, o, _ in rows], "close": [c for _, _, c in rows]},
                        ]
                    },
                }
            ]
        }
    }


def test_fetch_quotes_skips_bad_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-positive closes, session-only series, and HTTP errors are skipped without aborting."""
    session = SESS[2]
    prev = SESS[1]
    older = SESS[0]
    good = _yahoo_payload(
        [(older, None, 99.0), (prev, 100.0, 102.0), (session, 100.0, 103.0)]
    )
    bad = _yahoo_payload([(prev, 100.0, 102.0), (session, 100.0, -5.0)])
    only = _yahoo_payload([(session, 100.0, 103.0)])

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "GOOD" in url:
            return httpx.Response(200, json=good)
        if "BAD" in url:
            return httpx.Response(200, json=bad)
        if "ONLY" in url:
            return httpx.Response(200, json=only)
        return httpx.Response(500, json={})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    bars = fetch_quotes(["GOOD", "BAD", "ERR", "ONLY"], session, ".KS", 20.0)
    assert sorted(bars) == ["GOOD"]
    assert bars["GOOD"].prev_close == pytest.approx(102.0)
    assert bars["GOOD"].close == pytest.approx(103.0)


def test_build_daily_card_no_data_paths(tmp_path: Path) -> None:
    """Missing snapshots or mirror-side quotes yield NO_DATA with the matching warning."""
    archive_root = tmp_path / "lb"
    archive_root.mkdir()
    quotes_root = tmp_path / "quotes"
    quotes_root.mkdir()
    config = _config(_shadow_cfg())
    stale = build_daily_card(SESS[2], CAL, archive_root, quotes_root, config, "HY2", None, [])
    assert stale.action == ShadowAction.NO_DATA
    assert stale.mirror_target is None
    assert stale.warnings == ("LEADERBOARD_STALE",)
    assert stale.our_equity is None and stale.our_equity_source is None
    manual = build_daily_card(SESS[2], CAL, archive_root, quotes_root, config, "HY2", (0.97, "OUR_RETURN_MANUAL"), [])
    assert manual.our_equity == pytest.approx(0.97)
    assert manual.our_equity_source == "manual"

    _put_snapshot(
        archive_root, SESS[2],
        [(1, "leader", 8.0, 2.714), (30, NICK, 3.0, 0.5), (50, "floor", 1.7, 0.0)],
    )
    _put_quotes(quotes_root, SESS[2], [_bar("233740", 1000.0, 1032.0)])
    missing = build_daily_card(SESS[2], CAL, archive_root, quotes_root, config, "HY2", None, [])
    assert missing.action == ShadowAction.NO_DATA
    assert missing.warnings == ("QUOTES_MISSING",)


def test_build_daily_card_record_and_causality(tmp_path: Path) -> None:
    """A fully quoted session records a mirror target; later-session corruption cannot change it."""
    archive_root = tmp_path / "lb"
    archive_root.mkdir()
    quotes_root = tmp_path / "quotes"
    quotes_root.mkdir()
    config = _config(_shadow_cfg())
    prev_day, day, next_day = SESS[0], SESS[1], SESS[2]
    for current in (prev_day, day, next_day):
        _put_snapshot(
            archive_root, current,
            [(1, "champ", 12.0, 2.714), (30, NICK, 3.0, 0.5), (50, "floor", 1.7, 0.0)],
        )
        _put_quotes(quotes_root, current, _std_bars())
    card = build_daily_card(day, CAL, archive_root, quotes_root, config, "HY2", None, [])
    assert card.action == ShadowAction.RECORD
    assert card.mirror_target in ("HY2", "HY2I")
    assert card.execution_session == CAL.next_session(day)
    assert card.our_equity == pytest.approx(1.03)
    assert card.our_equity_source == "leaderboard"
    assert card.identified_top_share is not None
    row = card_metric_row(card)
    assert row["session"] == day.isoformat() and row["action"] == "RECORD"
    text = render_daily_markdown(card)
    assert "그림자 추천" in text and "게이트 현황" in text
    assert "※ 그림자 모드: 실제 매매 기준은 주간 카드/현재 보유" in text
    before = daily_to_dict(card)

    quotes_path = quotes_root / f"{next_day.strftime('%Y%m%d')}.json"
    quotes_path.write_text('{"session": "corrupt", "bars": {}}', encoding="utf-8")
    snap_path = archive_root / next_day.strftime("%Y%m%d") / "etfRankTotal.json"
    snap_path.write_text('{"baseDt": "20990101", "data": []}', encoding="utf-8")
    again = build_daily_card(day, CAL, archive_root, quotes_root, config, "HY2", None, [])
    assert daily_to_dict(again) == before


def test_build_daily_card_contest_over(tmp_path: Path) -> None:
    """Sessions after end_date yield CONTEST_OVER with no recommendation."""
    archive_root = tmp_path / "lb"
    archive_root.mkdir()
    quotes_root = tmp_path / "quotes"
    quotes_root.mkdir()
    config = _config(_shadow_cfg())
    late = date(2026, 11, 14)
    _put_snapshot(archive_root, late, [(30, NICK, 3.0, 0.5)])
    with_snapshot = build_daily_card(late, CAL, archive_root, quotes_root, config, None, None, [])
    assert with_snapshot.action == ShadowAction.CONTEST_OVER
    assert with_snapshot.mirror_target is None
    assert with_snapshot.execution_session is None
    assert with_snapshot.our_equity == pytest.approx(1.03)
    without_snapshot = build_daily_card(date(2026, 11, 16), CAL, archive_root, quotes_root, config, None, None, [])
    assert without_snapshot.action == ShadowAction.CONTEST_OVER
    assert without_snapshot.our_equity is None


def test_evaluate_gates_pending_and_adoption() -> None:
    """PENDING before gate_session; ADOPT only when identification, churn, and freshness all pass."""
    shadow = _shadow_cfg()
    gate_day = date.fromisoformat(shadow["gate_session"])
    days = [SESS[i] for i in range(5)]
    shares = [0.6, 0.6, 0.4, 0.55, 0.7]
    rows = [
        {"session": day.isoformat(), "action": "RECORD",
         "identified_top_share": share, "daily_churn": 0.05}
        for day, share in zip(days, shares, strict=True)
    ]
    rows.append({"session": gate_day.isoformat(), "action": "NO_DATA",
                 "identified_top_share": None, "daily_churn": None})
    rows.append({"session": "bogus", "action": "RECORD",
                 "identified_top_share": 1.0, "daily_churn": 0.0})
    rows.append({"session": 12345, "action": "RECORD",
                 "identified_top_share": 1.0, "daily_churn": 0.0})
    pending, _ = evaluate_gates(rows, days[3], shadow)
    assert pending == GateVerdict.PENDING
    adopt, detail = evaluate_gates(rows, gate_day, shadow)
    assert adopt == GateVerdict.ADOPT
    assert detail["g1"] == {"value": 4, "pass": True}
    churny = [dict(row, daily_churn=0.2) if row.get("action") == "RECORD" else dict(row) for row in rows]
    assert evaluate_gates(churny, gate_day, shadow)[0] == GateVerdict.KEEP_STATIC
    stale = [
        *rows,
        {"session": gate_day.isoformat(), "action": "NO_DATA",
         "identified_top_share": None, "daily_churn": None},
        {"session": gate_day.isoformat(), "action": "NO_DATA",
         "identified_top_share": None, "daily_churn": None},
    ]
    assert evaluate_gates(stale, gate_day, shadow)[0] == GateVerdict.KEEP_STATIC


def test_evaluate_gates_empty_churn_and_date_rows() -> None:
    """RECORD rows without churn values fail G2; date-typed sessions are accepted."""
    shadow = _shadow_cfg()
    gate_day = date.fromisoformat(shadow["gate_session"])
    rows = [
        {"session": SESS[i], "action": "RECORD", "identified_top_share": 0.9}
        for i in range(3)
    ]
    verdict, detail = evaluate_gates(rows, gate_day, shadow)
    assert verdict == GateVerdict.KEEP_STATIC
    assert detail["g2"] == {"value": None, "pass": False}


def test_build_daily_card_churn_and_review(tmp_path: Path) -> None:
    """Four users identified on both sessions with one switch give churn 0.25 and an estimated equity."""
    archive_root = tmp_path / "lb"
    archive_root.mkdir()
    quotes_root = tmp_path / "quotes"
    quotes_root.mkdir()
    shadow = _shadow_cfg(top_n=4)
    config = _config(shadow)
    day_a, day_b = SESS[0], SESS[1]
    _put_snapshot(
        archive_root, day_a,
        [(1, "u1", 9.0, 2.714), (2, "u2", 8.0, 2.714), (3, "u3", 7.0, 2.714),
         (4, "u4", 6.0, 2.714), (50, "floor", 1.0, 0.0)],
    )
    _put_snapshot(
        archive_root, day_b,
        [(1, "u1", 9.5, 3.19), (2, "u2", 8.5, 2.714), (3, "u3", 7.5, 2.714),
         (4, "u4", 6.5, 2.714), (50, "floor", 1.0, 0.0)],
    )
    _put_quotes(quotes_root, day_a, _std_bars())
    _put_quotes(quotes_root, day_b, _std_bars())
    card = build_daily_card(day_b, CAL, archive_root, quotes_root, config, "HY2", (0.9, "OUR_RETURN_ESTIMATED"), [])
    assert card.daily_churn == pytest.approx(0.25)
    assert card.identified_top_share == pytest.approx(0.75)
    assert card.our_equity == pytest.approx(0.9)
    assert card.our_equity_source == "estimated"
    lonely = build_daily_card(day_b, CAL, archive_root, quotes_root, config, "HY2", None, [])
    assert lonely.our_equity is None


def test_build_daily_card_review_flag(tmp_path: Path) -> None:
    """A same-exposure leader ahead triggers review at streak 3 but not at streak 2."""
    archive_root = tmp_path / "lb"
    archive_root.mkdir()
    quotes_root = tmp_path / "quotes"
    quotes_root.mkdir()
    config = _config(_shadow_cfg())
    long_days = [SESS[0], SESS[1], SESS[2]]
    for current in long_days:
        _put_snapshot(
            archive_root, current,
            [(1, "champ", 12.0, 2.714), (30, NICK, 3.0, 0.5), (50, "floor", 1.7, 0.0)],
        )
        _put_quotes(quotes_root, current, _std_bars())
    assert build_daily_card(long_days[2], CAL, archive_root, quotes_root, config, "HY2", None, []).review is True
    assert build_daily_card(long_days[1], CAL, archive_root, quotes_root, config, "HY2", None, []).review is False
    assert build_daily_card(long_days[2], CAL, archive_root, quotes_root, config, "Q2", None, []).review is False


def test_build_daily_card_single_session_history(tmp_path: Path) -> None:
    """A calendar without earlier sessions still records with churn unknown."""
    archive_root = tmp_path / "lb"
    archive_root.mkdir()
    quotes_root = tmp_path / "quotes"
    quotes_root.mkdir()
    config = _config(_shadow_cfg())

    class _SingleCalendar:
        def previous_session(self, day: date, offset: int = 1) -> date:
            raise ValueError("no earlier session")

        def next_session(self, day: date, offset: int = 1) -> date:
            return date(2026, 10, 1)

    _put_snapshot(
        archive_root, SESS[2],
        [(1, "champ", 12.0, 2.714), (30, NICK, 3.0, 0.5), (50, "floor", 1.7, 0.0)],
    )
    _put_quotes(quotes_root, SESS[2], _std_bars())
    card = build_daily_card(SESS[2], _SingleCalendar(), archive_root, quotes_root, config, "HY2", None, [])
    assert card.action == ShadowAction.RECORD
    assert card.daily_churn is None
    assert card.execution_session == date(2026, 10, 1)


def test_quote_changes_skips_unquotable_bars() -> None:
    """Missing, non-positive, and non-finite bars narrow inference without aborting."""
    from src.contest.daily import _quote_changes

    exposures: dict[str, list[str]] = {"HY2": ["A", "B", "C", "D"], "HY2I": ["E"]}
    bars = {
        "A": _bar("A", 100.0, 102.0),
        "B": _bar("B", 0.0, 102.0),
        "C": _bar("C", 100.0, float("inf")),
    }
    changes, exposure_of = _quote_changes(bars, exposures)
    assert sorted(changes) == ["HY2@A"]
    assert exposure_of == {"HY2@A": "HY2"}


def test_daily_card_serializes() -> None:
    """Cards and NO_DATA cards render Korean markdown with the shadow disclaimer."""
    card = DailyShadowCard(
        session=SESS[2], execution_session=SESS[3], action=ShadowAction.RECORD,
        mirror_target="HY2I", long_leader=("a", 1.101), inverse_leader=("b", 1.078),
        our_equity=1.03, our_equity_source="leaderboard",
        identified_top_share=0.6, daily_churn=0.05, review=True,
        gate=GateVerdict.PENDING, gate_detail={"g1": {"value": 1, "pass": True}},
        holders=(StreakHolder("a", "HY2", 0.99, 2),), warnings=(),
    )
    blob = daily_to_dict(card)
    assert blob["mirror_target"] == "HY2I" and blob["gate"] == "PENDING"
    text = render_daily_markdown(card)
    assert "롱 선두" in text and "인버스 선두" in text and "REVIEW" in text
    empty = DailyShadowCard(
        session=SESS[2], execution_session=None, action=ShadowAction.NO_DATA,
        mirror_target=None, long_leader=None, inverse_leader=None,
        our_equity=None, our_equity_source=None,
        identified_top_share=None, daily_churn=None, review=False,
        gate=GateVerdict.PENDING, gate_detail={},
        holders=(), warnings=("LEADERBOARD_STALE",),
    )
    text2 = render_daily_markdown(empty)
    assert "없음" in text2 and "LEADERBOARD_STALE" in text2


def test_mirror_target_tie_prefers_long_alias() -> None:
    """With no identified leader on either side, both sides sit at the floor and the tie picks the long alias."""
    from src.contest.daily import mirror_target
    from src.contest.leaderboard import LeaderboardEntry, LeaderboardSnapshot

    snap = LeaderboardSnapshot(
        base_date=date(2026, 9, 23), requested_at="",
        entries=(LeaderboardEntry(rank=1, user_name="a", total_return_pct=5.0, daily_return_pct=0.0),),
        purchases={},
    )
    target, long_leader, inverse_leader = mirror_target(snap, {}, "me", "HY2", "HY2I", 2)
    assert (target, long_leader, inverse_leader) == ("HY2", None, None)
