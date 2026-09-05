from datetime import date


def test_window_opportunities_ceiling_and_breadth() -> None:
    from src.tournament.attainability import window_opportunities

    # Given: 4 sessions, horizon 2 -> only window start index 0 is emittable (needs index 0+1+2 = 3)
    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    open_by_session = [
        {"A": 100.0, "B": 100.0},
        {"A": 100.0, "B": 100.0},
        {"A": 120.0, "B": 110.0},
        {"A": 150.0, "B": 110.0},
    ]
    candidates = {sessions[0]: ["A", "B"], sessions[1]: [], sessions[2]: ["A"], sessions[3]: ["A"]}

    # When
    opps = window_opportunities(open_by_session, sessions, candidates, horizon=2)

    # Then: window starts at sessions[1], entry at open index 1, exit at open index 3
    assert len(opps) == 1
    assert opps[0].window_start == sessions[1]
    assert opps[0].breadth == 2
    assert abs(opps[0].ceiling - 0.50) < 1e-12


def test_capture_rate_returns_none_below_min_attainable() -> None:
    from src.tournament.attainability import WindowOpportunity, attainability_curve, capture_rate

    # Given: 4 windows, 2 of which had a >40% ceiling; the strategy captured 1 of those 2
    d = [date(2026, 1, i) for i in (5, 6, 7, 8)]
    opps = [
        WindowOpportunity(window_start=d[0], breadth=2, ceiling=0.80),
        WindowOpportunity(window_start=d[1], breadth=2, ceiling=0.50),
        WindowOpportunity(window_start=d[2], breadth=1, ceiling=0.10),
        WindowOpportunity(window_start=d[3], breadth=0, ceiling=None),
    ]
    returns = [0.60, 0.05, 0.02, 0.00]

    # When
    capture, n = capture_rate(returns, opps, 0.40, min_attainable=1)
    blocked, n_blocked = capture_rate(returns, opps, 0.40, min_attainable=5)
    curve = attainability_curve(opps, [0.40])

    # Then
    assert n == 2
    assert abs(capture - 0.5) < 1e-12
    assert blocked is None and n_blocked == 2
    assert abs(curve[0.40] - 0.5) < 1e-12


def test_build_attainability_summary_wires_capture_and_curve() -> None:
    from src.tournament.attainability import build_attainability_summary

    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    open_map = {
        sessions[0]: {"A": 100.0, "B": 100.0},
        sessions[1]: {"A": 100.0, "B": 100.0},
        sessions[2]: {"A": 120.0, "B": 110.0},
        sessions[3]: {"A": 150.0, "B": 110.0},
    }
    candidates = {sessions[0]: ["A", "B"], sessions[1]: [], sessions[2]: ["A"], sessions[3]: ["A"]}
    payload = build_attainability_summary(
        sessions=sessions,
        open_map=open_map,
        candidates_by_session=candidates,
        window_returns=[0.60],
        horizon=2,
        thresholds=[0.40],
        min_attainable_windows=1,
    )
    assert abs(float(payload["attainability"]["0.4"]) - 1.0) < 1e-12
    assert abs(float(payload["capture"]["0.4"]) - 1.0) < 1e-12
    assert payload["n_attainable"]["0.4"] == 1
    assert float(payload["breadth_mean"]) == 2.0


def test_build_attainability_summary_empty_sessions_returns_null_capture() -> None:
    from src.tournament.attainability import build_attainability_summary

    payload = build_attainability_summary(
        sessions=[],
        open_map={},
        candidates_by_session={},
        window_returns=[],
        horizon=36,
        thresholds=[0.40],
        min_attainable_windows=30,
    )
    assert payload["capture"]["0.4"] is None
    assert payload["attainability"]["0.4"] == 0.0


def test_window_opportunities_rejects_non_positive_horizon() -> None:
    from src.tournament.attainability import window_opportunities

    assert window_opportunities([], [], {}, horizon=0) == ()


def test_load_attainability_config_reads_gates_yaml() -> None:
    from src.tournament.attainability import load_attainability_config

    thresholds, min_att, min_disc = load_attainability_config()
    assert 0.40 in thresholds
    assert min_att >= 1
    assert min_disc >= 0


def test_candidates_by_session_from_cache_uses_universe_when_scores_missing() -> None:
    from datetime import date
    from types import SimpleNamespace

    from src.tournament.attainability import candidates_by_session_from_cache

    d0 = date(2026, 1, 5)
    cache = SimpleNamespace(scores={}, universes={d0: SimpleNamespace(tickers=["C"])})
    assert candidates_by_session_from_cache([d0], cache)[d0] == ("C",)


def test_enrich_backtest_run_artifacts_updates_meta_and_summary() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.tournament.attainability import enrich_backtest_run_artifacts

    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    panel = pl.DataFrame(
        {"date": sessions, "ticker": ["A"] * 3, "open": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0]}
    )
    meta: dict[str, object] = {}
    summary: dict[str, object] = {}
    meta, summary = enrich_backtest_run_artifacts(
        meta,
        summary,
        calendar=SimpleNamespace(sessions=lambda _s, _e: sessions),
        panel=panel,
        engine=SimpleNamespace(),
        model=SimpleNamespace(),
        case_config=SimpleNamespace(start=sessions[0], end=sessions[-1]),
        rolling=SimpleNamespace(starts=sessions[:1], returns=(0.0,)),
        horizon=1,
        shared_cache=SimpleNamespace(scores={sessions[0]: {"A": 1.0}}, universes={}, open_map={d: {"A": 1.0} for d in sessions}),
    )
    assert "phantom_sessions" in meta
    assert "attainability" in summary


def test_window_opportunities_skips_non_positive_entry_prices() -> None:
    from src.tournament.attainability import window_opportunities

    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    open_by = [
        {"A": 100.0},
        {"A": 0.0},
        {"A": 100.0},
        {"A": 200.0},
    ]
    opps = window_opportunities(open_by, sessions, {sessions[0]: ["A"]}, horizon=1)
    assert opps[0].breadth == 0


def test_load_attainability_config_falls_back_on_io_error(tmp_path) -> None:
    from src.tournament.attainability import load_attainability_config

    missing = tmp_path / "missing.yaml"
    thresholds, min_att, min_disc = load_attainability_config(str(missing))
    assert 0.40 in thresholds
    assert min_att == 30
    assert min_disc == 5


def test_build_attainability_summary_tolerates_bad_window_returns() -> None:
    from collections.abc import Sequence

    from src.tournament.attainability import build_attainability_summary

    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]

    class _BadReturns(Sequence[float]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> float:
            raise RuntimeError("bad return")

    payload = build_attainability_summary(
        sessions=sessions,
        open_map={
            sessions[0]: {"A": 100.0},
            sessions[1]: {"A": 100.0},
            sessions[2]: {"A": 150.0},
            sessions[3]: {"A": 200.0},
        },
        candidates_by_session={sessions[0]: ["A"]},
        window_returns=_BadReturns(),
        horizon=1,
        thresholds=[0.40],
        min_attainable_windows=1,
    )
    assert payload["capture"]["0.4"] == 0.0


def test_window_opportunity_and_capture_exception_paths() -> None:
    from collections.abc import Mapping, Sequence
    from types import SimpleNamespace
    from unittest.mock import patch

    from src.tournament.attainability import (
        WindowOpportunity,
        attainability_curve,
        backtest_attainability_payload,
        build_attainability_summary,
        capture_rate,
        candidates_by_session_from_cache,
        enrich_backtest_run_artifacts,
        load_attainability_config,
        window_opportunities,
    )

    sessions = [date(2026, 1, 5)]
    assert window_opportunities([{"A": "bad"}], sessions, {sessions[0]: ["A"]}, horizon="bad") == ()  # type: ignore[arg-type]
    opps = (WindowOpportunity(window_start=sessions[0], breadth=1, ceiling=0.5),)
    assert attainability_curve(opps, ["bad"]) == {}
    assert capture_rate([0.5], opps, "bad")[0] is None


def test_load_attainability_config_invalid_sections(tmp_path) -> None:
    from src.tournament.attainability import load_attainability_config

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "attainability:\n  thresholds: bad\n  min_attainable_windows: bad\n  min_effective_discordant: bad\n",
        encoding="utf-8",
    )
    assert load_attainability_config(str(bad))[0] == (0.30, 0.40, 0.50, 0.60)


def test_candidates_by_session_from_cache_exception_paths() -> None:
    from types import SimpleNamespace

    from src.tournament.attainability import candidates_by_session_from_cache

    d0 = date(2026, 1, 5)

    class _BadUni:
        @property
        def tickers(self):
            raise RuntimeError("bad")

    cache = SimpleNamespace(scores={}, universes={d0: _BadUni()})
    assert candidates_by_session_from_cache([d0], cache)[d0] == ()


def test_backtest_attainability_payload_builds_cache_and_open_map() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    import polars as pl

    from src.tournament.attainability import backtest_attainability_payload

    sessions = [date(2026, 1, 5), date(2026, 1, 6)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["A", "A"], "open": [1.0, 1.0], "close": [1.0, 1.0]})
    with patch("src.backtest.session_cache.build_session_cache", side_effect=RuntimeError("cache fail")):
        payload = backtest_attainability_payload(
            calendar=SimpleNamespace(sessions=lambda _s, _e: sessions),
            panel=panel,
            engine=SimpleNamespace(),
            model=SimpleNamespace(),
            case_config=SimpleNamespace(start=sessions[0], end=sessions[-1]),
            rolling=SimpleNamespace(starts=sessions, returns=(0.0,)),
            horizon=1,
            shared_cache=None,
        )
    assert "attainability" in payload


def test_enrich_backtest_run_artifacts_handles_calendar_failure() -> None:
    from types import SimpleNamespace

    import polars as pl

    from src.tournament.attainability import enrich_backtest_run_artifacts

    sessions = [date(2026, 1, 5), date(2026, 1, 6)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["A", "A"], "open": [1.0, 1.0], "close": [1.0, 1.0]})
    meta, summary = enrich_backtest_run_artifacts(
        {},
        {},
        calendar=SimpleNamespace(sessions=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no cal"))),
        panel=panel,
        engine=SimpleNamespace(),
        model=SimpleNamespace(),
        case_config=SimpleNamespace(start=sessions[0], end=sessions[-1]),
        rolling=SimpleNamespace(starts=sessions, returns=(0.0,)),
        horizon=1,
        shared_cache=None,
    )
    assert meta["phantom_sessions"] == []


def test_window_opportunities_tolerates_bad_maps() -> None:
    from collections.abc import Mapping, Sequence

    from src.tournament.attainability import window_opportunities

    class _BadMap(Mapping[str, float]):
        def __getitem__(self, key: str) -> float:
            raise RuntimeError("bad")

        def __iter__(self):
            return iter(["A"])

        def __len__(self) -> int:
            return 1

    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    open_by = [{}, _BadMap(), {"A": 110.0}]
    opps = window_opportunities(open_by, sessions, {sessions[0]: ["A"]}, horizon=1)
    assert opps[0].breadth == 0

    class _BadCandidates(Mapping[date, Sequence[str]]):
        def __getitem__(self, key: date) -> Sequence[str]:
            raise RuntimeError("bad")

        def __iter__(self):
            return iter(())

        def __len__(self) -> int:
            return 0

    opps2 = window_opportunities([{}, {"A": 100.0}, {"A": 110.0}], sessions, _BadCandidates(), horizon=1)
    assert len(opps2) == 1


def test_attainability_helper_exception_paths(tmp_path) -> None:
    from types import SimpleNamespace

    from src.tournament.attainability import (
        _finite_positive,
        build_attainability_summary,
        capture_rate,
        candidates_by_session_from_cache,
        load_attainability_config,
        window_opportunities,
    )

    class _BadPrice:
        def __float__(self) -> float:
            raise ValueError("bad")

    assert _finite_positive(_BadPrice()) is None
    assert capture_rate([0.5], (), 0.40, min_attainable="bad") == (None, 0)

    class _BadReturns:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> float:
            raise RuntimeError("boom")

    opps = window_opportunities(
        [{}, {"A": 100.0}, {"A": 150.0}],
        [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)],
        {date(2026, 1, 5): ["A"]},
        horizon=1,
    )
    assert capture_rate(_BadReturns(), opps, 0.40, min_attainable=1) == (0.0, 1)

    plain = tmp_path / "plain.yaml"
    plain.write_text("attainability: plain\n", encoding="utf-8")
    assert load_attainability_config(str(plain))[0] == (0.30, 0.40, 0.50, 0.60)

    d0 = date(2026, 1, 5)
    cache = SimpleNamespace(scores={d0: object()}, universes={})
    assert candidates_by_session_from_cache([d0], cache)[d0] == ()

    empty = build_attainability_summary(
        sessions=[date(2026, 1, 5)],
        open_map={},
        candidates_by_session={},
        window_returns=[],
        horizon="bad",  # type: ignore[arg-type]
        thresholds=[0.40],
    )
    assert empty["capture"]["0.4"] is None


def test_build_attainability_summary_handles_bad_min_windows_and_breadth() -> None:
    from unittest.mock import patch

    from src.tournament.attainability import WindowOpportunity, build_attainability_summary

    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    payload = build_attainability_summary(
        sessions=sessions,
        open_map={sessions[0]: {"A": 100.0}, sessions[1]: {"A": 100.0}, sessions[2]: {"A": 150.0}},
        candidates_by_session={sessions[0]: ["A"]},
        window_returns=[0.5],
        horizon=1,
        thresholds=[0.40],
        min_attainable_windows="bad",  # type: ignore[arg-type]
    )
    assert payload["capture"]["0.4"] is not None

    bad_opp = WindowOpportunity(window_start=sessions[0], breadth="bad", ceiling=0.5)  # type: ignore[arg-type]
    with patch("src.tournament.attainability.window_opportunities", return_value=(bad_opp,)):
        payload2 = build_attainability_summary(
            sessions=sessions,
            open_map={},
            candidates_by_session={},
            window_returns=[0.1],
            horizon=1,
            thresholds=[0.40],
            min_attainable_windows=1,
        )
    assert payload2["breadth_mean"] == 0.0


def test_backtest_payload_falls_back_when_open_map_build_fails() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    import polars as pl

    from src.tournament.attainability import backtest_attainability_payload

    sessions = [date(2026, 1, 5), date(2026, 1, 6)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["A", "A"], "open": [1.0, 1.0], "close": [1.0, 1.0]})
    with patch("src.backtest.session_cache._build_open_map", side_effect=RuntimeError("open map")):
        payload = backtest_attainability_payload(
            calendar=SimpleNamespace(sessions=lambda _s, _e: sessions),
            panel=panel,
            engine=SimpleNamespace(),
            model=SimpleNamespace(),
            case_config=SimpleNamespace(start=sessions[0], end=sessions[-1]),
            rolling=SimpleNamespace(starts=sessions, returns=(0.0,)),
            horizon=1,
            shared_cache=None,
        )
    assert payload["breadth_mean"] == 0.0
