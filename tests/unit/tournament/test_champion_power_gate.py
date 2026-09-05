def test_resolve_promotion_status_insufficient_power() -> None:
    from src.tournament.champion_eval import is_promotable, resolve_promotion_status

    # Given / When: all gates pass but the paired sample cannot see a difference
    weak = resolve_promotion_status(
        aggressive_status="PASS",
        conservative_status="PASS",
        loyo_status="PASS",
        artifact_integrity=True,
        n_effective_discordant=2,
        min_effective_discordant=5,
    )
    strong = resolve_promotion_status(
        aggressive_status="PASS",
        conservative_status="PASS",
        loyo_status="PASS",
        artifact_integrity=True,
        n_effective_discordant=9,
        min_effective_discordant=5,
    )
    failed = resolve_promotion_status(
        aggressive_status="FAIL",
        conservative_status="PASS",
        loyo_status="PASS",
        artifact_integrity=True,
        n_effective_discordant=9,
        min_effective_discordant=5,
    )

    # Then
    assert weak == "INSUFFICIENT_POWER"
    assert strong == "PROMOTE"
    assert failed == "RESEARCH_ONLY"
    assert is_promotable(
        aggressive_status="PASS",
        conservative_status="PASS",
        loyo_status="PASS",
        artifact_integrity=True,
    ) is True


import pytest


def test_discordant_window_mask_marks_differing_windows() -> None:
    from src.tournament.champion_eval import discordant_window_mask

    # Given: they differ only on session index 3
    cand = ["A", "A", "A", "B", "A", "A"]
    inc = ["A", "A", "A", "A", "A", "A"]

    # When: horizon 2 -> windows [0,2) [1,3) [2,4) [3,5) [4,6)
    mask = discordant_window_mask(cand, inc, horizon=2)

    # Then: only the windows covering index 3 are discordant
    assert mask == (False, False, True, True, False)

    with pytest.raises(ValueError, match="mismatched"):
        discordant_window_mask(["A"], ["A", "B"], horizon=1)


def test_count_effective_discordant_on_paired_windows() -> None:
    from datetime import date

    import polars as pl

    from src.tournament.champion_eval import _count_effective_discordant

    sessions = [date(2026, 1, d) for d in (5, 6, 7, 8, 9, 10)]
    cand_trades = pl.DataFrame(
        {
            "decision_date": [sessions[0], sessions[3], sessions[4]],
            "ticker": ["A", "B", "B"],
            "weight_after": [0.95, 0.95, 0.95],
        }
    )
    inc_trades = pl.DataFrame(
        {
            "decision_date": [sessions[0], sessions[3], sessions[4]],
            "ticker": ["A", "A", "A"],
            "weight_after": [0.95, 0.95, 0.95],
        }
    )

    class _Bt:
        def __init__(self, trades: pl.DataFrame) -> None:
            self.trades = trades

    n = _count_effective_discordant(
        paired_starts=[sessions[0], sessions[3]],
        sessions=sessions,
        horizon=2,
        candidate_backtest=_Bt(cand_trades),
        incumbent_backtest=_Bt(inc_trades),
    )
    assert n == 1


def test_champion_promotion_status_reports_insufficient_power() -> None:
    from datetime import date

    from src.tournament.champion_eval import champion_promotion_status

    sessions = [date(2026, 1, d) for d in (5, 6, 7, 8)]
    status, n_disc, min_disc, eligible = champion_promotion_status(
        paired_starts=[sessions[0]],
        sessions=sessions,
        horizon=2,
        candidate_backtest=None,
        incumbent_backtest=None,
        aggressive_status="PASS",
        conservative_status="PASS",
        loyo_status="PASS",
        artifact_integrity=True,
    )
    assert status == "INSUFFICIENT_POWER"
    assert n_disc == 0
    assert min_disc >= 1
    assert eligible is False


def test_champion_evaluation_sessions_falls_back_to_roll_starts() -> None:
    from datetime import date
    from types import SimpleNamespace

    from src.tournament.champion_eval import champion_evaluation_sessions

    sessions = [date(2026, 1, 5), date(2026, 1, 6)]
    runtime = SimpleNamespace(
        engine=SimpleNamespace(calendar=SimpleNamespace(sessions=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no cal"))))
    )
    out = champion_evaluation_sessions(
        runtime,
        __import__("polars").DataFrame(),
        SimpleNamespace(start=sessions[0], end=sessions[-1]),
        SimpleNamespace(starts=sessions),
    )
    assert out == sessions


def test_champion_evaluation_sessions_uses_session_grid() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.tournament.champion_eval import champion_evaluation_sessions

    sessions = [date(2026, 1, 5), date(2026, 1, 6)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["A", "A"], "close": [1.0, 1.0]})
    runtime = SimpleNamespace(engine=SimpleNamespace(calendar=SimpleNamespace(sessions=lambda _s, _e: sessions)))
    out = champion_evaluation_sessions(
        runtime,
        panel,
        SimpleNamespace(start=sessions[0], end=sessions[-1]),
        SimpleNamespace(starts=sessions),
    )
    assert out == sessions


def test_primary_holdings_from_backtest_skips_bad_rows() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.tournament.champion_eval import _primary_holdings_from_backtest

    sessions = [date(2026, 1, 5), date(2026, 1, 6)]
    trades = pl.DataFrame(
        {
            "decision_date": [sessions[0]],
            "ticker": ["A"],
            "weight_after": ["0.9"],
        },
        schema={
            "decision_date": pl.Date,
            "ticker": pl.Utf8,
            "weight_after": pl.Utf8,
        },
    )
    out = _primary_holdings_from_backtest(SimpleNamespace(trades=trades), sessions)
    assert out == ["A", "A"]


def test_primary_holdings_from_backtest_skips_unreadable_weights() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.tournament.champion_eval import _primary_holdings_from_backtest

    sessions = [date(2026, 1, 5), date(2026, 1, 6)]
    trades = pl.DataFrame(
        {
            "decision_date": [sessions[1]],
            "ticker": ["B"],
            "weight_after": ["bad"],
        },
        schema={
            "decision_date": pl.Date,
            "ticker": pl.Utf8,
            "weight_after": pl.Utf8,
        },
    )
    out = _primary_holdings_from_backtest(SimpleNamespace(trades=trades), sessions)
    assert out == [None, None]


def test_primary_holdings_from_backtest_ignores_non_date_keys() -> None:
    from datetime import date
    from types import SimpleNamespace

    from src.tournament.champion_eval import _primary_holdings_from_backtest

    sessions = [date(2026, 1, 5), date(2026, 1, 6)]

    class _DateCol:
        def unique(self):
            return self

        def to_list(self) -> list[object]:
            return [123, sessions[0]]

    class _FakeTrades:
        height = 1
        columns = ("decision_date", "ticker", "weight_after")

        def __getitem__(self, key: str) -> object:
            if key == "decision_date":
                return _DateCol()
            raise KeyError(key)

        def filter(self, *_a, **_k):
            return self

        def iter_rows(self, named: bool = True):
            yield {"decision_date": sessions[0], "ticker": "A", "weight_after": 0.9}

    out = _primary_holdings_from_backtest(SimpleNamespace(trades=_FakeTrades()), sessions)
    assert out[0] == "A"
    assert out[1] == "A"


def test_primary_holdings_from_backtest_clears_on_trade_failure() -> None:
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import patch

    import polars as pl

    from src.tournament.champion_eval import _primary_holdings_from_backtest

    sessions = [date(2026, 1, 5), date(2026, 1, 6)]
    trades = pl.DataFrame(
        {"decision_date": [sessions[0]], "ticker": ["A"], "weight_after": [0.9]}
    )
    with patch.object(trades, "filter", side_effect=RuntimeError("bad")):
        out = _primary_holdings_from_backtest(SimpleNamespace(trades=trades), sessions)
    assert out == [None, None]


def test_count_effective_discordant_returns_zero_without_paired_starts() -> None:
    from datetime import date

    from src.tournament.champion_eval import _count_effective_discordant

    assert (
        _count_effective_discordant(
            paired_starts=(),
            sessions=[date(2026, 1, 5)],
            horizon=1,
            candidate_backtest=None,
            incumbent_backtest=None,
        )
        == 0
    )


def test_discordant_window_mask_rejects_bad_inputs() -> None:
    import pytest

    from src.tournament.champion_eval import discordant_window_mask

    with pytest.raises(ValueError, match="holdings must be sequences"):
        discordant_window_mask(object(), ["A"], horizon=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horizon must be int"):
        discordant_window_mask(["A"], ["A"], horizon="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horizon must be positive"):
        discordant_window_mask(["A"], ["A"], horizon=0)


class _BadHolding:
    def __eq__(self, other: object) -> bool:
        raise RuntimeError("bad compare")


def test_discordant_window_mask_tolerates_bad_holding_compare() -> None:
    from src.tournament.champion_eval import discordant_window_mask

    mask = discordant_window_mask(["A", _BadHolding()], ["A", "A"], horizon=1)
    assert mask == (False, False)


def test_count_effective_discordant_returns_zero_on_mask_failure() -> None:
    from datetime import date
    from unittest.mock import patch

    from src.tournament.champion_eval import _count_effective_discordant

    sessions = [date(2026, 1, 5), date(2026, 1, 6)]
    with patch("src.tournament.champion_eval.discordant_window_mask", side_effect=ValueError("bad")):
        assert (
            _count_effective_discordant(
                paired_starts=[sessions[0]],
                sessions=sessions,
                horizon=1,
                candidate_backtest=None,
                incumbent_backtest=None,
            )
            == 0
        )


def test_champion_promotion_status_falls_back_when_config_unreadable() -> None:
    from datetime import date
    from unittest.mock import patch

    from src.tournament.champion_eval import champion_promotion_status

    sessions = [date(2026, 1, 5), date(2026, 1, 6)]
    with patch("src.tournament.attainability.load_attainability_config", side_effect=RuntimeError("bad")):
        status, n_disc, min_disc, eligible = champion_promotion_status(
            paired_starts=[sessions[0]],
            sessions=sessions,
            horizon=1,
            candidate_backtest=None,
            incumbent_backtest=None,
            aggressive_status="PASS",
            conservative_status="PASS",
            loyo_status="PASS",
            artifact_integrity=True,
        )
    assert status == "INSUFFICIENT_POWER"
    assert min_disc == 5
    assert eligible is False


def test_discordant_window_mask_returns_empty_when_window_invalid() -> None:
    from src.tournament.champion_eval import discordant_window_mask

    assert discordant_window_mask(["A"], ["A"], horizon=2) == ()


def test_resolve_promotion_status_tolerates_bad_power_counts() -> None:
    from src.tournament.champion_eval import resolve_promotion_status

    assert (
        resolve_promotion_status(
            aggressive_status="PASS",
            conservative_status="PASS",
            loyo_status="PASS",
            artifact_integrity=True,
            n_effective_discordant=object(),  # type: ignore[arg-type]
            min_effective_discordant=object(),  # type: ignore[arg-type]
        )
        == "PROMOTE"
    )
