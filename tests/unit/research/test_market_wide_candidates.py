from datetime import date, timedelta
import polars as pl


def _panel(sessions: list[date]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for day in sessions:
        rows.append({"date": day, "ticker": "111", "name": "KODEX 반도체", "open": 100.0, "trading_value": 1_000.0, "is_tradable": True})
        rows.append({"date": day, "ticker": "222", "name": "UNKNOWNBRAND 반도체", "open": 100.0, "trading_value": 1_000.0, "is_tradable": True})
        rows.append({"date": day, "ticker": "333", "name": "KODEX 합성 반도체", "open": 100.0, "trading_value": 1_000.0, "is_tradable": True})
        rows.append({"date": day, "ticker": "444", "name": "KODEX 저유동", "open": 100.0, "trading_value": 1.0, "is_tradable": True})
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "ticker": pl.String, "name": pl.String, "open": pl.Float64, "trading_value": pl.Float64, "is_tradable": pl.Boolean},
    )


def test_market_wide_session_candidates_selects_only_eligible_sponsor_tickers() -> None:
    from src.research.executable_oracle import market_wide_session_candidates

    # Given: 8 sessions; 111 is sponsor+liquid, 222 non-sponsor, 333 synthetic, 444 illiquid
    sessions = [date(2026, 1, 5) + timedelta(days=i) for i in range(8)]

    # When
    out = market_wide_session_candidates(
        _panel(sessions),
        sessions=sessions,
        sponsor_issuers=frozenset({"삼성자산운용"}),
        brand_map={"KODEX": "삼성자산운용"},
        min_adv=100.0,
        min_history_sessions=2,
    )

    # Then: every session is keyed (R2)
    assert set(out) == set(sessions)
    # Then: ADV rolling needs 5 samples -> nothing eligible before index 4
    assert out[sessions[0]] == ()
    assert out[sessions[3]] == ()
    # Then: only the sponsor, non-synthetic, liquid ticker survives
    assert out[sessions[4]] == ("111",)
    assert out[sessions[7]] == ("111",)


def test_market_wide_session_candidates_sorts_tickers_and_handles_empty_sessions() -> None:
    from src.research.executable_oracle import market_wide_session_candidates

    # Given: two eligible sponsor tickers deliberately inserted in descending ticker order
    sessions = [date(2026, 1, 5) + timedelta(days=i) for i in range(8)]
    rows: list[dict[str, object]] = [
        {"date": day, "ticker": ticker, "name": "KODEX 반도체", "open": 100.0, "trading_value": 1_000.0, "is_tradable": True}
        for day in sessions
        for ticker in ("999", "111")
    ]
    panel = pl.DataFrame(
        rows,
        schema={"date": pl.Date, "ticker": pl.String, "name": pl.String, "open": pl.Float64, "trading_value": pl.Float64, "is_tradable": pl.Boolean},
    )

    # When
    out = market_wide_session_candidates(
        panel,
        sessions=sessions,
        sponsor_issuers=frozenset({"삼성자산운용"}),
        brand_map={"KODEX": "삼성자산운용"},
        min_adv=100.0,
        min_history_sessions=2,
    )

    # Then: deterministic ascending ticker order
    assert out[sessions[5]] == ("111", "999")

    # Then: no sessions -> empty mapping, no crash
    assert market_wide_session_candidates(
        panel,
        sessions=[],
        sponsor_issuers=frozenset({"삼성자산운용"}),
        brand_map={"KODEX": "삼성자산운용"},
    ) == {}
