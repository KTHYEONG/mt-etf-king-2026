from __future__ import annotations


def test_name_excluded_macro_tokens() -> None:
    from src.strategies.sticky.model import DEFAULT_EXCLUDE_NAME_TOKENS, name_excluded

    toks = DEFAULT_EXCLUDE_NAME_TOKENS
    assert name_excluded("RISE 국채30년레버리지(합성)", toks) is True
    assert name_excluded("ACE 미국30년국채선물레버리지(합성 H)", toks) is True
    assert name_excluded("TIGER 미국달러선물레버리지", toks) is True
    assert name_excluded("KINDEX 골드선물 레버리지(합성 H)", toks) is True
    assert name_excluded("마이티 200커버드콜ATM레버리지", toks) is True
    assert name_excluded("TIGER 200IT레버리지", toks) is False
    assert name_excluded("TIGER 차이나항셍테크레버리지(합성 H)", toks) is False
    assert name_excluded("KODEX 코스닥150레버리지", toks) is False


def test_filter_plus2_scores_excludes_macro_names() -> None:
    import polars as pl

    from src.strategies.sticky.model import DEFAULT_EXCLUDE_NAME_TOKENS, StickyLeaderConfig, filter_plus2_scores

    snap = pl.DataFrame(
        {
            "ticker": ["BOND", "EQ"],
            "name": ["RISE 국채30년레버리지(합성)", "TIGER 200IT레버리지"],
            "mom_60": [0.90, 0.40],
            "volume_expansion": [1.0, 1.0],
        }
    )
    cfg = StickyLeaderConfig(
        mom_col="mom_60",
        only_plus_2=True,
        no_inverse=True,
        exclude_name_tokens=DEFAULT_EXCLUDE_NAME_TOKENS,
    )
    out = filter_plus2_scores(snap, cfg)
    assert "BOND" not in out
    assert out.get("EQ") == 0.40


def test_blend_rank_scores_weights() -> None:
    from src.strategies.sticky.model import blend_rank_scores, cross_section_percentile_ranks

    primary = {"A": 0.1, "B": 0.5, "C": 0.9}
    aux = {"A": 0.9, "B": 0.5, "C": 0.1}
    ranks_p = cross_section_percentile_ranks(primary)
    ranks_a = cross_section_percentile_ranks(aux)
    assert ranks_p["C"] == 1.0 and ranks_p["A"] == 0.0
    out = blend_rank_scores(primary, aux, w_primary=0.7, w_aux=0.3)
    assert set(out) == {"A", "B", "C"}
    # C high mom low vol vs A low mom high vol: with 0.7/0.3 C still wins
    assert sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] == "C"
    assert abs(out["C"] - (0.7 * ranks_p["C"] + 0.3 * ranks_a["C"])) < 1e-12


def test_p29_factory_excludes_macro_keeps_p27_clean() -> None:
    from src.alpha.baselines import BASELINES
    from src.portfolio.constraints import load_p27_exposure_limits, resolve_exposure_limits_for_model
    from src.strategies.sticky.model import DEFAULT_EXCLUDE_NAME_TOKENS

    assert "P29" in BASELINES and "P29V" in BASELINES
    p27 = BASELINES["P27"]()
    p29 = BASELINES["P29"]()
    p29v = BASELINES["P29V"]()
    assert p29.name == "P29" and p29v.name == "P29V"
    assert tuple(getattr(p27.config, "exclude_name_tokens", ())) == ()
    assert tuple(p29.config.exclude_name_tokens) == DEFAULT_EXCLUDE_NAME_TOKENS
    assert str(p29.config.mom_col) == "mom_60"
    assert float(p29.config.min_gap) == 0.04
    assert p29v.config.score_aux_col == "volume_expansion"
    assert abs(float(p29v.config.score_aux_weight) - 0.3) < 1e-12
    assert float(p29v.config.min_gap) == 0.10
    assert resolve_exposure_limits_for_model("P29", comparison_mode="full_strategy_own") == load_p27_exposure_limits()
    assert resolve_exposure_limits_for_model("P29V", comparison_mode="full_strategy_own") == load_p27_exposure_limits()


def test_p29_score_prefers_equity_over_higher_mom_bond() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["BOND", "EQ"],
            "name": ["ACE 미국30년국채선물레버리지(합성 H)", "TIGER 200IT레버리지"],
            "mom_60": [0.95, 0.55],
            "mom_5": [0.0, 0.0],
            "volume_expansion": [0.5, 2.0],
            "trading_value": [1e11, 1e11],
        }
    )
    rules = TournamentRules(
        name="t",
        start_date=date(2024, 9, 23),
        end_date=date(2024, 11, 15),
        initial_capital=1_000_000_000,
        category="autonomous",
        leverage_allowed=True,
        inverse_allowed=True,
        max_weight=1.0,
        cash_allowed=True,
        sponsor_etf_only=True,
        manifest_path=None,
        issuer_whitelist=None,
        commission_bps=3.0,
        slippage_bps=5.0,
        max_order_to_adv=0.01,
        stress_grid=(0.01, 0.02, 0.05),
    )
    ctx = DecisionContext(decision_date=date(2024, 6, 3), regime=None, capital=1.0e9, held={}, rules=rules)
    p27 = BASELINES["P27"]()
    p29 = BASELINES["P29"]()
    s27 = p27.score(snap, ctx)
    s29 = p29.score(snap, ctx)
    assert isinstance(s27, dict) and isinstance(s29, dict)
    top27 = sorted(s27.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]
    top29 = sorted(s29.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]
    assert top27 == "EQ"
    assert top29 == "EQ"
    assert "BOND" not in s27
    assert "BOND" not in s29


def test_p29v_blend_changes_leader_vs_p29() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "name": ["TIGER 200IT레버리지", "KODEX 반도체레버리지", "TIGER 200에너지화학레버리지"],
            "mom_60": [0.50, 0.49, 0.10],
            "mom_5": [0.0, 0.0, 0.0],
            "volume_expansion": [0.5, 3.0, 1.0],
            "trading_value": [1e11, 1e11, 1e11],
        }
    )
    rules = TournamentRules(
        name="t",
        start_date=date(2025, 9, 22),
        end_date=date(2025, 11, 14),
        initial_capital=1_000_000_000,
        category="autonomous",
        leverage_allowed=True,
        inverse_allowed=True,
        max_weight=1.0,
        cash_allowed=True,
        sponsor_etf_only=True,
        manifest_path=None,
        issuer_whitelist=None,
        commission_bps=3.0,
        slippage_bps=5.0,
        max_order_to_adv=0.01,
        stress_grid=(0.01, 0.02, 0.05),
    )
    ctx = DecisionContext(decision_date=date(2025, 3, 3), regime=None, capital=1.0e9, held={}, rules=rules)
    p29 = BASELINES["P29"]()
    p29v = BASELINES["P29V"]()
    s29 = p29.score(snap, ctx)
    s29v = p29v.score(snap, ctx)
    assert isinstance(s29, dict) and isinstance(s29v, dict)
    top29 = sorted(s29.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]
    top29v = sorted(s29v.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]
    assert top29 == "A"
    assert top29v == "B"
