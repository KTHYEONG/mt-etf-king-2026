from __future__ import annotations


def test_is_beta_family_exact_not_sector_substring() -> None:
    from src.strategies.convex_impulse import DEFAULT_BETA_FAMILY_KEYS, is_beta_family

    assert is_beta_family("코스피 200", DEFAULT_BETA_FAMILY_KEYS) is True
    assert is_beta_family("코스피 200 선물지수", DEFAULT_BETA_FAMILY_KEYS) is True
    assert is_beta_family("코스닥 150", DEFAULT_BETA_FAMILY_KEYS) is True
    assert is_beta_family("F-코스닥150 지수", DEFAULT_BETA_FAMILY_KEYS) is True
    assert is_beta_family("KRX 300", DEFAULT_BETA_FAMILY_KEYS) is True
    assert is_beta_family("코스피 200 정보기술", DEFAULT_BETA_FAMILY_KEYS) is False
    assert is_beta_family("KRX 반도체", DEFAULT_BETA_FAMILY_KEYS) is False
    assert is_beta_family(" 코스피 200 ", DEFAULT_BETA_FAMILY_KEYS) is True
    assert is_beta_family("", DEFAULT_BETA_FAMILY_KEYS) is False


def test_filter_convex_plus2_drops_beta_macro_synth() -> None:
    import polars as pl

    from src.strategies.convex_impulse import ConvexImpulseConfig, filter_convex_plus2_rows
    from src.strategies.sticky.model import DEFAULT_EXCLUDE_NAME_TOKENS

    snap = pl.DataFrame(
        {
            "ticker": ["122630", "233740", "494310", "BOND", "SYN", "INV", "243880"],
            "name": [
                "KODEX 레버리지",
                "KODEX 코스닥150레버리지",
                "KODEX 반도체레버리지",
                "RISE 국채30년레버리지(합성)",
                "TIGER 차이나전기차레버리지(합성)",
                "KODEX 인버스",
                "TIGER 200IT레버리지",
            ],
            "underlying_index_name": [
                "코스피 200",
                "코스닥 150",
                "KRX 반도체",
                "KAP 국채30년 TR 지수(총수익)",
                "Solactive China Electric Vehicle and Battery Index(PR)",
                "코스피 200",
                "코스피 200 정보기술",
            ],
            "mom_5": [0.1] * 7,
            "mom_10": [0.1] * 7,
            "mom_20": [0.05] * 7,
            "mom_60": [0.3] * 7,
            "volume_expansion": [1.5] * 7,
            "trading_value": [1e11] * 7,
            "drawdown_20": [0.0] * 7,
        }
    )
    cfg = ConvexImpulseConfig(
        exclude_name_tokens=DEFAULT_EXCLUDE_NAME_TOKENS,
        exclude_synthetic=True,
    )
    rows = filter_convex_plus2_rows(snap, cfg)
    tickers = {str(r["ticker"]) for r in rows}
    assert "494310" in tickers
    assert "243880" in tickers
    assert "122630" not in tickers
    assert "233740" not in tickers
    assert "BOND" not in tickers
    assert "SYN" not in tickers
    assert "INV" not in tickers


def test_classify_setup_impulse_vs_continuation() -> None:
    from src.strategies.convex_impulse import ConvexImpulseConfig, classify_setup

    cfg = ConvexImpulseConfig()
    impulse_row = {
        "mom_5": 0.12,
        "mom_10": 0.10,
        "mom_20": 0.04,
        "mom_60": 0.05,
        "volume_expansion": 1.5,
    }
    cont_row = {
        "mom_5": 0.02,
        "mom_10": 0.03,
        "mom_20": 0.10,
        "mom_60": 0.25,
        "volume_expansion": 0.8,
    }
    none_row = {
        "mom_5": 0.01,
        "mom_10": 0.02,
        "mom_20": 0.03,
        "mom_60": 0.04,
        "volume_expansion": 0.9,
    }
    assert classify_setup(impulse_row, cfg) == "impulse"
    assert classify_setup(cont_row, cfg) == "continuation"
    assert classify_setup(none_row, cfg) == "none"


def test_pick_prefers_impulse_over_continuation() -> None:
    from src.strategies.convex_impulse import ConvexImpulseConfig, pick_convex_ticker

    cfg = ConvexImpulseConfig()
    rows = [
        {
            "ticker": "CONT",
            "mom_5": 0.02,
            "mom_10": 0.03,
            "mom_20": 0.10,
            "mom_60": 0.40,
            "volume_expansion": 0.8,
        },
        {
            "ticker": "IMP",
            "mom_5": 0.12,
            "mom_10": 0.11,
            "mom_20": 0.04,
            "mom_60": 0.05,
            "volume_expansion": 1.6,
        },
    ]
    picked = pick_convex_ticker(rows, cfg, held=None, hold_len=0)
    assert picked == "IMP"


def test_score_returns_cash_when_no_setup() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.portfolio.intent import CASH_INTENT
    from src.strategies.convex_impulse import ConvexImpulseConfig, ConvexImpulseModel
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["122630", "494310"],
            "name": ["KODEX 레버리지", "KODEX 반도체레버리지"],
            "underlying_index_name": ["코스피 200", "KRX 반도체"],
            "mom_5": [0.20, 0.01],
            "mom_10": [0.30, 0.02],
            "mom_20": [0.25, 0.03],
            "mom_60": [0.50, 0.04],
            "volume_expansion": [2.0, 0.8],
            "trading_value": [1e12, 1e11],
            "drawdown_20": [0.0, 0.0],
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
    ctx = DecisionContext(decision_date=date(2024, 6, 3), regime=None, capital=1.0e9, held={}, rules=rules)
    model = ConvexImpulseModel(name="P31", config=ConvexImpulseConfig(exclude_synthetic=True, min_fill_ratio=0.0))
    out = model.score(snap, ctx)
    assert out is CASH_INTENT or getattr(out, "kind", None) == "cash"


def test_score_never_selects_beta_leverage() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.convex_impulse import ConvexImpulseConfig, ConvexImpulseModel
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["122630", "494310"],
            "name": ["KODEX 레버리지", "KODEX 반도체레버리지"],
            "underlying_index_name": ["코스피 200", "KRX 반도체"],
            "mom_5": [0.05, 0.04],
            "mom_10": [0.08, 0.06],
            "mom_20": [0.12, 0.11],
            "mom_60": [0.80, 0.30],
            "volume_expansion": [1.0, 1.0],
            "trading_value": [1e12, 1e11],
            "drawdown_20": [0.0, 0.0],
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
    ctx = DecisionContext(decision_date=date(2025, 9, 22), regime=None, capital=1.0e9, held={}, rules=rules)
    model = ConvexImpulseModel(name="P31", config=ConvexImpulseConfig(exclude_synthetic=True, min_fill_ratio=0.0))
    out = model.score(snap, ctx)
    assert isinstance(out, dict)
    assert "122630" not in out
    assert "494310" in out
    top = sorted(out.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]
    assert top == "494310"


def test_crash_cash_exits_to_cash_intent() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.portfolio.intent import CASH_INTENT
    from src.strategies.convex_impulse import ConvexImpulseConfig, ConvexImpulseModel
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            "ticker": ["494310"],
            "name": ["KODEX 반도체레버리지"],
            "underlying_index_name": ["KRX 반도체"],
            "mom_5": [0.04],
            "mom_10": [0.06],
            "mom_20": [0.11],
            "mom_60": [0.30],
            "volume_expansion": [1.0],
            "trading_value": [1e11],
            "drawdown_20": [-0.20],
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
    ctx = DecisionContext(
        decision_date=date(2025, 9, 22),
        regime=None,
        capital=1.0e9,
        held={"494310": 0.95},
        rules=rules,
    )
    model = ConvexImpulseModel(name="P31", config=ConvexImpulseConfig(crash_drawdown=-0.12, min_fill_ratio=0.0))
    model.restore_state("494310", 5)
    out = model.score(snap, ctx)
    assert out is CASH_INTENT or getattr(out, "kind", None) == "cash"


def test_p31_factory_registry_exposure() -> None:
    from src.alpha.baselines import BASELINES
    from src.cli.constants import CHAMPION_STRATEGY
    from src.portfolio.constraints import load_p27_exposure_limits, resolve_exposure_limits_for_model
    from src.strategies.convex_impulse import ConvexImpulseModel, DEFAULT_BETA_FAMILY_KEYS
    from src.strategies.ids import CONVEX_LOTTERY_IMPULSE, STICKY_MOM60_RAW
    from src.strategies.registry import resolve_strategy_id

    assert resolve_strategy_id("P31") == CONVEX_LOTTERY_IMPULSE
    assert CONVEX_LOTTERY_IMPULSE == "convex.lottery_impulse"
    assert "P31" in BASELINES
    model = BASELINES["P31"]()
    assert isinstance(model, ConvexImpulseModel)
    assert getattr(model, "name") == "P31"
    assert bool(getattr(model, "path_dependent")) is True
    assert tuple(model.config.beta_family_keys) == DEFAULT_BETA_FAMILY_KEYS
    assert abs(float(model.config.impulse_min) - 0.08) < 1e-12
    assert abs(float(model.config.volx_min) - 1.2) < 1e-12
    assert abs(float(model.config.continuation_min) - 0.20) < 1e-12
    assert abs(float(model.config.crash_drawdown) + 0.12) < 1e-12
    assert resolve_exposure_limits_for_model("P31", comparison_mode="full_strategy_own") == load_p27_exposure_limits()
    assert CHAMPION_STRATEGY == STICKY_MOM60_RAW
    assert BASELINES["P27"]().name == "P27"
    from src.cli._impl import STICKY_ADOPTION_MODELS

    assert "P31" in STICKY_ADOPTION_MODELS
