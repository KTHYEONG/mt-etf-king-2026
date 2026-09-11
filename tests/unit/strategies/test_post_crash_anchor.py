# ruff: noqa
import polars as pl


def test_post_crash_anchor_scores_picks_highest_score_anchor_only() -> None:
    from src.strategies.sticky.model_scores import post_crash_anchor_scores

    # Given: 앵커 2종 + 더 강한 비앵커 +2x
    snapshot = pl.DataFrame(
        {
            "ticker": ["233740", "122630", "SEMI", "THIN"],
            "name": [
                "KODEX 코스닥150레버리지",
                "KODEX 레버리지",
                "TIGER 반도체TOP10레버리지",
                "SOL SK하이닉스단일종목레버리지",
            ],
            "mom_60": [-0.10, -0.05, 0.30, 0.20],
            "mom_20": [0.20, 0.15, 0.45, 0.60],
            "drawdown_20": [-0.02, -0.01, -0.05, -0.08],
            "trading_value": [675_000_000_000.0, 1_800_000_000_000.0, 190_000_000_000.0, 2_000_000_000.0],
        }
    )

    # When
    out = post_crash_anchor_scores(snapshot, anchor_tickers=("233740", "122630"), held=None)

    # Then: mom_20 최고 앵커 1종목만
    assert out == {"233740": 0.20}



import polars as pl


def test_post_crash_anchor_scores_keeps_held_anchor_latch() -> None:
    from src.strategies.sticky.model_scores import post_crash_anchor_scores

    snapshot = pl.DataFrame(
        {
            "ticker": ["233740", "122630", "SEMI", "THIN"],
            "name": [
                "KODEX 코스닥150레버리지",
                "KODEX 레버리지",
                "TIGER 반도체TOP10레버리지",
                "SOL SK하이닉스단일종목레버리지",
            ],
            "mom_60": [-0.10, -0.05, 0.30, 0.20],
            "mom_20": [0.20, 0.15, 0.45, 0.60],
            "drawdown_20": [-0.02, -0.01, -0.05, -0.08],
            "trading_value": [675_000_000_000.0, 1_800_000_000_000.0, 190_000_000_000.0, 2_000_000_000.0],
        }
    )

    out = post_crash_anchor_scores(snapshot, anchor_tickers=("233740", "122630"), held="122630")

    assert out == {"122630": 0.15}



import polars as pl


def test_post_crash_anchor_scores_stop_is_inclusive_and_rotates() -> None:
    from src.strategies.sticky.model_scores import post_crash_anchor_scores

    snapshot = pl.DataFrame(
        {
            "ticker": ["233740", "122630", "SEMI", "THIN"],
            "name": [
                "KODEX 코스닥150레버리지",
                "KODEX 레버리지",
                "TIGER 반도체TOP10레버리지",
                "SOL SK하이닉스단일종목레버리지",
            ],
            "mom_60": [-0.10, -0.05, 0.30, 0.20],
            "mom_20": [0.20, 0.15, 0.45, 0.60],
            "drawdown_20": [-0.02, -0.01, -0.05, -0.08],
            "trading_value": [675_000_000_000.0, 1_800_000_000_000.0, 190_000_000_000.0, 2_000_000_000.0],
        }
    )

    # Given: 보유 앵커 233740 이 정확히 -15% (경계 포함) 손절 도달
    stopped = snapshot.with_columns(
        pl.when(pl.col("ticker") == "233740").then(-0.15).otherwise(pl.col("drawdown_20")).alias("drawdown_20")
    )

    out = post_crash_anchor_scores(
        stopped, anchor_tickers=("233740", "122630"), held="233740", stop_drawdown=0.15
    )

    # Then: 다른 앵커로 교체
    assert out == {"122630": 0.15}

    # And: 경계 바로 위(-0.1499)는 손절 아님 -> 보유 유지
    near = snapshot.with_columns(
        pl.when(pl.col("ticker") == "233740").then(-0.1499).otherwise(pl.col("drawdown_20")).alias("drawdown_20")
    )
    assert post_crash_anchor_scores(near, anchor_tickers=("233740", "122630"), held="233740") == {"233740": 0.20}



import polars as pl


def test_post_crash_anchor_scores_all_stopped_returns_empty() -> None:
    from src.strategies.sticky.model_scores import post_crash_anchor_scores

    snapshot = pl.DataFrame(
        {
            "ticker": ["233740", "122630", "SEMI", "THIN"],
            "name": [
                "KODEX 코스닥150레버리지",
                "KODEX 레버리지",
                "TIGER 반도체TOP10레버리지",
                "SOL SK하이닉스단일종목레버리지",
            ],
            "mom_60": [-0.10, -0.05, 0.30, 0.20],
            "mom_20": [0.20, 0.15, 0.45, 0.60],
            "drawdown_20": [-0.02, -0.01, -0.05, -0.08],
            "trading_value": [675_000_000_000.0, 1_800_000_000_000.0, 190_000_000_000.0, 2_000_000_000.0],
        }
    )

    all_stopped = snapshot.with_columns(pl.lit(-0.20).alias("drawdown_20"))

    assert post_crash_anchor_scores(all_stopped, anchor_tickers=("233740", "122630"), held="233740") == {}



import math

import polars as pl


def test_post_crash_anchor_scores_fail_closed_inputs() -> None:
    from src.strategies.sticky.model_scores import post_crash_anchor_scores

    snapshot = pl.DataFrame(
        {
            "ticker": ["233740", "122630", "SEMI", "THIN"],
            "name": [
                "KODEX 코스닥150레버리지",
                "KODEX 레버리지",
                "TIGER 반도체TOP10레버리지",
                "SOL SK하이닉스단일종목레버리지",
            ],
            "mom_60": [-0.10, -0.05, 0.30, 0.20],
            "mom_20": [0.20, 0.15, 0.45, 0.60],
            "drawdown_20": [-0.02, -0.01, -0.05, -0.08],
            "trading_value": [675_000_000_000.0, 1_800_000_000_000.0, 190_000_000_000.0, 2_000_000_000.0],
        }
    )

    anchors = ("233740", "122630")

    # 구조적 결손 -> {}
    assert post_crash_anchor_scores(None, anchor_tickers=anchors, held=None) == {}  # type: ignore[arg-type]
    assert post_crash_anchor_scores(snapshot.head(0), anchor_tickers=anchors, held=None) == {}
    assert post_crash_anchor_scores(snapshot, anchor_tickers=(), held=None) == {}
    assert post_crash_anchor_scores(snapshot.drop("drawdown_20"), anchor_tickers=anchors, held=None) == {}
    assert post_crash_anchor_scores(snapshot.drop("mom_20"), anchor_tickers=anchors, held=None) == {}
    assert post_crash_anchor_scores(snapshot.drop("ticker"), anchor_tickers=anchors, held=None) == {}

    # 개별 앵커 결손 -> 해당 앵커만 제외
    dd_null = snapshot.with_columns(
        pl.when(pl.col("ticker") == "233740").then(None).otherwise(pl.col("drawdown_20")).alias("drawdown_20")
    )
    assert post_crash_anchor_scores(dd_null, anchor_tickers=anchors, held="233740") == {"122630": 0.15}
    score_nan = snapshot.with_columns(
        pl.when(pl.col("ticker") == "233740").then(math.nan).otherwise(pl.col("mom_20")).alias("mom_20")
    )
    assert post_crash_anchor_scores(score_nan, anchor_tickers=anchors, held=None) == {"122630": 0.15}
    dd_inf = snapshot.with_columns(
        pl.when(pl.col("ticker") == "122630").then(-math.inf).otherwise(pl.col("drawdown_20")).alias("drawdown_20")
    )
    assert post_crash_anchor_scores(dd_inf, anchor_tickers=("122630",), held=None) == {}
    assert post_crash_anchor_scores(snapshot, anchor_tickers=("MISSING", "122630"), held="MISSING") == {"122630": 0.15}



import polars as pl


def test_post_crash_anchor_scores_tie_breaks_to_smallest_ticker() -> None:
    from src.strategies.sticky.model_scores import post_crash_anchor_scores

    snapshot = pl.DataFrame(
        {
            "ticker": ["233740", "122630", "SEMI", "THIN"],
            "name": [
                "KODEX 코스닥150레버리지",
                "KODEX 레버리지",
                "TIGER 반도체TOP10레버리지",
                "SOL SK하이닉스단일종목레버리지",
            ],
            "mom_60": [-0.10, -0.05, 0.30, 0.20],
            "mom_20": [0.20, 0.15, 0.45, 0.60],
            "drawdown_20": [-0.02, -0.01, -0.05, -0.08],
            "trading_value": [675_000_000_000.0, 1_800_000_000_000.0, 190_000_000_000.0, 2_000_000_000.0],
        }
    )

    tied = snapshot.with_columns(pl.lit(0.10).alias("mom_20"))

    out = post_crash_anchor_scores(tied, anchor_tickers=("233740", "122630"), held=None)

    assert out == {"122630": 0.10}



import math

import polars as pl
import pytest


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, math.nan, math.inf, True])
def test_post_crash_anchor_scores_rejects_invalid_stop(bad: float) -> None:
    from src.strategies.sticky.model_scores import post_crash_anchor_scores

    snapshot = pl.DataFrame(
        {
            "ticker": ["233740", "122630", "SEMI", "THIN"],
            "name": [
                "KODEX 코스닥150레버리지",
                "KODEX 레버리지",
                "TIGER 반도체TOP10레버리지",
                "SOL SK하이닉스단일종목레버리지",
            ],
            "mom_60": [-0.10, -0.05, 0.30, 0.20],
            "mom_20": [0.20, 0.15, 0.45, 0.60],
            "drawdown_20": [-0.02, -0.01, -0.05, -0.08],
            "trading_value": [675_000_000_000.0, 1_800_000_000_000.0, 190_000_000_000.0, 2_000_000_000.0],
        }
    )

    with pytest.raises(ValueError):
        post_crash_anchor_scores(snapshot, anchor_tickers=("233740", "122630"), held=None, stop_drawdown=bad)



def test_sticky_config_from_yaml_parses_post_crash_anchor_fields() -> None:
    from src.strategies.sticky.model_config import StickyLeaderConfig

    cfg = StickyLeaderConfig.from_yaml(
        {
            "mom_col": "mom_60",
            "post_crash_anchor": True,
            "anchor_tickers": ["233740", "122630"],
            "anchor_score_col": "mom_20",
            "anchor_stop_drawdown": 0.15,
        }
    )
    assert cfg.post_crash_anchor is True
    assert cfg.anchor_tickers == ("233740", "122630")
    assert cfg.anchor_score_col == "mom_20"
    assert cfg.anchor_stop_drawdown == 0.15

    default = StickyLeaderConfig.from_yaml({"mom_col": "mom_60"})
    assert default.post_crash_anchor is False
    assert default.anchor_tickers == ()
    assert default.anchor_score_col == "mom_20"
    assert default.anchor_stop_drawdown == 0.15



import math

import pytest


@pytest.mark.parametrize(
    "raw",
    [
        {"post_crash_anchor": True},
        {"post_crash_anchor": True, "anchor_tickers": []},
        {"post_crash_anchor": "yes", "anchor_tickers": ["233740"]},
        {"post_crash_anchor": True, "anchor_tickers": "233740"},
        {"post_crash_anchor": True, "anchor_tickers": ["233740", ""]},
        {"post_crash_anchor": True, "anchor_tickers": ["233740", 122630]},
        {"post_crash_anchor": True, "anchor_tickers": ["233740"], "anchor_stop_drawdown": 0.0},
        {"post_crash_anchor": True, "anchor_tickers": ["233740"], "anchor_stop_drawdown": 1.0},
        {"post_crash_anchor": True, "anchor_tickers": ["233740"], "anchor_stop_drawdown": True},
        {"post_crash_anchor": True, "anchor_tickers": ["233740"], "anchor_stop_drawdown": math.nan},
        {"post_crash_anchor": True, "anchor_tickers": ["233740"], "anchor_score_col": ""},
    ],
)
def test_sticky_config_from_yaml_rejects_invalid_post_crash_anchor(raw: dict[str, object]) -> None:
    from src.strategies.sticky.model_config import StickyLeaderConfig

    with pytest.raises(ValueError):
        StickyLeaderConfig.from_yaml(raw)



from datetime import date
from types import SimpleNamespace

import polars as pl


def test_score_post_crash_anchor_replaces_rebound_route_in_crash_rebound() -> None:
    from src.alpha.base import DecisionContext
    from src.strategies.sticky.model_config import StickyLeaderConfig
    from src.strategies.sticky.model_runner import StickyLeaderModel
    from src.tournament.objective.cutoff_auc import CUTOFF_AUC_IS_PRODUCTION_GATE

    assert CUTOFF_AUC_IS_PRODUCTION_GATE is True

    snapshot = pl.DataFrame(
        {
            "ticker": ["233740", "122630", "SEMI", "THIN"],
            "name": [
                "KODEX 코스닥150레버리지",
                "KODEX 레버리지",
                "TIGER 반도체TOP10레버리지",
                "SOL SK하이닉스단일종목레버리지",
            ],
            "mom_60": [-0.10, -0.05, 0.30, 0.20],
            "mom_20": [0.20, 0.15, 0.45, 0.60],
            "drawdown_20": [-0.02, -0.01, -0.05, -0.08],
            "trading_value": [675_000_000_000.0, 1_800_000_000_000.0, 190_000_000_000.0, 2_000_000_000.0],
        }
    )
    def ctx(sleeve: str, held: dict[str, float], day: int) -> DecisionContext:
        return DecisionContext(
            decision_date=date(2026, 9, day),
            regime=None,
            capital=1_000_000_000.0,
            held=dict(held),
            rules=SimpleNamespace(initial_capital=1_000_000_000.0, max_order_to_adv=0.01),  # type: ignore[arg-type]
            championship_sleeve=sleeve,
        )

    # Given: 플래그 OFF(현행 P27) -> 기존 rebound 라우트
    base = StickyLeaderModel(name="sticky.mom60_raw", config=StickyLeaderConfig(mom_col="mom_60", min_fill_ratio=0.25))
    baseline = base.score(snapshot, ctx("CRASH_REBOUND", {}, 18))
    assert isinstance(baseline, dict)
    assert max(baseline, key=baseline.get) == "SEMI"
    assert "THIN" not in baseline

    # When: 플래그 ON
    anchor = StickyLeaderModel(
        name="sticky.mom60_post_crash_anchor",
        config=StickyLeaderConfig(
            mom_col="mom_60",
            min_fill_ratio=0.25,
            post_crash_anchor=True,
            anchor_tickers=("233740", "122630"),
        ),
    )
    result = anchor.score(snapshot, ctx("CRASH_REBOUND", {}, 18))

    # Then: 앵커 1종목
    assert result == {"233740": 0.20}



from datetime import date
from types import SimpleNamespace

import polars as pl


def test_score_post_crash_anchor_latch_release_and_fail_closed() -> None:
    from src.alpha.base import DecisionContext
    from src.portfolio.intent import CASH_INTENT
    from src.strategies.sticky.model_config import StickyLeaderConfig
    from src.strategies.sticky.model_runner import StickyLeaderModel

    snapshot = pl.DataFrame(
        {
            "ticker": ["233740", "122630", "SEMI", "THIN"],
            "name": [
                "KODEX 코스닥150레버리지",
                "KODEX 레버리지",
                "TIGER 반도체TOP10레버리지",
                "SOL SK하이닉스단일종목레버리지",
            ],
            "mom_60": [-0.10, -0.05, 0.30, 0.20],
            "mom_20": [0.20, 0.15, 0.45, 0.60],
            "drawdown_20": [-0.02, -0.01, -0.05, -0.08],
            "trading_value": [675_000_000_000.0, 1_800_000_000_000.0, 190_000_000_000.0, 2_000_000_000.0],
        }
    )
    def ctx(sleeve: str, held: dict[str, float], day: int) -> DecisionContext:
        return DecisionContext(
            decision_date=date(2026, 9, day),
            regime=None,
            capital=1_000_000_000.0,
            held=dict(held),
            rules=SimpleNamespace(initial_capital=1_000_000_000.0, max_order_to_adv=0.01),  # type: ignore[arg-type]
            championship_sleeve=sleeve,
        )

    model = StickyLeaderModel(
        name="sticky.mom60_post_crash_anchor",
        config=StickyLeaderConfig(
            mom_col="mom_60",
            min_fill_ratio=0.25,
            post_crash_anchor=True,
            anchor_tickers=("233740", "122630"),
        ),
    )

    # INACTIVE + 보유 앵커 -> 래치 유지 (점수 높은 233740 으로 갈아타지 않음)
    assert model.score(snapshot, ctx("INACTIVE", {"122630": 0.95}, 21)) == {"122630": 0.15}

    # INACTIVE + 미보유 -> 현금 (신규 진입은 CRASH_REBOUND 에서만)
    model.reset_trackers()
    assert model.score(snapshot, ctx("INACTIVE", {}, 22)) is CASH_INTENT

    # UNCERTAIN -> 보유 앵커가 있어도 현금 (fail-closed)
    model.reset_trackers()
    assert model.score(snapshot, ctx("UNCERTAIN", {"122630": 0.95}, 23)) is CASH_INTENT

    # 보유 앵커 손절 -> 다른 앵커로 교체
    stopped = snapshot.with_columns(
        pl.when(pl.col("ticker") == "122630").then(-0.20).otherwise(pl.col("drawdown_20")).alias("drawdown_20")
    )
    model.reset_trackers()
    assert model.score(stopped, ctx("INACTIVE", {"122630": 0.95}, 24)) == {"233740": 0.20}

    # 전 앵커 손절 -> 현금
    all_stopped = snapshot.with_columns(pl.lit(-0.20).alias("drawdown_20"))
    model.reset_trackers()
    assert model.score(all_stopped, ctx("CRASH_REBOUND", {}, 25)) is CASH_INTENT

    # LOTTERY_ON -> P27 mom60 경로로 인계 (앵커 라우트 미적용)
    model.reset_trackers()
    lottery = model.score(snapshot, ctx("LOTTERY_ON", {}, 28))
    assert isinstance(lottery, dict)
    assert max(lottery, key=lottery.get) == "SEMI"



def test_make_sticky_mom60_post_crash_anchor_builds_from_yaml() -> None:
    from src.strategies.ids import STICKY_MOM60_POST_CRASH_ANCHOR
    from src.strategies.registry import STRATEGIES, resolve_strategy_id

    assert STICKY_MOM60_POST_CRASH_ANCHOR == "sticky.mom60_post_crash_anchor"
    assert resolve_strategy_id("sticky.mom60_post_crash_anchor") == STICKY_MOM60_POST_CRASH_ANCHOR

    model = STRATEGIES[STICKY_MOM60_POST_CRASH_ANCHOR]()
    cfg = model.config  # type: ignore[attr-defined]

    assert model.name == STICKY_MOM60_POST_CRASH_ANCHOR
    assert cfg.post_crash_anchor is True
    assert cfg.anchor_tickers == ("233740", "122630")
    assert cfg.anchor_score_col == "mom_20"
    assert cfg.anchor_stop_drawdown == 0.15
    # P27 레시피 상속
    assert cfg.mom_col == "mom_60"
    assert cfg.min_gap == 0.04
    assert cfg.min_hold == 2
    assert cfg.abs_mom_cash is True
    assert cfg.exclude_synthetic is True
    assert cfg.min_fill_ratio == 0.25
    assert cfg.inactive_participation is False



import pytest


def test_make_sticky_mom60_post_crash_anchor_fails_closed_without_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.strategies.factories import sticky_mom60

    monkeypatch.setattr(
        "src.strategies.sticky.config.read_sticky_yaml_block", lambda strategy_id, path=None: {}
    )

    with pytest.raises(ValueError):
        sticky_mom60.make_sticky_mom60_post_crash_anchor()



def test_post_crash_anchor_yaml_block_matches_probe_parameters() -> None:
    import yaml

    from src.core.config import config_path

    with config_path("strategies").open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    block = document["portfolio"]["sticky"]["mom60_post_crash_anchor"]

    assert block["post_crash_anchor"] is True
    assert block["anchor_tickers"] == ["233740", "122630"]
    assert block["anchor_score_col"] == "mom_20"
    assert block["anchor_stop_drawdown"] == 0.15
    assert block["mom_col"] == "mom_60"
    assert block["min_fill_ratio"] == 0.25
    assert block["exclude_synthetic"] is True
    assert block["max_single_weight"] == 0.95
    assert block["max_gross_exposure"] == 1.9
    assert block["min_cash"] == 0.05



def test_post_crash_anchor_uses_p27_exposure_limits_and_facade() -> None:
    from src.portfolio.constraints import load_p27_exposure_limits, resolve_exposure_limits_for_model
    from src.strategies.factories import sticky as facade
    from src.strategies.factories.sticky_mom60 import make_sticky_mom60_post_crash_anchor as origin

    assert resolve_exposure_limits_for_model("sticky.mom60_post_crash_anchor") == load_p27_exposure_limits()
    assert facade.make_sticky_mom60_post_crash_anchor is origin
    assert "make_sticky_mom60_post_crash_anchor" in facade.__all__

