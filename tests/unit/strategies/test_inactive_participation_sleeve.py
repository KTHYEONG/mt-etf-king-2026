# ruff: noqa
def test_score_inactive_participation_selects_capacitated_high_vol_name() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.portfolio.intent import CASH_INTENT
    from src.strategies.sticky.model_config import StickyLeaderConfig
    from src.strategies.sticky.model_runner import StickyLeaderModel
    from src.tournament.objective.cutoff_auc import CUTOFF_AUC_IS_PRODUCTION_GATE

    assert CUTOFF_AUC_IS_PRODUCTION_GATE is True, "attack-sleeve 라우팅이 프로덕션 경로여야 한다"

    capital = 1_000_000_000.0
    snapshot = pl.DataFrame(
        {
            "ticker": ["THIN", "DEEP"],
            "name": [
                "SOL \uc18c\ud615\uc8fc\ub2e8\uc77c\uc885\ubaa9\ub808\ubc84\ub9ac\uc9c0",
                "KODEX \ucf54\uc2a4\ub2e5150 \ub808\ubc84\ub9ac\uc9c0",
            ],
            "mom_60": [0.20, 0.10],
            "rv_20": [0.90, 0.05],
            "trading_value": [2_000_000_000.0, 400_000_000_000.0],
        }
    )
    ctx = DecisionContext(
        decision_date=date(2026, 9, 8),
        regime=None,
        capital=capital,
        held={},
        rules=SimpleNamespace(initial_capital=capital, max_order_to_adv=0.01),  # type: ignore[arg-type]
        championship_sleeve="INACTIVE",
    )

    # 플래그 OFF: 기존 동작(현금) 유지
    off = StickyLeaderModel(
        name="sticky.mom60_raw",
        config=StickyLeaderConfig(mom_col="mom_60", min_fill_ratio=0.25),
    )
    assert off.score(snapshot, ctx) is CASH_INTENT

    # 플래그 ON: 체결가능한 종목 선택 (THIN은 rv 최대지만 용량 미달)
    on = StickyLeaderModel(
        name="sticky.mom60_inactive_participate",
        config=StickyLeaderConfig(
            mom_col="mom_60", min_fill_ratio=0.25, inactive_participation=True
        ),
    )
    result = on.score(snapshot, ctx)
    assert isinstance(result, dict)
    assert set(result) == {"DEEP"}



def test_score_inactive_participation_latches_stop_for_rest_of_campaign() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.portfolio.intent import CASH_INTENT
    from src.strategies.sticky.model_config import StickyLeaderConfig
    from src.strategies.sticky.model_runner import StickyLeaderModel

    snapshot = pl.DataFrame(
        {
            "ticker": ["DEEP"],
            "name": ["KODEX \ucf54\uc2a4\ub2e5150 \ub808\ubc84\ub9ac\uc9c0"],
            "mom_60": [0.10],
            "rv_20": [0.05],
            "trading_value": [400_000_000_000.0],
        }
    )

    def ctx_for(capital: float, day: int, held: dict[str, float]) -> DecisionContext:
        return DecisionContext(
            decision_date=date(2026, 9, day),
            regime=None,
            capital=capital,
            held=dict(held),
            rules=SimpleNamespace(initial_capital=1_000_000_000.0, max_order_to_adv=0.01),  # type: ignore[arg-type]
            championship_sleeve="INACTIVE",
        )

    model = StickyLeaderModel(
        name="sticky.mom60_inactive_participate",
        config=StickyLeaderConfig(
            mom_col="mom_60",
            min_fill_ratio=0.25,
            inactive_participation=True,
            inactive_stop_drawdown=0.15,
        ),
    )

    # 1일차: 진입 (peak = 1.0e9)
    first = model.score(snapshot, ctx_for(1_000_000_000.0, 1, {}))
    assert isinstance(first, dict) and set(first) == {"DEEP"}

    # 2일차: 자본 상승 -> peak 갱신 (1.2e9), 계속 보유
    second = model.score(snapshot, ctx_for(1_200_000_000.0, 2, {"DEEP": 0.95}))
    assert isinstance(second, dict) and set(second) == {"DEEP"}

    # 3일차: peak 대비 -16% (1.008e9) -> 손절 래치
    third = model.score(snapshot, ctx_for(1_008_000_000.0, 3, {"DEEP": 0.95}))
    assert third is CASH_INTENT

    # 4일차: 자본이 신고가로 회복되어도 같은 캠페인에서는 재진입 금지
    fourth = model.score(snapshot, ctx_for(1_500_000_000.0, 4, {}))
    assert fourth is CASH_INTENT

    # 캠페인 경계(reset_trackers) 이후에는 다시 진입 가능
    model.reset_trackers()
    assert model._inactive_stopped is False
    assert model._inactive_peak_capital is None
    revived = model.score(snapshot, ctx_for(1_500_000_000.0, 7, {}))
    assert isinstance(revived, dict) and set(revived) == {"DEEP"}



def test_score_inactive_participation_fail_closed_paths() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.portfolio.intent import CASH_INTENT
    from src.strategies.sticky.model_config import StickyLeaderConfig
    from src.strategies.sticky.model_runner import StickyLeaderModel

    snapshot = pl.DataFrame(
        {
            "ticker": ["DEEP"],
            "name": ["KODEX \ucf54\uc2a4\ub2e5150 \ub808\ubc84\ub9ac\uc9c0"],
            "mom_60": [0.10],
            "rv_20": [0.05],
            "trading_value": [400_000_000_000.0],
        }
    )
    rules = SimpleNamespace(initial_capital=1_000_000_000.0, max_order_to_adv=0.01)

    def build(cfg: StickyLeaderConfig) -> StickyLeaderModel:
        return StickyLeaderModel(name="sticky.mom60_inactive_participate", config=cfg)

    def ctx(sleeve: str | None, capital: float) -> DecisionContext:
        return DecisionContext(
            decision_date=date(2026, 9, 8),
            regime=None,
            capital=capital,
            held={},
            rules=rules,  # type: ignore[arg-type]
            championship_sleeve=sleeve,
        )

    on_cfg = dict(mom_col="mom_60", min_fill_ratio=0.25, inactive_participation=True)

    # 국면 미확정 -> 참여 금지
    for sleeve in ("UNCERTAIN", None, "NOT_A_SLEEVE"):
        assert build(StickyLeaderConfig(**on_cfg)).score(snapshot, ctx(sleeve, 1_000_000_000.0)) is CASH_INTENT, sleeve

    # 자본 비유한/비양수 -> 참여 금지
    for cap in (0.0, -1.0, float("nan")):
        assert build(StickyLeaderConfig(**on_cfg)).score(snapshot, ctx("INACTIVE", cap)) is CASH_INTENT, cap

    # min_fill_ratio 비활성(용량 파라미터 없음) -> 참여 금지
    no_cap = StickyLeaderConfig(mom_col="mom_60", min_fill_ratio=0.0, inactive_participation=True)
    assert build(no_cap).score(snapshot, ctx("INACTIVE", 1_000_000_000.0)) is CASH_INTENT



def test_score_inactive_participation_does_not_alter_other_sleeves() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.sticky.model_config import StickyLeaderConfig
    from src.strategies.sticky.model_runner import StickyLeaderModel

    snapshot = pl.DataFrame(
        {
            "ticker": ["DEEP", "OTHER"],
            "name": [
                "KODEX \ucf54\uc2a4\ub2e5150 \ub808\ubc84\ub9ac\uc9c0",
                "KODEX \ub808\ubc84\ub9ac\uc9c0",
            ],
            "mom_60": [0.10, 0.30],
            "mom_20": [0.40, 0.20],
            "rv_20": [0.90, 0.05],
            "trading_value": [400_000_000_000.0, 900_000_000_000.0],
            "volume_expansion": [1.0, 1.0],
        }
    )
    rules = SimpleNamespace(initial_capital=1_000_000_000.0, max_order_to_adv=0.01)

    def result_for(participation: bool, sleeve: str) -> object:
        model = StickyLeaderModel(
            name="sticky.mom60_inactive_participate" if participation else "sticky.mom60_raw",
            config=StickyLeaderConfig(
                mom_col="mom_60", min_fill_ratio=0.25, inactive_participation=participation
            ),
        )
        ctx = DecisionContext(
            decision_date=date(2026, 9, 8),
            regime=None,
            capital=1_000_000_000.0,
            held={},
            rules=rules,  # type: ignore[arg-type]
            championship_sleeve=sleeve,
        )
        return model.score(snapshot, ctx)

    for sleeve in ("LOTTERY_ON", "CRASH_REBOUND"):
        off = result_for(False, sleeve)
        on = result_for(True, sleeve)
        assert type(off) is type(on), sleeve
        if isinstance(off, dict):
            assert off == on, sleeve
        else:
            assert off is on, sleeve



def test_inactive_participate_strategy_registered_with_p27_limits() -> None:
    from src.portfolio.constraints import resolve_exposure_limits_for_model
    from src.strategies.ids import STICKY_MOM60_INACTIVE_PARTICIPATE
    from src.strategies.registry import STRATEGIES

    assert STICKY_MOM60_INACTIVE_PARTICIPATE == "sticky.mom60_inactive_participate"
    assert STICKY_MOM60_INACTIVE_PARTICIPATE in STRATEGIES

    model = STRATEGIES[STICKY_MOM60_INACTIVE_PARTICIPATE]()
    assert model.name == STICKY_MOM60_INACTIVE_PARTICIPATE

    # P27 레시피 상속
    cfg = model.config
    assert cfg.mom_col == "mom_60"
    assert cfg.min_gap == 0.04
    assert cfg.min_hold == 2
    assert cfg.only_plus_2 is True
    assert cfg.no_inverse is True
    assert cfg.abs_mom_cash is True
    assert cfg.exclude_synthetic is True
    assert cfg.min_fill_ratio == 0.25
    # 이 전략의 정체성
    assert cfg.inactive_participation is True

    # INV-17/18: P27과 동일 한도
    limits = resolve_exposure_limits_for_model(STICKY_MOM60_INACTIVE_PARTICIPATE)
    assert limits == resolve_exposure_limits_for_model("sticky.mom60_raw")
    assert abs(limits[0] - 0.95) < 1e-9
    assert abs(limits[1] - 1.90) < 1e-9
    assert abs(limits[2] - 0.05) < 1e-9

