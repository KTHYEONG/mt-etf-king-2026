# ruff: noqa
def test_inactive_leader_scores_picks_highest_rv_among_capacitated_plus2() -> None:
    import polars as pl

    from src.strategies.sticky.model_scores import inactive_leader_scores

    capital = 1_000_000_000.0
    phi = 0.01
    min_w = 0.30
    # 달성가능 비중 = min(0.95, tv*0.01/1e9, 0.95); min_weight 0.30 -> tv >= 30e9 필요
    snapshot = pl.DataFrame(
        {
            "ticker": ["THIN_HIRV", "DEEP_MIDRV", "DEEP_LORV"],
            "name": [
                "SOL \uc18c\ud615\uc8fc\ub2e8\uc77c\uc885\ubaa9\ub808\ubc84\ub9ac\uc9c0",
                "KODEX \ucf54\uc2a4\ub2e5150 \ub808\ubc84\ub9ac\uc9c0",
                "KODEX \ub808\ubc84\ub9ac\uc9c0",
            ],
            "rv_20": [0.90, 0.05, 0.02],
            "trading_value": [2_000_000_000.0, 400_000_000_000.0, 900_000_000_000.0],
        }
    )

    scores = inactive_leader_scores(
        snapshot, capital=capital, max_order_to_adv=phi, min_weight=min_w
    )

    # THIN_HIRV는 rv 최대지만 tv 20\uc5b5 -> 비중 2% -> 용량 게이트 탈락
    assert scores == {"DEEP_MIDRV": 0.05}



def test_inactive_leader_scores_excludes_non_plus2_and_synthetic() -> None:
    import polars as pl

    from src.strategies.sticky.model_scores import inactive_leader_scores

    # 전부 초대형 유동성이지만 +2X 실물이 아닌 것들
    snapshot = pl.DataFrame(
        {
            "ticker": ["ONE_X", "INV_1X", "INV_2X", "SYNTH_2X"],
            "name": [
                "KODEX 200",
                "KODEX \uc778\ubc84\uc2a4",
                "KODEX 200\uc120\ubb3c\uc778\ubc84\uc2a42X",
                "TIGER \ucc28\uc774\ub098\uc804\uae30\ucc28\ub808\ubc84\ub9ac\uc9c0(\ud569\uc131)",
            ],
            "rv_20": [0.5, 0.6, 0.7, 0.9],
            "trading_value": [900_000_000_000.0] * 4,
        }
    )

    scores = inactive_leader_scores(
        snapshot, capital=1_000_000_000.0, max_order_to_adv=0.01, min_weight=0.30
    )

    assert scores == {}



def test_inactive_leader_scores_fail_closed_on_invalid_inputs() -> None:
    import polars as pl

    from src.strategies.sticky.model_scores import inactive_leader_scores

    good = pl.DataFrame(
        {
            "ticker": ["DEEP"],
            "name": ["KODEX \ucf54\uc2a4\ub2e5150 \ub808\ubc84\ub9ac\uc9c0"],
            "rv_20": [0.05],
            "trading_value": [400_000_000_000.0],
        }
    )
    kw = dict(capital=1_000_000_000.0, max_order_to_adv=0.01, min_weight=0.30)
    assert inactive_leader_scores(good, **kw) == {"DEEP": 0.05}, "sanity: 정상 입력은 선택된다"

    # 비 DataFrame / 빈 프레임 / 컬럼 결측
    assert inactive_leader_scores(object(), **kw) == {}  # type: ignore[arg-type]
    assert inactive_leader_scores(pl.DataFrame(), **kw) == {}
    assert inactive_leader_scores(good.drop("rv_20"), **kw) == {}
    assert inactive_leader_scores(good.drop("name"), **kw) == {}

    # 파라미터 fail-closed
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        assert inactive_leader_scores(good, capital=bad, max_order_to_adv=0.01, min_weight=0.30) == {}, bad
        assert inactive_leader_scores(good, capital=1e9, max_order_to_adv=bad, min_weight=0.30) == {}, bad
        assert inactive_leader_scores(good, capital=1e9, max_order_to_adv=0.01, min_weight=bad) == {}, bad

    # 비유한 스코어 / 결측 유동성
    nan_score = good.with_columns(pl.lit(float("nan")).alias("rv_20"))
    assert inactive_leader_scores(nan_score, **kw) == {}
    null_adv = good.with_columns(pl.lit(None, dtype=pl.Float64).alias("trading_value"))
    assert inactive_leader_scores(null_adv, **kw) == {}



def test_inactive_leader_scores_prefers_adv_column_and_breaks_ties_deterministically() -> None:
    import polars as pl

    from src.strategies.sticky.model_scores import inactive_leader_scores

    kw = dict(capital=1_000_000_000.0, max_order_to_adv=0.01, min_weight=0.30)

    # 'adv'가 있으면 그것이 우선: trading_value는 충분하지만 adv가 미달이면 탈락
    adv_wins = pl.DataFrame(
        {
            "ticker": ["DEEP"],
            "name": ["KODEX \ucf54\uc2a4\ub2e5150 \ub808\ubc84\ub9ac\uc9c0"],
            "rv_20": [0.05],
            "trading_value": [900_000_000_000.0],
            "adv": [1_000_000_000.0],
        }
    )
    assert inactive_leader_scores(adv_wins, **kw) == {}

    # 동점 -> ticker 오름차순의 작은 쪽
    tied = pl.DataFrame(
        {
            "ticker": ["BBB", "AAA"],
            "name": ["KODEX \ub808\ubc84\ub9ac\uc9c0", "TIGER \ub808\ubc84\ub9ac\uc9c0"],
            "rv_20": [0.30, 0.30],
            "trading_value": [900_000_000_000.0, 900_000_000_000.0],
        }
    )
    assert inactive_leader_scores(tied, **kw) == {"AAA": 0.30}

