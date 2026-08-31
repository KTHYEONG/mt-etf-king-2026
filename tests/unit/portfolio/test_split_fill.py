def test_fillable_weight_inv11_bounds() -> None:
    import math

    from src.portfolio.split_fill import fillable_weight

    equity = 1_000_000_000.0
    part = 0.01
    assert abs(fillable_weight(1.0e11, equity, part) - 1.0) < 1e-12
    assert abs(fillable_weight(4.56e10, equity, part) - 0.456) < 1e-12
    assert fillable_weight(0.0, equity, part) == 0.0
    assert fillable_weight(float("nan"), equity, part) == 0.0
    assert fillable_weight(1.0e11, 0.0, part) == 1.0
    assert fillable_weight(1.0e11, equity, 0.0) == 1.0
    w = fillable_weight(4.56e10, equity, part)
    assert w * equity <= 4.56e10 * part + 1e-6
    assert 0.0 <= w <= 1.0
    assert math.isfinite(w)


def test_split_residual_plus2_full_when_adv_covers() -> None:
    from src.portfolio.split_fill import split_residual_plus2

    scores = {"122630": 0.21, "494310": 0.55}
    adv = {"122630": 1.0e12, "494310": 4.56e10}
    fam = {"122630": "KOSPI 200", "494310": "KRX 반도체"}
    out = split_residual_plus2("122630", scores, adv, family_by_ticker=fam, equity=1.0e9, participation=0.01)
    assert out.weights == {"122630": 1.0}
    assert out.reason == "FULL_LEADER"
    assert out.sleeve is None
    assert abs(out.fillable - 1.0) < 1e-12


def test_split_residual_plus2_theme_leader_uses_liquid_sleeve() -> None:
    from src.portfolio.split_fill import split_residual_plus2

    scores = {"494310": 0.557, "122630": 0.212, "069500": 0.10}
    adv = {"494310": 4.56e10, "122630": 6.6e12, "069500": 9.0e12}
    fam = {"494310": "KRX 반도체", "122630": "KOSPI 200", "069500": "KOSPI 200"}
    out = split_residual_plus2("494310", scores, adv, family_by_ticker=fam, equity=1.0e9, participation=0.01)
    assert out.reason == "SPLIT_SLEEVE"
    assert out.leader == "494310"
    assert out.sleeve == "122630"
    assert "069500" not in out.weights
    w_l = out.weights["494310"]
    w_s = out.weights["122630"]
    assert abs(w_l - 0.456) < 1e-9
    assert abs(w_s - (1.0 - 0.456)) < 1e-9
    assert abs(w_l + w_s - 1.0) < 1e-9
    assert w_l * 1.0e9 <= 4.56e10 * 0.01 + 1.0


def test_split_residual_plus2_sleeve_excludes_same_family() -> None:
    from src.portfolio.split_fill import pick_liquidity_sleeve, split_residual_plus2

    scores = {"494310": 0.55, "488080": 0.49, "122630": 0.21}
    adv = {"494310": 4.0e10, "488080": 8.0e10, "122630": 5.0e12}
    fam = {"494310": "반도체", "488080": "반도체", "122630": "KOSPI 200"}
    sleeve = pick_liquidity_sleeve("494310", scores, adv, fam)
    assert sleeve == "122630"
    out = split_residual_plus2("494310", scores, adv, family_by_ticker=fam, equity=1.0e9, participation=0.01)
    assert out.sleeve == "122630"
    assert "488080" not in out.weights


def test_split_residual_plus2_fail_closed_empty() -> None:
    from src.portfolio.split_fill import split_residual_plus2

    fam = {"A": "f"}
    assert split_residual_plus2(None, {"A": 1.0}, {"A": 1e12}, family_by_ticker=fam, equity=1e9, participation=0.01).weights == {}
    assert split_residual_plus2("A", {}, {"A": 1e12}, family_by_ticker=fam, equity=1e9, participation=0.01).weights == {}
    assert split_residual_plus2("Z", {"A": 1.0}, {"A": 1e12}, family_by_ticker=fam, equity=1e9, participation=0.01).weights == {}


def test_split_residual_plus2_logs_algo_tag(caplog) -> None:
    import logging

    from src.portfolio.split_fill import split_residual_plus2

    scores = {"494310": 0.55, "122630": 0.21}
    adv = {"494310": 4.56e10, "122630": 6.6e12}
    fam = {"494310": "SEMI", "122630": "KOSPI 200"}
    with caplog.at_level(logging.DEBUG):
        split_residual_plus2("494310", scores, adv, family_by_ticker=fam, equity=1.0e9, participation=0.01)
    joined = "\n".join(r.message for r in caplog.records)
    assert "[ALGO]" in joined
    assert "leader=" in joined
    assert "fillable=" in joined
