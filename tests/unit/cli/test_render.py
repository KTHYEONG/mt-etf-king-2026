"""P4 CLI decomposition: decide/render.py + scoring.py surface."""
from datetime import date
from types import SimpleNamespace

import pytest

from src.cli.commands.decide import render as render_mod
from src.cli.commands.decide import scoring as scoring_mod
from src.cli.commands.decide.render import render_decision


def test_p4_decide_render_scoring_surface() -> None:
    assert callable(render_decision)
    assert callable(scoring_mod._load_panel_for_backtest)
    assert render_mod is not None


def test_render_decision_appends_order_estimate_block(capsys: pytest.CaptureFixture[str]) -> None:
    from src.tournament.live_decision import LiveOrderEstimate

    d = date(2026, 8, 27)
    estimate = LiveOrderEstimate(ticker="412570", weight=1.0, price_basis_date=d, price=726_500.0, est_shares=1_376, est_krw=1_376 * 726_500.0)

    code = render_decision(
        weights={"412570": 1.0},
        decision_weights=None,
        scores={"412570": 0.67},
        decision_date=d,
        args=SimpleNamespace(output=None, trace=False),
        peak_is_locked=False,
        house_money_is_locked=False,
        order_estimates={"412570": estimate},
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "412570" in out
    assert "1376" in out or "1,376" in out
    assert "추정" in out


def test_render_decision_persists_order_estimates_into_json_artifact(tmp_path) -> None:
    import json

    from src.tournament.live_decision import LiveOrderEstimate

    d = date(2026, 8, 27)
    estimate = LiveOrderEstimate(ticker="412570", weight=1.0, price_basis_date=d, price=726_500.0, est_shares=1_376, est_krw=1_376 * 726_500.0)
    out_path = tmp_path / "decision.json"

    code = render_decision(
        weights={"412570": 1.0},
        decision_weights=None,
        scores={"412570": 0.67},
        decision_date=d,
        args=SimpleNamespace(output=str(out_path), trace=False),
        peak_is_locked=False,
        house_money_is_locked=False,
        order_estimates={"412570": estimate},
    )

    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    item = next(s for s in payload["selected"] if s["ticker"] == "412570")
    assert item["est_shares"] == 1_376
    assert item["est_krw"] == pytest.approx(1_376 * 726_500.0)


def test_render_decision_cash_uses_explanation_reason_code(capsys: pytest.CaptureFixture[str]) -> None:
    from types import SimpleNamespace

    d = date(2026, 9, 21)
    code = render_decision(
        weights={},
        decision_weights=None,
        scores={"122630": 0.5},
        decision_date=d,
        args=SimpleNamespace(output=None, trace=False),
        peak_is_locked=True,
        house_money_is_locked=False,
        order_estimates=None,
        explanation_payload={
            "execution_date": "2026-09-28",
            "action": {"code": "SELL_ALL", "ko": "전량 매도", "from": "122630", "to": None},
            "regime": {"sleeve": "CRASH_REBOUND", "ko": "급락 후 반등"},
            "reason_code": "ANCHOR_STOP_CASH",
            "reason_ko": "손절로 현금 전환",
            "anchor_monitor": [],
            "quantity_note_ko": "주문 불필요",
        },
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "ANCHOR_STOP_CASH" in out
    assert "peak_lock" not in out


def test_render_decision_cash_without_codes_keeps_legacy_text(capsys: pytest.CaptureFixture[str]) -> None:
    from types import SimpleNamespace

    d = date(2026, 9, 21)
    for house_locked in (True, False):
        code = render_decision(
            weights={},
            decision_weights=None,
            scores={"122630": 0.5},
            decision_date=d,
            args=SimpleNamespace(output=None, trace=False),
            peak_is_locked=True,
            house_money_is_locked=house_locked,
            order_estimates=None,
            explanation_payload={"explanation_error": "boom"},
        )

        assert code == 0
        out = capsys.readouterr().out
        if house_locked:
            assert "house_money" in out
        else:
            assert "peak_lock" in out


def test_render_decision_appends_explanation_block(capsys: pytest.CaptureFixture[str]) -> None:
    from types import SimpleNamespace

    from src.tournament.live_decision import LiveOrderEstimate

    d = date(2026, 9, 21)
    estimate = LiveOrderEstimate(ticker="122630", weight=0.95, price_basis_date=d, price=114060.0, est_shares=8328, est_krw=8328 * 114060.0)
    code = render_decision(
        weights={"122630": 0.95},
        decision_weights=None,
        scores={"122630": 0.5},
        decision_date=d,
        args=SimpleNamespace(output=None, trace=False),
        peak_is_locked=False,
        house_money_is_locked=False,
        order_estimates={"122630": estimate},
        explanation_payload={
            "execution_date": "2026-09-28",
            "action": {"code": "HOLD", "ko": "보유 유지(주문 없음)", "from": "122630", "to": "122630"},
            "regime": {"sleeve": "CRASH_REBOUND", "ko": "급락 후 반등"},
            "reason_code": "ANCHOR_LATCH_HOLD",
            "reason_ko": "보유 앵커 122630 유지",
            "anchor_monitor": [
                {"ticker": "122630", "name": "KODEX 레버리지", "close": 114060.0, "mom_20": 0.05, "drawdown_20": -0.0132, "roll_max_20": 115580.0, "stop_close": 98243.0, "stopped": False},
            ],
            "quantity_note_ko": "보유 유지 시 주문 불필요",
        },
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "보유 유지(주문 없음)" in out
    assert "122630 → 122630" in out
    assert "2026-09-28" in out
    assert "보유 앵커 122630 유지" in out
    assert "98243" in out
    assert "목표 수량" in out
