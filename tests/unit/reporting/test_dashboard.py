"""SCENARIO-08-14"""
from datetime import date

from src.reporting.dashboard import DailyDecision, build_rationale, render_dashboard, write_decision_artifact


def test_SCENARIO_08_14_rationale_and_dashboard() -> None:  # noqa: N802
    pos = {"ticker": "069500", "weight": 0.6, "state": "HOLD", "theme": "ThemeA"}
    r = build_rationale(pos)
    assert isinstance(r, str)  # noqa: PT018
    assert len(r) > 0  # noqa: PT018
    # render dashboard includes WHY and tickers
    dd = DailyDecision(
        decision_date=date(2026, 10, 7),
        weights={"069500": 0.6, "451060": 0.4},
        rationales={"069500": r, "451060": build_rationale({"ticker": "451060", "weight": 0.4})},
    )
    out = render_dashboard(dd)
    assert "WHY" in out
    assert "069500" in out
    assert "451060" in out
    # missing rationale -> omit position
    dd2 = DailyDecision(
        decision_date=date(2026, 10, 7),
        weights={"069500": 0.6, "X": 0.4},
        rationales={"069500": r, "X": ""},
    )
    out2 = render_dashboard(dd2)
    assert "X: 0.4" not in out2  # noqa: PT018


def test_write_decision_artifact_includes_order_estimate_shares_and_krw(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    dd = DailyDecision(
        decision_date=date(2026, 8, 27),
        weights={"412570": 1.0},
        rationales={"412570": "WHY: 412570 weight=1.000 state=HOLD"},
    )
    est = SimpleNamespace(est_shares=1_376, est_krw=1_376 * 726_500.0)
    out_path = tmp_path / "decision.json"

    write_decision_artifact(dd, out_path, order_estimates={"412570": est})

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    item = next(s for s in payload["selected"] if s["ticker"] == "412570")
    assert item["est_shares"] == 1_376
    assert item["est_krw"] == 1_376 * 726_500.0


def test_write_decision_artifact_omits_estimate_fields_when_none_given(tmp_path) -> None:
    import json

    dd = DailyDecision(
        decision_date=date(2026, 8, 27),
        weights={"412570": 1.0},
        rationales={"412570": "WHY: 412570 weight=1.000 state=HOLD"},
    )
    out_path = tmp_path / "decision.json"

    write_decision_artifact(dd, out_path)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    item = next(s for s in payload["selected"] if s["ticker"] == "412570")
    assert "est_shares" not in item
    assert "est_krw" not in item


def test_write_decision_artifact_includes_korean_fields(tmp_path) -> None:
    import json

    dd = DailyDecision(
        decision_date=date(2026, 9, 17),
        weights={"122630": 0.95},
        rationales={"122630": "WHY: 122630 weight=0.950 state=HOLD theme=ThemeA allocated via ClusterAwareSelection and confidence sizing"},
    )
    out_path = tmp_path / "decision.json"
    write_decision_artifact(dd, out_path)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    item = next(s for s in payload["selected"] if s["ticker"] == "122630")
    assert item["name"] == "KODEX 레버리지"
    assert "HOLD" in item["state"]
    assert "보유" in item["state"]
    assert "클러스터" in item["reason_ko"]
    assert "95.0%" in item["reason_ko"]
    assert item["weight"] == 0.95
    assert "WHY:" in item["reason"]


import pytest


@pytest.mark.parametrize("scenario_id", ["SCENARIO-08-14"])
def test_SCENARIO_hyphen_wrapper(scenario_id: str) -> None:  # noqa: N802
    if scenario_id == "SCENARIO-08-14":
        test_SCENARIO_08_14_rationale_and_dashboard()


def test_write_decision_artifact_merges_hold_explanation(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    dd = DailyDecision(
        decision_date=date(2026, 9, 21),
        weights={"122630": 0.95},
        rationales={"122630": "WHY: 122630 weight=0.950 state=HOLD"},
    )
    est = SimpleNamespace(est_shares=8328, est_krw=1.0)
    payload_in = {
        "execution_date": "2026-09-28",
        "action": {"code": "HOLD", "ko": "보유 유지(주문 없음)", "from": "122630", "to": "122630"},
        "regime": {"sleeve": "CRASH_REBOUND", "ko": "급락 후 반등"},
        "reason_code": "ANCHOR_LATCH_HOLD",
        "reason_ko": "보유 앵커 122630 유지",
        "anchor_monitor": [],
        "quantity_note_ko": "주문 불필요",
    }
    out_path = tmp_path / "decision.json"
    write_decision_artifact(dd, out_path, order_estimates={"122630": est}, explanation_payload=payload_in)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["action"]["code"] == "HOLD"
    item = next(s for s in payload["selected"] if s["ticker"] == "122630")
    assert item["state"] == "보유 유지(주문 없음)"
    assert item["reason_ko"] == "보유 앵커 122630 유지"
    assert item["est_basis_ko"] == "초기자본·결정일 종가 기준 목표 수량"
    assert item["est_shares"] == 8328


def test_write_decision_artifact_cash_is_explicit(tmp_path) -> None:
    import json

    dd = DailyDecision(
        decision_date=date(2026, 9, 21),
        weights={},
        rationales={"CASH": "WHY: ANCHOR_STOP_CASH state=SELL_ALL"},
    )
    payload_in = {
        "execution_date": "2026-09-28",
        "action": {"code": "SELL_ALL", "ko": "전량 매도", "from": "122630", "to": None},
        "regime": {"sleeve": "CRASH_REBOUND", "ko": "급락 후 반등"},
        "reason_code": "ANCHOR_STOP_CASH",
        "reason_ko": "손절로 현금 전환",
        "anchor_monitor": [],
        "quantity_note_ko": "주문 불필요",
    }
    out_path = tmp_path / "decision.json"
    write_decision_artifact(dd, out_path, explanation_payload=payload_in)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["selected"] == []
    assert payload["action"]["code"] == "SELL_ALL"
    assert payload["reason_code"] == "ANCHOR_STOP_CASH"


def test_write_decision_artifact_falls_back_to_state_map_without_ko(tmp_path) -> None:
    import json

    dd = DailyDecision(
        decision_date=date(2026, 9, 21),
        weights={"122630": 0.95},
        rationales={"122630": "WHY: 122630 weight=0.950 state=HOLD"},
    )
    payload_in = {
        "action": {"code": "SWITCH", "from": "122630", "to": "233740"},
        "reason_ko": "교체",
    }
    out_path = tmp_path / "decision.json"
    write_decision_artifact(dd, out_path, explanation_payload=payload_in)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    item = next(s for s in payload["selected"] if s["ticker"] == "122630")
    assert item["state"] == "종목 교체(전량 매도 후 매수)"
    assert item["reason_ko"] == "교체"

