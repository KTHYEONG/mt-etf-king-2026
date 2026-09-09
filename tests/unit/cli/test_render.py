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
