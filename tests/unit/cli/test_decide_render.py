"""P4 CLI decomposition: decide/render.py + scoring.py surface."""
from src.cli.commands.decide import render as render_mod
from src.cli.commands.decide import scoring as scoring_mod
from src.cli.commands.decide.render import render_decision


def test_p4_decide_render_scoring_surface() -> None:
    assert callable(render_decision)
    assert callable(scoring_mod._load_panel_for_backtest)
    assert render_mod is not None
