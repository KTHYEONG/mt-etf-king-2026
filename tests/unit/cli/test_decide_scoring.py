"""P4 CLI decomposition: decide/scoring.py panel loader surface."""
from src.cli.commands.decide import scoring as scoring_mod


def test_p4_decide_scoring_surface() -> None:
    assert callable(scoring_mod._load_panel_for_backtest)
