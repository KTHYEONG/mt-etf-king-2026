"""P4 CLI decomposition: backtest/_prep.py run preparation surface."""
import dataclasses

from src.cli.commands.backtest._prep import _Prep, prepare_run


def test_p4_backtest_prep_surface() -> None:
    assert callable(prepare_run)
    prep_fields = {f.name for f in dataclasses.fields(_Prep)}
    assert "b1_anchor_cache" in prep_fields
    assert "close_map" in prep_fields
    assert _Prep().b1_anchor_cache == {}
