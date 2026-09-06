"""P4 CLI decomposition: backtest/_core.py run loop surface."""
import dataclasses

from src.cli.commands.backtest import _core as core_mod
from src.cli.commands.backtest._core import _Cell, run_cells, run_preflight


def test_p4_backtest_core_surface() -> None:
    assert callable(run_preflight)
    assert callable(run_cells)
    assert callable(core_mod._run_common_tail)
    assert isinstance(core_mod._PREFLIGHT_STEPS, tuple)
    cell_fields = {f.name for f in dataclasses.fields(_Cell)}
    assert "model" in cell_fields
    assert "b1_anchor_cache" in cell_fields
