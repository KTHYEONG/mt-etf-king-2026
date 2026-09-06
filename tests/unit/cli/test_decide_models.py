"""P4 CLI decomposition: decide/models.py hook tables."""
from src.cli.commands.decide.models import _ALLOCATE_HOOKS, _DecideState, _OVERLAY_HOOKS


def test_p4_decide_models_hook_tables() -> None:
    assert isinstance(_DecideState, type)
    assert "sticky.split_fill_lock" in _ALLOCATE_HOOKS
    assert "sticky.mom60_raw" in _OVERLAY_HOOKS
    assert "convex.lottery_impulse" in _OVERLAY_HOOKS
    assert all(callable(h) for h in _ALLOCATE_HOOKS.values())
    assert all(callable(h) for h in _OVERLAY_HOOKS.values())
