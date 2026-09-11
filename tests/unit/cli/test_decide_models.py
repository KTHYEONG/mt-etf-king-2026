"""P4 CLI decomposition: decide/models.py hook tables."""
from src.cli.commands.decide.models import _ALLOCATE_HOOKS, _DecideState, _OVERLAY_HOOKS


def test_p4_decide_models_hook_tables() -> None:
    assert isinstance(_DecideState, type)
    assert "sticky.split_fill_lock" in _ALLOCATE_HOOKS
    assert "sticky.mom60_raw" in _OVERLAY_HOOKS
    assert "convex.lottery_impulse" in _OVERLAY_HOOKS
    assert all(callable(h) for h in _ALLOCATE_HOOKS.values())
    assert all(callable(h) for h in _OVERLAY_HOOKS.values())


def test_decide_models_hook_tables_wiring_preserved() -> None:
    assert set(_ALLOCATE_HOOKS) == {"sticky.split_fill_lock", "sticky.mom60_raw", "sticky.mom60_post_crash_anchor"}
    assert set(_OVERLAY_HOOKS) == {
        "sticky.split_fill_lock",
        "sticky.mom60_peak_lock",
        "sticky.house_money",
        "sticky.mom60_concentrated",
        "sticky.mom60_raw",
        "sticky.mom60_hold",
        "sticky.mom60_abs_cash",
        "sticky.equity_mom60",
        "sticky.equity_mom60_vol",
        "sticky.fillable_mom60",
        "convex.lottery_impulse",
        "sticky.mom60_runner_reversal",
        "sticky.mom60_post_crash_anchor",
    }
    for hook in _OVERLAY_HOOKS.values():
        assert callable(hook)
    assert _DecideState.__dataclass_fields__["decision_date"]
