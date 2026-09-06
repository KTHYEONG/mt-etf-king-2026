def test_r2_shim_line_budget() -> None:
    from pathlib import Path

    caps = {
        "src/strategies/baselines/core.py": 600,
        "src/strategies/sticky/model.py": 700,
    }
    for rel, limit in caps.items():
        with Path(rel).open(encoding="utf-8") as fh:
            count = sum(1 for _ in fh)
        assert count <= limit, f"{rel} has {count} lines (max {limit})"


def test_r2_baselines_shim_reexports_core() -> None:
    # P3: src/alpha/baselines.py was deleted (factories moved to src/strategies/factories/);
    # BuyAndHoldBaseline now lives solely in src/strategies/baselines/core.py.
    from src.strategies.baselines.core import BuyAndHoldBaseline, make_baseline_buy_hold

    assert make_baseline_buy_hold().__class__ is BuyAndHoldBaseline


def test_r2_sticky_shim_reexports_model() -> None:
    from src.alpha.sticky import StickyLeaderModel
    from src.strategies.sticky.model import StickyLeaderModel as CoreModel

    assert StickyLeaderModel is CoreModel


def test_r2_factory_mom60_raw_matches_p27_invariants() -> None:
    from src.strategies.ids import STICKY_MOM60_RAW
    from src.strategies.sticky.factories import make_sticky_mom60_raw
    from src.strategies.sticky.model import StickyLeaderModel

    model = make_sticky_mom60_raw()
    assert isinstance(model, StickyLeaderModel)
    assert getattr(model, "name", None) in (STICKY_MOM60_RAW, "sticky.mom60_raw")
    assert str(model.config.mom_col) == "mom_60"
    assert float(model.config.min_gap) == 0.04
    assert int(model.config.min_hold) == 2


# test_r2_strategies_modules_line_budget removed (P0): superseded by the global,
# waiver-free AST statement budget in tests/unit/architecture/test_module_line_budget.py,
# which already covers src/strategies/** and cannot be gamed by raw line-packing.
