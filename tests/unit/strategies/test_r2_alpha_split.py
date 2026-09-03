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
    from src.alpha.baselines import BuyAndHoldBaseline
    from src.strategies.baselines.core import BuyAndHoldBaseline as CoreBuyAndHold

    assert BuyAndHoldBaseline is CoreBuyAndHold


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
    assert getattr(model, "name", None) in (STICKY_MOM60_RAW, "P27")
    assert str(model.config.mom_col) == "mom_60"
    assert float(model.config.min_gap) == 0.04
    assert int(model.config.min_hold) == 2


def test_r2_strategies_modules_line_budget() -> None:
    from pathlib import Path

    per_path_caps = {
        "src/strategies/sticky/model.py": 700,
        "src/strategies/sticky/capacity.py": 200,
    }
    default_cap = 600
    offenders: list[str] = []
    for path in Path("src/strategies").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = str(path)
        limit = per_path_caps.get(rel, default_cap)
        with path.open(encoding="utf-8") as fh:
            count = sum(1 for _ in fh)
        if count > limit:
            offenders.append(f"{rel}:{count}>{limit}")
    assert offenders == [], "strategies modules exceed line budget: " + ", ".join(offenders)
