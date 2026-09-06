import pytest

from src.strategies.ids import STICKY_MOM60_RAW
from src.strategies.registry import resolve_strategy_id


@pytest.mark.parametrize("legacy", ["P27", "P20", "P28A", "B0", "M07", "p27"])
def test_resolve_strategy_id_rejects_legacy_codes(legacy: str) -> None:
    with pytest.raises(ValueError, match="unknown strategy") as exc:
        resolve_strategy_id(legacy)
    assert "unknown strategy" in str(exc.value).lower()

    assert resolve_strategy_id(STICKY_MOM60_RAW) == STICKY_MOM60_RAW
    assert resolve_strategy_id("STICKY.MOM60_RAW") == STICKY_MOM60_RAW


def test_every_strategy_name_equals_its_semantic_id() -> None:
    from src.strategies.registry import STRATEGIES

    mismatches: list[str] = []
    for semantic_id, factory in sorted(STRATEGIES.items()):
        obj = factory()
        name = getattr(obj, "name", None)
        if name != semantic_id:
            mismatches.append(f"{semantic_id}->{name!r}")

    assert mismatches == [], "strategy identity is not single-valued: " + ", ".join(mismatches)
    assert len(STRATEGIES) >= 30
