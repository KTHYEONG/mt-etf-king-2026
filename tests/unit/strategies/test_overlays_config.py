import pytest

from src.core.config import ConfigError
from src.strategies import ids


def test_overlay_param_replaces_all_per_strategy_loaders() -> None:
    from src.strategies.sticky import overlays

    removed = [
        "load_p22_lock_level",
        "load_p24_lock_level",
        "load_p24_trail",
        "load_p24_mom_col",
        "load_p25_arm",
        "load_p25_lock_remaining",
        "load_p26_arm",
        "load_p26_lock_remaining",
    ]
    still_present = [n for n in removed if hasattr(overlays, n)]
    assert still_present == [], f"legacy per-strategy loaders remain: {still_present}"

    value = overlays.overlay_param(ids.STICKY_FAMILY_PEAK_LOCK, "lock_level", default=0.50)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0

    with pytest.raises(ConfigError, match="unknown strategy"):
        overlays.overlay_param("not.a_strategy", "lock_level")
