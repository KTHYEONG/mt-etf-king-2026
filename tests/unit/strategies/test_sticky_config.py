def test_load_overlay_mode_semantic_default() -> None:
    from src.strategies.ids import STICKY_MOM60_RAW
    from src.strategies.sticky.config import load_overlay_mode

    assert load_overlay_mode(strategy_key=STICKY_MOM60_RAW) == "identity"
