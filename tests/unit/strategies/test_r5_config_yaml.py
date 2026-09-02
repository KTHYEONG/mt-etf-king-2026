def test_r5_yaml_semantic_only_blocks() -> None:
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(Path("configs/strategies.yaml").read_text(encoding="utf-8"))
    port = raw["portfolio"]
    assert "sticky" in port
    assert "mom60_raw" in port["sticky"]
    assert "p27" not in port
    assert "p21" not in port

def test_r5_read_sticky_yaml_semantic_first() -> None:
    from src.strategies.ids import STICKY_MOM60_RAW
    from src.strategies.sticky.config import read_sticky_yaml_block

    block = read_sticky_yaml_block(STICKY_MOM60_RAW)
    assert block.get("overlay_mode") == "identity"
