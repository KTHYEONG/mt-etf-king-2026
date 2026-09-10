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


def test_inactive_participate_yaml_block_matches_probe_parameters() -> None:
    import yaml

    from src.core.config import config_path

    with config_path("strategies").open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    block = document["portfolio"]["sticky"]["mom60_inactive_participate"]

    assert block["inactive_participation"] is True
    assert block["inactive_score_col"] == "rv_20"
    assert block["inactive_min_weight"] == 0.30
    assert block["inactive_stop_drawdown"] == 0.15
    # P27 레시피/한도 상속
    assert block["mom_col"] == "mom_60"
    assert block["min_fill_ratio"] == 0.25
    assert block["exclude_synthetic"] is True
    assert block["max_single_weight"] == 0.95
    assert block["max_gross_exposure"] == 1.9
    assert block["min_cash"] == 0.05

