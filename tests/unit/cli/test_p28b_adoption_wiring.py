def test_p28b_cli_championship_wires_p27_champion() -> None:
    import re
    from pathlib import Path

    from src.cli import STICKY_ADOPTION_MODELS

    bt = Path("src/cli.py").read_text(encoding="utf-8")
    assert "P28B" in STICKY_ADOPTION_MODELS
    assert 'if model_key == "P28B"' in bt
    idx = bt.find('if model_key == "P28B"')
    assert idx > 0
    block = bt[idx : idx + 4000]
    assert "evaluate_championship_adoption" in block
    assert 'BASELINES["P27"]' in block or "BASELINES['P27']" in block
    assert "run_rolling" in block
    assert re.search(r"^\s*_ = diagnostics\b", block, flags=re.M) is None
