import ast
import sys
from pathlib import Path


def _load_verify_wiring():
    sys.path.insert(0, str(Path("tools/agent_skills").resolve()))
    import lean_check

    return lean_check.verify_wiring


def test_verify_wiring_rejects_string_literal_only(tmp_path: Path) -> None:
    # Given: a module that only mentions the symbol inside a dead anchor
    fake = tmp_path / "anchor_only.py"
    fake.write_text(
        "from __future__ import annotations\n"
        "_ = \"resolve_strategy_id(model_key)\"\n"
        "_ = 'if model_key == \"P23\":'\n",
        encoding="utf-8",
    )
    verify_wiring = _load_verify_wiring()

    # When
    ok, reason = verify_wiring(fake, "resolve_strategy_id", "resolve_strategy_id(model_key)")

    # Then
    assert ok is False
    assert reason

    # Given: a module that genuinely imports and calls it
    real = tmp_path / "real_call.py"
    real.write_text(
        "from src.strategies.registry import resolve_strategy_id\n"
        "\n"
        "def run(model_key: str) -> str:\n"
        "    return resolve_strategy_id(model_key)\n",
        encoding="utf-8",
    )
    ok2, _reason2 = verify_wiring(real, "resolve_strategy_id", "resolve_strategy_id(model_key)")
    assert ok2 is True
    assert isinstance(ast.parse(real.read_text(encoding="utf-8")), ast.Module)


def test_verify_wiring_missing_file_fails_closed(tmp_path: Path) -> None:
    verify_wiring = _load_verify_wiring()

    # Given: a path that does not exist
    missing = tmp_path / "does_not_exist.py"

    # When / Then: fails closed instead of raising
    ok, reason = verify_wiring(missing, "foo", "foo()")
    assert ok is False
    assert reason

    # Given: a file with a syntax error
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n", encoding="utf-8")

    # When / Then
    ok2, reason2 = verify_wiring(broken, "foo", "foo()")
    assert ok2 is False
    assert reason2
