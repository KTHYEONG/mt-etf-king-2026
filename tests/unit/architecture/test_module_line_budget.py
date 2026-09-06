import ast
from pathlib import Path

STATEMENT_BUDGET = 400


def test_module_statement_budget() -> None:
    offenders: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.stmt))
        if count > STATEMENT_BUDGET:
            offenders.append(f"{path}:{count}>{STATEMENT_BUDGET}")

    assert offenders == [], "modules exceed statement budget: " + ", ".join(offenders)


def test_no_multi_statement_lines() -> None:
    offenders: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(source.splitlines(), start=1):
            if len(line) > 200:
                offenders.append(f"{path}:{lineno}:len={len(line)}")
        seen: dict[int, int] = {}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.stmt) and hasattr(node, "lineno"):
                seen[node.lineno] = seen.get(node.lineno, 0) + 1
        for lineno, count in sorted(seen.items()):
            if count > 1 and not _is_compound_header(source, lineno):
                offenders.append(f"{path}:{lineno}:stmts={count}")

    assert offenders == [], "packed lines: " + ", ".join(offenders[:20])


def _is_compound_header(source: str, lineno: int) -> bool:
    """A `class X: pass` style header legitimately holds two stmt nodes on one line only if it has no ';'."""
    lines = source.splitlines()
    if lineno - 1 >= len(lines):
        return False
    return ";" not in lines[lineno - 1]
