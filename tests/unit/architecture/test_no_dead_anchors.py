import ast
from pathlib import Path


def test_no_dead_anchors() -> None:
    offenders: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "_":
                offenders.append(f"{path}:{node.lineno}")

    assert offenders == [], f"{len(offenders)} dead wiring anchors remain: " + ", ".join(offenders[:20])
