import ast
from pathlib import Path

FORBIDDEN = {
    "src/strategies/factories": ("src.portfolio", "src.tournament", "src.cli"),
    "src/alpha": ("src.tournament", "src.cli"),
    "src/portfolio": ("src.tournament", "src.cli"),
}


def test_layer_boundaries_preserved() -> None:
    offenders: list[str] = []
    for root, forbidden_prefixes in FORBIDDEN.items():
        for path in sorted(Path(root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                module = ""
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module
                elif isinstance(node, ast.Import):
                    module = node.names[0].name
                if module and module.startswith(forbidden_prefixes):
                    offenders.append(f"{path}:{node.lineno}:{module}")

    assert offenders == [], "layer boundary violation (ARCH-1/INV-24): " + ", ".join(offenders[:20])
