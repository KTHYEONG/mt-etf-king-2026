def test_module_line_budget_after_refactor() -> None:
    """R1 scope: new strategy/cli helper modules must stay under 600 lines."""
    from pathlib import Path

    per_path_caps = {
        "src/strategies/sticky/model.py": 700,
        "src/strategies/sticky/capacity.py": 200,
    }
    default_cap = 600
    scoped_roots = [
        Path("src/strategies"),
        Path("src/cli/constants.py"),
        Path("src/cli/main.py"),
        Path("src/cli/dispatch.py"),
        Path("src/cli/commands"),
    ]
    offenders: list[str] = []
    for root in scoped_roots:
        paths = [root] if root.is_file() else list(root.rglob("*.py"))
        for path in paths:
            if "__pycache__" in path.parts:
                continue
            rel = str(path)
            limit = per_path_caps.get(rel, default_cap)
            with path.open(encoding="utf-8") as fh:
                line_count = sum(1 for _ in fh)
            if line_count > limit:
                offenders.append(f"{rel}:{line_count}>{limit}")
    assert offenders == [], "modules exceed line budget: " + ", ".join(offenders)


def test_global_src_line_budget() -> None:
    from pathlib import Path

    pending_split = {
        "src/cli/_impl.py",
        "src/alpha/baselines.py",
        "src/alpha/sticky.py",
        "src/strategies/sticky/model.py",
        "src/execution/ledger.py",
        "src/reporting/tail_forensics.py",
        "src/tournament/distribution_core.py",
        "src/tournament/objective_impl.py",
        "src/tournament/overlay_returns.py",
        "src/backtest/engine.py",
        "src/portfolio/policy.py",
        "src/tournament/simulator.py",
    }
    offenders: list[str] = []
    for path in Path("src").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = str(path)
        if rel in pending_split:
            continue
        count = sum(1 for _ in path.open(encoding="utf-8"))
        if count > 600:
            offenders.append(f"{rel}:{count}")
    assert offenders == [], "modules exceed 600 lines: " + ", ".join(offenders)
