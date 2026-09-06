import json
from pathlib import Path


def test_migrate_runs_registry_ids_dry_run_reports_counts(tmp_path: Path) -> None:
    from tools.migrate_runs_registry_ids import migrate_runs_registry_ids

    registry = tmp_path / "runs_registry.jsonl"
    registry.write_text(
        json.dumps({"run_id": "r1", "model": "P27"}) + "\n" + json.dumps({"run_id": "r2", "model": "B0"}) + "\n",
        encoding="utf-8",
    )
    before = registry.read_text(encoding="utf-8")

    counts = migrate_runs_registry_ids(registry, dry_run=True)

    assert counts == {"P27": 1, "B0": 1}
    assert registry.read_text(encoding="utf-8") == before

    counts2 = migrate_runs_registry_ids(registry, dry_run=False)
    assert counts2 == {"P27": 1, "B0": 1}
    rows = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["model"] for r in rows] == ["sticky.mom60_raw", "baseline.buy_hold"]
    assert registry.with_suffix(registry.suffix + ".bak").exists()
