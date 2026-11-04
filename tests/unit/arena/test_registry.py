import fcntl
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from src.arena import registry
from src.arena.evaluation import EvalConfig, EvaluationReport, evaluate
from src.arena.registry import (
    RegistryCorruptError,
    append_trial,
    config_digest,
    count_trials,
    evaluate_and_register,
)

CFG = EvalConfig(
    horizon_sessions=20,
    bootstrap_mean_block=10,
    bootstrap_draws=400,
    seed=7,
    ci_level=0.95,
    exceedance_grid=(1.1, 1.3, 1.5, 2.0, 2.3, 3.0),
    min_sessions=60,
)
NOW = datetime(2026, 11, 4, 9, 0, tzinfo=UTC)
SESSIONS = [date(2026, 1, 1) + timedelta(days=i) for i in range(100)]


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    costs = np.full(100, 0.001)
    net = np.random.default_rng(31).normal(0.001, 0.02, 100) - costs
    return net, costs


def _register(path: Path, family: str, params: dict[str, int]) -> int:
    net, costs = _inputs()
    report = evaluate_and_register(path, "s", family, params, SESSIONS, net, costs, CFG, recorded_at=NOW)
    return report.edge.n_trials


def test_digest_order_insensitive() -> None:
    assert config_digest({"a": 1, "b": 2}) == config_digest({"b": 2, "a": 1})
    assert config_digest({"a": 1, "b": 2}) != config_digest({"a": 1, "b": 3})
    assert len(config_digest({"a": 1})) == 16


def test_count_missing_file_is_zero(tmp_path: Path) -> None:
    assert count_trials(tmp_path / "missing.jsonl", "F") == 0


def test_append_then_count_distinct(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    net, costs = _inputs()
    report = evaluate("s", "a", SESSIONS, net, costs, CFG, 1)
    for digest, family in [("a", "F"), ("a", "F"), ("b", "F"), ("c", "G")]:
        rep = _with_hash(report, digest)
        append_trial(path, rep, {"h": digest}, strategy_family=family, recorded_at=NOW)
    assert count_trials(path, "F") == 2
    assert count_trials(path, "G") == 1


def _with_hash(report: EvaluationReport, digest: str) -> EvaluationReport:
    return replace(report, config_hash=digest)


def test_append_only_never_rewrites(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    net, costs = _inputs()
    report = evaluate("s", "a", SESSIONS, net, costs, CFG, 1)
    append_trial(path, report, {"k": 1}, strategy_family="F", recorded_at=NOW)
    before = path.read_bytes()
    append_trial(path, report, {"k": 2}, strategy_family="F", recorded_at=NOW)
    after = path.read_bytes()
    assert after.startswith(before)
    assert len(after.splitlines()) == 2


def test_corrupt_line_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    net, costs = _inputs()
    report = evaluate("s", "a", SESSIONS, net, costs, CFG, 1)
    append_trial(path, report, {"k": 1}, strategy_family="F", recorded_at=NOW)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("garbage\n")
    with pytest.raises(RegistryCorruptError, match="line 2"):
        count_trials(path, "F")


def test_missing_key_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text(json.dumps({"strategy_family": "F", "config_hash": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(RegistryCorruptError, match="line 1"):
        count_trials(path, "F")


def test_naive_timestamp_rejected(tmp_path: Path) -> None:
    net, costs = _inputs()
    report = evaluate("s", "a", SESSIONS, net, costs, CFG, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        append_trial(tmp_path / "r.jsonl", report, {}, strategy_family="F", recorded_at=datetime(2026, 1, 1))


def test_empty_family_rejected(tmp_path: Path) -> None:
    net, costs = _inputs()
    report = evaluate("s", "a", SESSIONS, net, costs, CFG, 1)
    with pytest.raises(ValueError, match="non-empty"):
        append_trial(tmp_path / "r.jsonl", report, {}, strategy_family="", recorded_at=NOW)
    assert not (tmp_path / "r.jsonl").exists()


def test_record_timestamp_is_utc_z(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    net, costs = _inputs()
    report = evaluate("s", "a", SESSIONS, net, costs, CFG, 1)
    local = datetime(2026, 11, 4, 18, 0, tzinfo=timezone(timedelta(hours=9)))
    append_trial(path, report, {}, strategy_family="F", recorded_at=local)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["recorded_at"] == "2026-11-04T09:00:00Z"
    assert record["strategy_family"] == "F"


def test_evaluate_and_register_counts_itself(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    assert _register(path, "F", {"c": 1}) == 1
    assert _register(path, "F", {"c": 2}) == 2
    assert _register(path, "F", {"c": 1}) == 2
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    assert registry.count_trials(path, "F") == 2


def test_register_counts_under_exclusive_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "registry.jsonl"
    lock_path = path.with_suffix(path.suffix + ".lock")
    original = registry._family_hashes
    observed: list[bool] = []

    def probe(p: Path, family: str) -> set[str]:
        with lock_path.open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                observed.append(True)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
                observed.append(False)
        return original(p, family)

    monkeypatch.setattr(registry, "_family_hashes", probe)
    _register(path, "F", {"c": 1})
    assert observed == [True]
