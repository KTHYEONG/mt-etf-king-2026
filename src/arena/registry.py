"""Append-only trial registry that counts configurations per strategy family for multiplicity haircuts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy import typing as npt

from src.arena.evaluation import EvalConfig, EvaluationReport, evaluate, report_to_dict
from src.core.locks import data_lock

LOCK_TIMEOUT_S: Final = 10.0
_REQUIRED_KEYS: Final = frozenset({"recorded_at", "strategy_id", "strategy_family", "config_hash", "params", "report"})
_UTC_TIMESTAMP_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"


class RegistryCorruptError(RuntimeError):
    """Raised when a registry line is not a valid trial record; the registry is never silently skipped."""


def config_digest(params: Mapping[str, Any]) -> str:
    """Return a 16-hex-char sha256 digest of the canonical JSON form of `params`."""
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RegistryCorruptError(f"registry line {lineno} is not valid JSON: {path}") from exc
            if (
                not isinstance(record, dict)
                or not record.keys() >= _REQUIRED_KEYS
                or not isinstance(record["strategy_family"], str)
                or not isinstance(record["config_hash"], str)
            ):
                raise RegistryCorruptError(f"registry line {lineno} is missing required trial fields: {path}")
            records.append(record)
    return records


def _family_hashes(path: Path, strategy_family: str) -> set[str]:
    return {r["config_hash"] for r in _read_records(path) if r["strategy_family"] == strategy_family}


def count_trials(path: Path, strategy_family: str) -> int:
    """Return the number of distinct config hashes registered for `strategy_family`; a missing file counts as 0."""
    return len(_family_hashes(path, strategy_family))


def _record_line(
    report: EvaluationReport, params: Mapping[str, Any], strategy_family: str, recorded_at: datetime
) -> str:
    if recorded_at.utcoffset() is None:
        raise ValueError("recorded_at must be timezone-aware")
    if not strategy_family:
        raise ValueError("strategy_family must be non-empty")
    record = {
        "recorded_at": recorded_at.astimezone(UTC).strftime(_UTC_TIMESTAMP_FORMAT),
        "strategy_id": report.strategy_id,
        "strategy_family": strategy_family,
        "config_hash": report.config_hash,
        "params": params,
        "report": report_to_dict(report),
    }
    return json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with data_lock(path.with_suffix(path.suffix + ".lock"), shared=False, timeout_s=LOCK_TIMEOUT_S):
        yield


def _write_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def append_trial(
    path: Path,
    report: EvaluationReport,
    params: Mapping[str, Any],
    *,
    strategy_family: str,
    recorded_at: datetime,
) -> None:
    """Append one trial record as a JSON line; raises ValueError for a naive timestamp or empty family."""
    line = _record_line(report, params, strategy_family, recorded_at)
    with _locked(path):
        _write_line(path, line)


def evaluate_and_register(
    path: Path,
    strategy_id: str,
    strategy_family: str,
    params: Mapping[str, Any],
    sessions: Sequence[date],
    net_returns: npt.NDArray[np.float64],
    costs: npt.NDArray[np.float64],
    cfg: EvalConfig,
    *,
    recorded_at: datetime,
) -> EvaluationReport:
    """Count, evaluate and append under one exclusive lock so concurrent runs cannot undercount trials."""
    digest = config_digest(params)
    with _locked(path):
        known = _family_hashes(path, strategy_family)
        n_trials = len(known) + (0 if digest in known else 1)
        report = evaluate(strategy_id, digest, sessions, net_returns, costs, cfg, n_trials)
        _write_line(path, _record_line(report, params, strategy_family, recorded_at))
    return report
