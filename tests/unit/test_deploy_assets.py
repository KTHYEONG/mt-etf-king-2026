"""Guards for systemd units and wrapper scripts (deploy hygiene)."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
SCRIPTS_DIR = REPO_ROOT / "scripts"

_WRAPPERS = (
    ("daily_pipeline.sh", "daily_refresh"),
    ("contest_archive.sh", "contest_archive"),
    ("contest_daily.sh", "contest_daily"),
    ("contest_weekly.sh", "contest_weekly"),
)


def _service_section(path: Path) -> dict[str, str]:
    section: dict[str, str] = {}
    in_service = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_service = stripped == "[Service]"
            continue
        if in_service and "=" in stripped and not stripped.startswith("#"):
            key, _, value = stripped.partition("=")
            section[key.strip()] = value.strip()
    return section


def test_every_service_is_niced() -> None:
    """Every timer service runs at lowered priority."""
    services = sorted(SYSTEMD_DIR.glob("mt-etf-*.service"))
    assert services, "expected mt-etf-*.service units"
    for service in services:
        section = _service_section(service)
        assert "Nice" in section, f"{service.name} lacks Nice="
        assert int(section["Nice"]) >= 10, f"{service.name} Nice={section['Nice']} is too eager"


def test_daily_refresh_is_cpu_capped() -> None:
    """The shared-host refresh job is capped at one vCPU with no PATH override."""
    text = (SYSTEMD_DIR / "mt-etf-daily-refresh.service").read_text(encoding="utf-8")
    section = _service_section(SYSTEMD_DIR / "mt-etf-daily-refresh.service")
    assert section.get("CPUQuota") == "100%"
    assert "Environment=PATH" not in text


def test_morning_slots_avoid_co_tenant_auction_window() -> None:
    """No timer slot starts inside the open-auction capture window, with tight accuracy."""
    text = (SYSTEMD_DIR / "mt-etf-daily-refresh.timer").read_text(encoding="utf-8")
    times = re.findall(r"OnCalendar=\S+\s+(\d{2}):(\d{2}):(\d{2})", text)
    assert times, "expected OnCalendar= entries"
    for hour, minute, second in times:
        stamp = f"{hour}:{minute}:{second}"
        assert not ("08:37:00" <= stamp <= "09:03:59"), f"slot {stamp} lands in the auction window"
    accuracy = re.search(r"AccuracySec=(\d+)(s|min|m)?", text)
    assert accuracy is not None, "expected AccuracySec="
    value, unit = int(accuracy.group(1)), accuracy.group(2) or "s"
    limit_s = value * 60 if unit in ("min", "m") else value
    assert limit_s <= 10, f"AccuracySec allows {limit_s}s of jitter"


def _stage_wrapper(tmp_path: Path, name: str) -> Path:
    staged = tmp_path / "scripts" / name
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPTS_DIR / name, staged)
    staged.chmod(staged.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return staged


def _stub_uv(tmp_path: Path, exit_code: int, record_args: bool = False) -> Path:
    stub = tmp_path / "uv-stub.sh"
    lines = ["#!/usr/bin/env bash"]
    if record_args:
        lines.append('echo "$@" >> "$ARGS_LOG"')
    lines.append(f"exit {exit_code}")
    stub.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _run_wrapper(script: Path, stub: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["UV_BIN"] = str(stub)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=60)  # noqa: S603, S607


def _kst_today() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%F")


def _wrapper_log(tmp_path: Path, prefix: str) -> Path:
    return tmp_path / "logs" / f"{prefix}_{_kst_today()}.log"


def test_wrappers_record_failure_status(tmp_path: Path) -> None:
    """A failing uv run still exits with its status and writes the end-status line."""
    stub = _stub_uv(tmp_path, 7)
    for name, prefix in _WRAPPERS:
        script = _stage_wrapper(tmp_path, name)
        result = _run_wrapper(script, stub)
        assert result.returncode == 7, f"{name} exited {result.returncode}"
        log = _wrapper_log(tmp_path, prefix)
        lines = log.read_text(encoding="utf-8").splitlines()
        assert any("end status=7" in line for line in lines), f"{name} log misses the end line"


def test_wrappers_record_success_status(tmp_path: Path) -> None:
    """A successful uv run exits 0 with both start and end-status lines."""
    stub = _stub_uv(tmp_path, 0)
    for name, prefix in _WRAPPERS:
        script = _stage_wrapper(tmp_path, name)
        result = _run_wrapper(script, stub)
        assert result.returncode == 0, f"{name} exited {result.returncode}"
        text = _wrapper_log(tmp_path, prefix).read_text(encoding="utf-8")
        assert "start" in text and "end status=0" in text, f"{name} log misses start/end lines"


def test_weekly_retry_forces_after_no_data(tmp_path: Path) -> None:
    """The weekly wrapper passes --force only when the latest card decided NO_DATA."""
    cards = tmp_path / "results" / "contest_weekly"
    cards.mkdir(parents=True)
    (cards / "2026-09-26.json").write_text('{"action": "NO_DATA"}\n', encoding="utf-8")
    args_log = tmp_path / "args.log"
    stub = _stub_uv(tmp_path, 0, record_args=True)
    script = _stage_wrapper(tmp_path, "contest_weekly.sh")

    result = _run_wrapper(script, stub, {"ARGS_LOG": str(args_log)})
    assert result.returncode == 0
    assert args_log.read_text(encoding="utf-8").split() == ["run", "mt-etf", "contest-weekly", "--force"]

    (cards / "2026-09-26.json").unlink()
    args_log.unlink()
    result = _run_wrapper(script, stub, {"ARGS_LOG": str(args_log)})
    assert result.returncode == 0
    assert args_log.read_text(encoding="utf-8").split() == ["run", "mt-etf", "contest-weekly"]
