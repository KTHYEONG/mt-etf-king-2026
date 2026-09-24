"""Immutable archive and typed standings for the Money Today contest leaderboard.

The public ranking JSON files are overwritten every trading day at 16:00 KST,
so each snapshot must be fetched and archived under its ``baseDt`` directory.
This module is a pure data layer: network I/O lives only in
:func:`fetch_leaderboard_payloads`, the archive is append-only, and no
decision logic is included here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import httpx

logger = logging.getLogger(__name__)

_USER_AGENT: Final[str] = "mt-etf-king-2026 contest-archive/1.0"
_TOTAL_ENDPOINT: Final[str] = "etfRankTotal"
_PURCHASES_ENDPOINT: Final[str] = "etfRankProductPurchase"
_UNINFORMATIVE_MOVE_PCT: Final[float] = 0.3


class LeaderboardFetchError(RuntimeError):
    """Raised when any leaderboard endpoint fails (HTTP error, timeout, non-JSON)."""


class LeaderboardSchemaError(RuntimeError):
    """Raised when the ``etfRankTotal`` payload lacks baseDt/data or valid rows."""


class LeaderboardConflictError(RuntimeError):
    """Raised when a baseDt directory exists with different ``etfRankTotal`` content."""


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    rank: int
    user_name: str
    total_return_pct: float
    daily_return_pct: float


@dataclass(frozen=True, slots=True)
class LeaderboardSnapshot:
    """One archived 16:00 KST snapshot of the public contest ranking (top-50, cumulative and daily %)."""

    base_date: date
    requested_at: str
    entries: tuple[LeaderboardEntry, ...]
    purchases: dict[str, tuple[tuple[str, float, float], ...]]

    def entry_for(self, user_name: str) -> LeaderboardEntry | None:
        """Return the entry whose userName equals `user_name` exactly, or None when outside the visible top-50."""
        for entry in self.entries:
            if entry.user_name == user_name:
                return entry
        return None


def fetch_leaderboard_payloads(
    base_url: str, endpoints: Sequence[str], timeout_s: float
) -> dict[str, dict[str, Any]]:
    """Download every leaderboard JSON endpoint once.

    The ranking page is rebuilt from static JSON files overwritten every trading day at 16:00 KST, so the only way to
    retain history is to fetch and archive each day.

    Raises:
        LeaderboardFetchError: any endpoint fails (HTTP error, timeout, non-JSON body). Fail-closed: no partial dict.
    """
    base = base_url.rstrip("/")
    try:
        with httpx.Client(timeout=timeout_s, headers={"User-Agent": _USER_AGENT}) as client:
            payloads: dict[str, dict[str, Any]] = {}
            for endpoint in endpoints:
                url = f"{base}/{endpoint}.json"
                try:
                    response = client.get(url)
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    raise LeaderboardFetchError(f"fetch failed endpoint={endpoint} error={exc!r}") from exc
                try:
                    body = response.json()
                except ValueError as exc:
                    raise LeaderboardFetchError(f"non-JSON body endpoint={endpoint}") from exc
                if not isinstance(body, dict):
                    raise LeaderboardFetchError(f"non-object body endpoint={endpoint}")
                payloads[endpoint] = body
            return payloads
    except LeaderboardFetchError:
        raise


def _parse_base_date(raw: Any) -> date:
    if not isinstance(raw, str) or len(raw) != 8 or not raw.isdigit():
        raise LeaderboardSchemaError("etfRankTotal lacks a valid baseDt (YYYYMMDD)")
    try:
        return date(int(raw[0:4]), int(raw[4:6]), int(raw[6:8]))
    except ValueError as exc:
        raise LeaderboardSchemaError(f"invalid baseDt: {raw!r}") from exc


def _validate_total_rows(data: Any) -> list[Mapping[str, Any]]:
    if not isinstance(data, list) or not data:
        raise LeaderboardSchemaError("etfRankTotal lacks baseDt/data")
    rows: list[Mapping[str, Any]] = []
    for row in data:
        if not isinstance(row, Mapping):
            raise LeaderboardSchemaError("etfRankTotal row is not an object")
        for key in ("rank", "userName", "totalReturnRate", "dailyReturnRate"):
            if key not in row:
                raise LeaderboardSchemaError(f"etfRankTotal row lacks {key}")
        try:
            int(row["rank"])
            str(row["userName"])
            float(row["totalReturnRate"])
            float(row["dailyReturnRate"])
        except (TypeError, ValueError) as exc:
            raise LeaderboardSchemaError(f"etfRankTotal row has mistyped fields: {row!r}") from exc
        rows.append(row)
    return rows


def _serialize(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def archive_leaderboard(
    payloads: Mapping[str, Mapping[str, Any]], archive_root: Path
) -> tuple[date, bool]:
    """Persist payloads under `<archive_root>/<baseDt:%Y%m%d>/<endpoint>.json`.

    Returns:
        (base_date, written). `written` is False when that baseDt directory already exists with identical content
        (idempotent re-run / holiday re-fetch of an old snapshot).

    Raises:
        LeaderboardSchemaError: `etfRankTotal` lacks baseDt/data or rows lack rank/userName/totalReturnRate/dailyReturnRate.
        LeaderboardConflictError: the baseDt directory exists with different etfRankTotal content (never overwrite).
    """
    total = payloads.get(_TOTAL_ENDPOINT)
    if not isinstance(total, Mapping):
        raise LeaderboardSchemaError("etfRankTotal payload missing")
    base_date = _parse_base_date(total.get("baseDt"))
    data = total.get("data")
    _validate_total_rows(data)

    base_name = base_date.strftime("%Y%m%d")
    target_dir = archive_root / base_name
    serialized = {endpoint: _serialize(dict(body)) for endpoint, body in payloads.items()}

    if target_dir.exists():
        existing_total_path = target_dir / f"{_TOTAL_ENDPOINT}.json"
        if not existing_total_path.is_file():
            raise LeaderboardConflictError(f"baseDt dir exists without etfRankTotal: {target_dir}")
        existing_total = _read_json_file(existing_total_path)
        if existing_total != dict(total):
            raise LeaderboardConflictError(f"conflicting etfRankTotal for baseDt={base_name} (never overwrite)")
        for endpoint, text in serialized.items():
            existing_path = target_dir / f"{endpoint}.json"
            if not existing_path.is_file() or existing_path.read_text(encoding="utf-8") != text:
                raise LeaderboardConflictError(f"conflicting {endpoint} for baseDt={base_name} (never overwrite)")
        logger.info(f"[DATA] contest_archive base_date={base_date.isoformat()} entries={len(data)} written=False")  # type: ignore[arg-type]
        return base_date, False

    target_dir.mkdir(parents=True, exist_ok=False)
    for endpoint, text in serialized.items():
        (target_dir / f"{endpoint}.json").write_text(text, encoding="utf-8")
    logger.info(f"[DATA] contest_archive base_date={base_date.isoformat()} entries={len(data)} written=True")  # type: ignore[arg-type]
    return base_date, True


def _parse_entries(data: Any) -> tuple[LeaderboardEntry, ...]:
    rows = _validate_total_rows(data)
    entries = tuple(
        LeaderboardEntry(
            rank=int(row["rank"]),
            user_name=str(row["userName"]),
            total_return_pct=float(row["totalReturnRate"]),
            daily_return_pct=float(row["dailyReturnRate"]),
        )
        for row in rows
    )
    return entries


def _parse_purchases(payload: Any) -> dict[str, tuple[tuple[str, float, float], ...]]:
    if not isinstance(payload, Mapping):
        return {}
    raw: Any = payload.get("data", payload)
    if isinstance(raw, list):
        divisions: Mapping[str, Any] = {"all": raw}
    elif isinstance(raw, Mapping):
        divisions = {k: v for k, v in raw.items() if isinstance(v, list)}
        if not divisions:
            return {}
    else:
        return {}
    out: dict[str, tuple[tuple[str, float, float], ...]] = {}
    for division, rows in divisions.items():
        parsed: list[tuple[str, float, float]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                name = str(row["productName"])
                purchase = float(row["purchaseCount"])
                change = float(row["priceChangeRate"])
            except (KeyError, TypeError, ValueError):
                continue
            parsed.append((name, purchase, change))
        out[str(division)] = tuple(parsed)
    return out


def load_snapshot(archive_root: Path, base_date: date) -> LeaderboardSnapshot:
    """Load one archived snapshot. Raises FileNotFoundError when that baseDt was never archived."""
    target_dir = archive_root / base_date.strftime("%Y%m%d")
    total_path = target_dir / f"{_TOTAL_ENDPOINT}.json"
    if not total_path.is_file():
        raise FileNotFoundError(f"no archived snapshot for base_date={base_date.isoformat()}")
    total = _read_json_file(total_path)
    if not isinstance(total, dict):
        raise LeaderboardSchemaError("archived etfRankTotal is not an object")
    entries = _parse_entries(total.get("data"))
    requested_at = str(total.get("reqDateTime", total.get("requestedAt", "")))
    purchases_path = target_dir / f"{_PURCHASES_ENDPOINT}.json"
    purchases: dict[str, tuple[tuple[str, float, float], ...]] = {}
    if purchases_path.is_file():
        purchases = _parse_purchases(_read_json_file(purchases_path))
    return LeaderboardSnapshot(
        base_date=base_date,
        requested_at=requested_at,
        entries=entries,
        purchases=purchases,
    )


def latest_snapshot_on_or_before(archive_root: Path, session: date) -> LeaderboardSnapshot | None:
    """Most recent archived snapshot with base_date <= session, or None."""
    if not archive_root.is_dir():
        return None
    best: date | None = None
    for child in archive_root.iterdir():
        if not child.is_dir() or len(child.name) != 8 or not child.name.isdigit():
            continue
        try:
            candidate = date(int(child.name[0:4]), int(child.name[4:6]), int(child.name[6:8]))
        except ValueError:
            continue
        if (
            candidate <= session
            and (best is None or candidate > best)
            and (child / f"{_TOTAL_ENDPOINT}.json").is_file()
        ):
            best = candidate
    if best is None:
        return None
    return load_snapshot(archive_root, best)


def latest_entry_on_or_before(archive_root: Path, user_name: str, session: date) -> tuple[date, LeaderboardEntry] | None:
    """Most recent archived (base_date, entry) with base_date <= session in which `user_name` is visible, or None."""
    if not archive_root.is_dir():
        return None
    dates: list[date] = []
    for child in archive_root.iterdir():
        if not child.is_dir() or len(child.name) != 8 or not child.name.isdigit():
            continue
        try:
            candidate = date(int(child.name[0:4]), int(child.name[4:6]), int(child.name[6:8]))
        except ValueError:
            continue
        if candidate <= session and (child / f"{_TOTAL_ENDPOINT}.json").is_file():
            dates.append(candidate)
    for base_date in sorted(dates, reverse=True):
        entry = load_snapshot(archive_root, base_date).entry_for(user_name)
        if entry is not None:
            return base_date, entry
    return None


def infer_single_vehicle_holders(
    snapshot: LeaderboardSnapshot,
    vehicle_changes_pct: Mapping[str, float],
    tol_pct: float,
    min_weight: float,
    exposure_of: Mapping[str, str] | None = None,
) -> dict[str, tuple[str, float]]:
    """Map userName -> (exposure_alias, implied_weight) for entries whose daily return is explained by one exposure.

    `vehicle_changes_pct` keys are instrument keys (a vehicle alias or a wrapper key such as "HY2@0195S0").
    `exposure_of` maps each key to its exposure alias; a key missing from it (or None) is its own exposure. Several
    listed wrappers of one exposure (e.g. KODEX and TIGER SK Hynix 2x) differ by tracking/premium noise, so a match
    on any wrapper counts once for that exposure. Entries matching zero or two or more distinct exposures are omitted
    (ambiguous holdings are never guessed). When several wrappers of the single matched exposure match, the implied
    weight comes from the wrapper with the smallest absolute residual at its implied weight.
    """
    informative = {
        key: float(change)
        for key, change in vehicle_changes_pct.items()
        if abs(float(change)) >= _UNINFORMATIVE_MOVE_PCT
    }
    result: dict[str, tuple[str, float]] = {}
    for entry in snapshot.entries:
        daily = entry.daily_return_pct
        per_key: list[tuple[str, str, float, float]] = []
        for key, change in informative.items():
            w_star = daily / change
            if w_star < 0:
                continue
            lo = (daily - tol_pct) / change if change > 0 else (daily + tol_pct) / change
            hi = (daily + tol_pct) / change if change > 0 else (daily - tol_pct) / change
            if hi < min_weight or lo > 1.0:
                continue
            implied = min(1.0, max(min_weight, w_star))
            residual = abs(daily - implied * change)
            exposure = exposure_of[key] if exposure_of is not None and key in exposure_of else key
            per_key.append((key, exposure, implied, residual))
        by_exposure: dict[str, tuple[str, float, float]] = {}
        for key, exposure, implied, residual in per_key:
            prev = by_exposure.get(exposure)
            if prev is None or residual < prev[2]:
                by_exposure[exposure] = (key, implied, residual)
        if len(by_exposure) == 1:
            exposure = next(iter(by_exposure))
            _, implied, _ = by_exposure[exposure]
            result[entry.user_name] = (exposure, implied)
    return result
