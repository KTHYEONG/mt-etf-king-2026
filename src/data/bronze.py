from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime

from src.core.paths import DataPaths
from src.data.providers.base import RawRow


@dataclass(frozen=True)
class BronzeRecord:
    endpoint: str
    bas_dd: date
    fetched_at: datetime
    http_status: int
    row_count: int
    rows: list[RawRow]


class BronzeStore:
    def __init__(self, paths: DataPaths) -> None:
        self._paths = paths

    def write(self, record: BronzeRecord, allow_revision: bool = False) -> str:
        path = self._paths.bronze(record.endpoint, record.bas_dd)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "endpoint": record.endpoint,
            "bas_dd": record.bas_dd.strftime("%Y%m%d"),
            "fetched_at": record.fetched_at.isoformat(),
            "http_status": record.http_status,
            "row_count": record.row_count,
            "rows": record.rows,
        }
        content = json.dumps(envelope, ensure_ascii=False, indent=2)
        if path.exists():
            if not allow_revision:
                return "skipped"
            # Create sibling revision file, leave original untouched
            # Use timestamp to make unique
            ts = record.fetched_at.strftime("%Y%m%d%H%M%S%f")
            # Try to create revision file with increasing suffix
            rev_path = path.with_name(f"{path.stem}.rev.{ts}{path.suffix}")
            # If by chance already exists, add counter
            counter = 1
            while rev_path.exists():
                rev_path = path.with_name(f"{path.stem}.rev.{ts}.{counter}{path.suffix}")
                counter += 1
            rev_path.parent.mkdir(parents=True, exist_ok=True)
            with rev_path.open("w", encoding="utf-8") as f:
                f.write(content)
            return "revised"
        with path.open("w", encoding="utf-8") as f:
            f.write(content)
        return "written"

    def read(self, endpoint: str, bas_dd: date) -> BronzeRecord:
        path = self._paths.bronze(endpoint, bas_dd)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        bas_dd_str = data["bas_dd"]
        # Parse bas_dd: YYYYMMDD
        try:
            parsed_bas = datetime.strptime(bas_dd_str, "%Y%m%d").date()
        except ValueError:
            parsed_bas = date.fromisoformat(bas_dd_str)
        fetched_at_str = data["fetched_at"]
        try:
            fetched_at = datetime.fromisoformat(fetched_at_str)
        except ValueError:
            fetched_at = datetime.strptime(fetched_at_str, "%Y-%m-%dT%H:%M:%S%z")
        # Ensure timezone aware; if naive, assume UTC
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        return BronzeRecord(
            endpoint=data["endpoint"],
            bas_dd=parsed_bas,
            fetched_at=fetched_at,
            http_status=int(data["http_status"]),
            row_count=int(data["row_count"]),
            rows=list(data["rows"]),
        )

    def has_session(self, endpoint: str, bas_dd: date) -> bool:
        path = self._paths.bronze(endpoint, bas_dd)
        return path.exists()

    def available_sessions(self, endpoint: str) -> list[date]:
        # Scan for files under raw/krx/<endpoint>/**/*.json excluding revisions
        # DataPaths.bronze gives pattern root/raw/krx/endpoint/YYYY/file.json
        # We need to discover all primary files for this endpoint.
        # Use the root to glob.
        root = self._paths.root
        base = root / "raw" / "krx" / endpoint
        if not base.exists():
            return []
        results: list[date] = []
        for p in base.rglob("*.json"):
            # Exclude revision siblings: if stem contains ".rev"
            if ".rev" in p.name:
                continue
            # Expect filename like YYYYMMDD.json, extract date
            stem = p.stem
            # stem should be 8 digits
            if len(stem) != 8 or not stem.isdigit():
                continue
            try:
                d = datetime.strptime(stem, "%Y%m%d").date()
            except ValueError:
                continue
            # Only count if file is at expected year directory? But also include any
            results.append(d)
        results.sort()
        return results
