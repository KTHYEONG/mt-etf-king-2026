from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

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

    def _legacy_path(self, gz_path: Path) -> Path:
        # gz_path is .../<YYYYMMDD>.json.gz -> legacy .../<YYYYMMDD>.json
        if gz_path.suffix == ".gz":
            return gz_path.with_suffix("")
        return gz_path

    def _rev_path(self, gz_path: Path, fetched_at: datetime) -> Path:
        ts = fetched_at.strftime("%Y%m%d%H%M%S%f")
        # canonical gz -> base without .json.gz
        if gz_path.suffixes == [".json", ".gz"] or gz_path.name.endswith(".json.gz"):
            base = gz_path.name[: -len(".json.gz")]
            rev_name = f"{base}.rev.{ts}.json.gz"
        else:
            # fallback for legacy plain path passed in
            base = gz_path.stem
            # preserve suffix (either .json or .json.gz)
            suffix = "".join(gz_path.suffixes) if gz_path.suffixes else gz_path.suffix
            if not suffix:
                suffix = ".json.gz"
            rev_name = f"{base}.rev.{ts}{suffix}"
        return gz_path.with_name(rev_name)

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
        content = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        if path.exists():
            if not allow_revision:
                return "skipped"
            rev_path = self._rev_path(path, record.fetched_at)
            counter = 1
            while rev_path.exists():
                # add counter before suffix
                if rev_path.name.endswith(".json.gz"):
                    base_no_suffix = rev_path.name[: -len(".json.gz")]
                    rev_path = rev_path.with_name(f"{base_no_suffix}.{counter}.json.gz")
                else:
                    rev_path = rev_path.with_name(f"{rev_path.stem}.{counter}{rev_path.suffix}")
                counter += 1
            rev_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(rev_path, "wt", encoding="utf-8") as f:
                f.write(content)
            return "revised"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
        return "written"

    def _read_json_from_path(self, p: Path) -> dict[str, Any]:
        if p.suffix == ".gz" or p.name.endswith(".json.gz"):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                return json.load(f)  # type: ignore[no-any-return]
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    def read(self, endpoint: str, bas_dd: date) -> BronzeRecord:
        gz_path = self._paths.bronze(endpoint, bas_dd)
        legacy_path = self._legacy_path(gz_path)
        if gz_path.exists():
            data = self._read_json_from_path(gz_path)
        elif legacy_path.exists():
            data = self._read_json_from_path(legacy_path)
        else:
            # fallback: try legacy plain explicitly in case DataPaths changed but file still plain
            # Also attempt to open gz_path to raise proper error
            with gz_path.open("r", encoding="utf-8") as f:  # will raise FileNotFoundError
                data = json.load(f)
            # unreachable
        bas_dd_str = str(data["bas_dd"])
        try:
            parsed_bas = datetime.strptime(bas_dd_str, "%Y%m%d").date()
        except ValueError:
            parsed_bas = date.fromisoformat(bas_dd_str)
        fetched_at_str = str(data["fetched_at"])
        try:
            fetched_at = datetime.fromisoformat(fetched_at_str)
        except ValueError:
            fetched_at = datetime.strptime(fetched_at_str, "%Y-%m-%dT%H:%M:%S%z")
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        return BronzeRecord(
            endpoint=str(data["endpoint"]),
            bas_dd=parsed_bas,
            fetched_at=fetched_at,
            http_status=int(data["http_status"]),
            row_count=int(data["row_count"]),
            rows=list(data["rows"]),
        )

    def has_session(self, endpoint: str, bas_dd: date) -> bool:
        gz_path = self._paths.bronze(endpoint, bas_dd)
        legacy_path = self._legacy_path(gz_path)
        return gz_path.exists() or legacy_path.exists()

    def available_sessions(self, endpoint: str) -> list[date]:
        root = self._paths.root
        base = root / "raw" / "krx" / endpoint
        if not base.exists():
            return []
        results: set[date] = set()
        # discover both *.json and *.json.gz
        for p in base.rglob("*.json"):
            if ".rev." in p.name:
                continue
            # handle .json.gz: rglob *.json will also match .json.gz because it ends with .json? Actually *.json pattern may match .json.gz? In Path.rglob, *.json matches files ending with .json but not .json.gz? Need to also glob *.json.gz
            # Instead handle filtering after: if name endswith .json.gz, strip .json.gz and parse
            # For plain .json files, parse stem
            # For .json.gz files, the path's name ends with .json.gz but rglob *.json might not match it depending on OS glob behavior.
            # Safer to also explicitly glob *.json.gz in separate loop below; here handle plain.
            if p.name.endswith(".json.gz"):  # noqa: SIM108
                stem = p.name[: -len(".json.gz")]  # noqa: SIM108
            else:  # noqa: SIM108
                # plain .json
                # p.suffixes: [.json] or maybe has .rev already excluded
                stem = p.stem  # noqa: SIM108
            if len(stem) != 8 or not stem.isdigit():
                continue
            try:
                d = datetime.strptime(stem, "%Y%m%d").date()
            except ValueError:
                continue
            results.add(d)
        # additionally handle *.json.gz files that were not matched by *.json glob (on some systems)
        for p in base.rglob("*.json.gz"):
            if ".rev." in p.name:
                continue
            stem = p.name[: -len(".json.gz")]
            if len(stem) != 8 or not stem.isdigit():
                continue
            try:
                d = datetime.strptime(stem, "%Y%m%d").date()
            except ValueError:
                continue
            results.add(d)
        # dedupe already via set, sorted ascending
        return sorted(results)

    def migrate_plain_to_gzip(self, endpoint: str, *, delete_plain: bool = True) -> dict[str, int]:
        root = self._paths.root
        base = root / "raw" / "krx" / endpoint
        counts = {"migrated": 0, "skipped_existing_gz": 0, "failed": 0, "deleted_plain": 0}
        if not base.exists():
            return counts
        # Find legacy plain files (*.json without .gz) excluding .rev.
        plain_files: list[Path] = []
        for p in base.rglob("*.json"):
            if ".rev." in p.name:
                continue
            if p.name.endswith(".json.gz"):
                continue
            # exclude if it's actually a gz file discovered via previous handling? Ensure plain not gz
            if p.suffixes == [".json", ".gz"]:
                continue
            # ensure plain .json (stem is 8 digits)
            stem = p.stem
            if len(stem) != 8 or not stem.isdigit():
                continue
            plain_files.append(p)
        for plain in plain_files:
            # derive date and gz path
            stem = plain.stem
            try:
                bas_dd = datetime.strptime(stem, "%Y%m%d").date()
            except ValueError:
                continue
            gz_path = self._paths.bronze(endpoint, bas_dd)
            if gz_path.exists():
                counts["skipped_existing_gz"] += 1
                continue
            # read plain
            try:
                with plain.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                counts["failed"] += 1
                continue
            # write gz: compact
            try:
                content = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                gz_path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(gz_path, "wt", encoding="utf-8") as f:
                    f.write(content)
            except Exception:
                counts["failed"] += 1
                continue
            # verify round-trip row_count and rows
            try:
                with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                    gz_data = json.load(f)
                # compare row_count and rows payload
                plain_row_count = int(data.get("row_count", -1))
                gz_row_count = int(gz_data.get("row_count", -2))
                plain_rows = data.get("rows")
                gz_rows = gz_data.get("rows")
                if plain_row_count != gz_row_count or plain_rows != gz_rows:
                    raise ValueError("row_count/rows mismatch")
            except Exception:  # noqa: S110
                # remove partially written gz? Keep plain, increment failed, delete gz if exists?
                try:
                    if gz_path.exists():
                        gz_path.unlink()
                except Exception:  # noqa: S110
                    pass  # noqa: S110
                counts["failed"] += 1
                continue
            counts["migrated"] += 1
            if delete_plain:
                try:
                    plain.unlink()
                    counts["deleted_plain"] += 1
                except Exception:  # noqa: S110
                    # if delete fails, not counted
                    pass  # noqa: S110
        return counts
