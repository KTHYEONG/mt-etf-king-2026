from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path

    def _guard(self, candidate: Path) -> Path:
        root_resolved = self.root.resolve()
        cand_resolved = candidate.resolve()
        try:
            cand_resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"path escapes root {self.root}: {candidate}") from exc
        return candidate

    def _check_part(self, raw: str) -> None:
        p = Path(raw)
        if p.is_absolute():
            raise ValueError(f"absolute path not allowed: {raw}")
        if ".." in p.parts:
            raise ValueError(f"path traversal not allowed: {raw}")

    def bronze(self, endpoint: str, bas_dd: date) -> Path:
        self._check_part(endpoint)
        rel = Path("raw/krx") / endpoint / str(bas_dd.year) / f"{bas_dd:%Y%m%d}.json.gz"
        candidate = self.root / rel
        return self._guard(candidate)

    def results(self, run_id: str) -> Path:
        self._check_part(run_id)
        rel = Path("results") / run_id
        candidate = self.root / rel
        return self._guard(candidate)

    def silver(self, table: str) -> Path:
        self._check_part(table)
        rel = Path("normalized") / f"{table}.parquet"
        candidate = self.root / rel
        return self._guard(candidate)

    def gold(self, table: str) -> Path:
        self._check_part(table)
        rel = Path("features") / f"{table}.parquet"
        candidate = self.root / rel
        return self._guard(candidate)

    def state(self, name: str) -> Path:
        self._check_part(name)
        rel = Path("state") / f"{name}.json"
        candidate = self.root / rel
        return self._guard(candidate)
