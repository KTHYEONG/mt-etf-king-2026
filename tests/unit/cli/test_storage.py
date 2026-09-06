"""P4 CLI decomposition: storage.py moved verbatim from _impl.py."""
import argparse

import src.data.bronze as bronze_mod
import src.reporting.results as results_mod
from src.cli.commands.storage import cmd_storage_migrate


class _FakeStore:
    def __init__(self, paths) -> None:
        self._paths = paths
        self.migrate_plain_to_gzip = self._migrate

    def _migrate(self, endpoint, delete_plain=True):  # noqa: ANN001, ANN202
        return {"migrated": 0, "skipped_existing_gz": 0, "failed": 0, "deleted_plain": 0}


def test_p4_storage_migrate_fake_store_returns_0(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(bronze_mod, "BronzeStore", _FakeStore)
    monkeypatch.setattr(results_mod, "rebuild_runs_registry", lambda paths: 0)
    assert cmd_storage_migrate(argparse.Namespace(endpoint="x", delete_plain=False)) == 0
