"""P4 CLI decomposition: data.py (ingest/normalize) moved verbatim from _impl.py."""

from __future__ import annotations

import argparse
from datetime import date, datetime, UTC
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli.commands.data import cmd_ingest, cmd_normalize
from src.core.paths import DataPaths


def test_p4_data_ingest_missing_args_returns_1() -> None:
    assert cmd_ingest(argparse.Namespace()) == 1


def test_p4_data_normalize_invalid_mode_returns_1() -> None:
    assert cmd_normalize(argparse.Namespace(dataset="etf_daily", mode="bogus")) == 1


def test_normalize_summary_reports_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A successful normalize logs the summary line with the written flag."""
    from src.data.bronze import BronzeRecord, BronzeStore

    store = BronzeStore(DataPaths(root=tmp_path))
    rows = [
        {
            "BAS_DD": "20260827",
            "ISU_CD": "451060",
            "ISU_NM": "Test",
            "TDD_CLSPRC": "35000",
            "NAV": "35000",
            "TDD_OPNPRC": "35000",
            "TDD_HGPRC": "35001",
            "TDD_LWPRC": "34999",
            "ACC_TRDVOL": "1000",
            "ACC_TRDVAL": "1000000",
            "MKTCAP": "35000000",
            "INVSTASST_NETASST_TOTAMT": "35000000",
            "LIST_SHRS": "1000",
            "IDX_IND_NM": "Index",
            "OBJ_STKPRC_IDX": "1000",
        }
    ]
    store.write(
        BronzeRecord(
            endpoint="etp/etf_bydd_trd",
            bas_dd=date(2026, 8, 27),
            fetched_at=datetime.now(UTC),
            http_status=200,
            row_count=len(rows),
            rows=rows,
        )
    )
    monkeypatch.setattr(
        "src.cli.commands.data.get_settings", lambda: SimpleNamespace(data_root=tmp_path)
    )
    with caplog.at_level("INFO"):
        assert cmd_normalize(argparse.Namespace(dataset="etf_daily", mode="full")) == 0
    assert any("written=True" in record.message for record in caplog.records)
