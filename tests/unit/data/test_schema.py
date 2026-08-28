from __future__ import annotations

from datetime import date

import pytest

from src.data.schema import ETF_DAILY_SCHEMA, decode_bas_dd, decode_optional_float, decode_optional_int


def test_scenario_03_01_decode_optional() -> None:
    """SCENARIO-03-01"""
    assert decode_optional_float("", "NAV") is None
    assert decode_optional_float("34970", "TDD_CLSPRC") == 34970.0
    assert decode_optional_float("1,234.5", "MKTCAP") == 1234.5
    with pytest.raises(ValueError, match="NAV"):
        decode_optional_float("-", "NAV")
    assert decode_optional_int("", "LIST_SHRS") is None
    with pytest.raises(ValueError, match="LIST_SHRS"):
        decode_optional_int("34970.0", "LIST_SHRS")
    assert decode_bas_dd("20260827") == date(2026, 8, 27)


def test_scenario_03_02_etf_schema_decode() -> None:
    """SCENARIO-03-02"""
    row = {
        "BAS_DD": "20260827",
        "ISU_CD": "451060",
        "ISU_NM": "1Q 200액티브",
        "TDD_CLSPRC": "34970",
        "NAV": "35057.03",
        "TDD_OPNPRC": "35535",
        "TDD_HGPRC": "35535",
        "TDD_LWPRC": "34660",
        "ACC_TRDVOL": "173366",
        "ACC_TRDVAL": "6064924927",
        "MKTCAP": "440622000000",
        "INVSTASST_NETASST_TOTAMT": "441718559589",
        "LIST_SHRS": "12600000",
        "IDX_IND_NM": "코스피 200",
        "OBJ_STKPRC_IDX": "1088.61",
        "CMPPREVDD_PRC": "460",
        "FLUC_RT": "1.33",
        "CMPPREVDD_IDX": "17.45",
        "FLUC_RT_IDX": "1.63",
    }
    decoded = ETF_DAILY_SCHEMA.decode_rows([row])
    assert decoded[0]["ticker"] == "451060"
    assert decoded[0]["close"] == 34970.0
    assert decoded[0]["shares_outstanding"] == 12600000
    assert decoded[0]["underlying_index_name"] == "코스피 200"
    assert decoded[0]["market_cap"] == decoded[0]["shares_outstanding"] * decoded[0]["close"]  # type: ignore[operator]
    # ticker alphanum
    row2 = dict(row)
    row2["ISU_CD"] = "0131A0"
    assert ETF_DAILY_SCHEMA.decode_rows([row2])[0]["ticker"] == "0131A0"
