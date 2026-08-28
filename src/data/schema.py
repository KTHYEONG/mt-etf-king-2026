from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal

from src.data.providers.base import RawRow


def decode_optional_float(raw: str, field: str) -> float | None:
    if raw == "":
        return None
    # Remove commas and strip
    cleaned = raw.replace(",", "").strip()
    if cleaned == "":
        raise ValueError(f"field {field}: empty after cleaning raw={raw!r}")
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"field {field}: cannot parse {raw!r} as float") from exc


def decode_optional_int(raw: str, field: str) -> int | None:
    if raw == "":
        return None
    cleaned = raw.replace(",", "").strip()
    if cleaned == "":
        raise ValueError(f"field {field}: empty after cleaning raw={raw!r}")
    # Reject fractional part
    if "." in cleaned:
        raise ValueError(f"field {field}: fractional value {raw!r} in int field")
    # Also reject exponent? but int() would handle
    try:
        # Use int with base 10, reject floats
        if not re.fullmatch(r"-?\d+", cleaned):
            raise ValueError()
        return int(cleaned)
    except ValueError as exc:
        raise ValueError(f"field {field}: cannot parse {raw!r} as int") from exc


def decode_bas_dd(raw: str) -> date:
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"invalid BAS_DD {raw!r}") from exc


@dataclass(frozen=True)
class ColumnSpec:
    source_field: str
    column: str
    dtype: Literal["date", "str", "int", "float"]
    required: bool = False


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    endpoint: str
    columns: tuple[ColumnSpec, ...]
    key: tuple[str, ...]

    def decode_rows(self, rows: Sequence[RawRow]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        ticker_pattern = re.compile(r"^[0-9A-Z]{6}$")
        for r in rows:
            decoded: dict[str, object] = {}
            for col in self.columns:
                raw = r.get(col.source_field, "")
                # Ensure raw is str
                if not isinstance(raw, str):
                    raw = str(raw)
                if col.dtype == "date":
                    if raw == "":
                        if col.required:
                            raise ValueError(f"field {col.source_field}: required date missing")
                        decoded[col.column] = None
                    else:
                        decoded[col.column] = decode_bas_dd(raw)
                elif col.dtype == "str":
                    # Keep as string (including empty string becomes None? but spec says string fields keep value)
                    # For ticker, validate pattern if non-empty
                    decoded[col.column] = raw if raw != "" else None if not col.required else raw
                    # Actually for string nullable, keep None for empty? But for name etc empty may be allowed
                    # Keep original string for non-ticker; for ticker allow alphanum
                    if col.column == "ticker" and decoded[col.column] is not None:
                        val = decoded[col.column]
                        if isinstance(val, str) and val != "" and not ticker_pattern.match(val):  # noqa: SIM102
                            raise ValueError(f"field {col.source_field}: invalid ticker {val!r}")
                    # For other str fields, keep raw
                    if col.column != "ticker":
                        # Ensure we store raw string or None
                        decoded[col.column] = raw if raw != "" else None
                        # But for required str like name? Keep None if empty? Tests may expect None?
                        # For underlying_index_name empty maybe None?
                        pass
                elif col.dtype == "float":
                    decoded[col.column] = decode_optional_float(raw, col.source_field)
                elif col.dtype == "int":
                    decoded[col.column] = decode_optional_int(raw, col.source_field)
                else:
                    decoded[col.column] = raw
            out.append(decoded)
        return out


ETF_DAILY_SCHEMA: Final[DatasetSchema] = DatasetSchema(
    name="etf_daily",
    endpoint="etp/etf_bydd_trd",
    columns=(
        ColumnSpec(source_field="BAS_DD", column="date", dtype="date", required=True),
        ColumnSpec(source_field="ISU_CD", column="ticker", dtype="str", required=True),
        ColumnSpec(source_field="ISU_NM", column="name", dtype="str", required=True),
        ColumnSpec(source_field="TDD_CLSPRC", column="close", dtype="float"),
        ColumnSpec(source_field="TDD_OPNPRC", column="open", dtype="float"),
        ColumnSpec(source_field="TDD_HGPRC", column="high", dtype="float"),
        ColumnSpec(source_field="TDD_LWPRC", column="low", dtype="float"),
        ColumnSpec(source_field="ACC_TRDVOL", column="volume", dtype="int"),
        ColumnSpec(source_field="ACC_TRDVAL", column="trading_value", dtype="int"),
        ColumnSpec(source_field="NAV", column="nav", dtype="float"),
        ColumnSpec(source_field="MKTCAP", column="market_cap", dtype="int"),
        ColumnSpec(source_field="INVSTASST_NETASST_TOTAMT", column="net_assets", dtype="int"),
        ColumnSpec(source_field="LIST_SHRS", column="shares_outstanding", dtype="int"),
        ColumnSpec(source_field="IDX_IND_NM", column="underlying_index_name", dtype="str"),
        ColumnSpec(source_field="OBJ_STKPRC_IDX", column="underlying_index_close", dtype="float"),
    ),
    key=("date", "ticker"),
)

INDEX_DAILY_SCHEMA: Final[DatasetSchema] = DatasetSchema(
    name="index_daily",
    endpoint="idx/kospi_dd_trd",
    columns=(
        ColumnSpec(source_field="BAS_DD", column="date", dtype="date", required=True),
        ColumnSpec(source_field="IDX_CLSS", column="index_class", dtype="str", required=True),
        ColumnSpec(source_field="IDX_NM", column="index_name", dtype="str", required=True),
        ColumnSpec(source_field="CLSPRC_IDX", column="close", dtype="float"),
        ColumnSpec(source_field="OPNPRC_IDX", column="open", dtype="float"),
        ColumnSpec(source_field="HGPRC_IDX", column="high", dtype="float"),
        ColumnSpec(source_field="LWPRC_IDX", column="low", dtype="float"),
        ColumnSpec(source_field="ACC_TRDVOL", column="volume", dtype="int"),
        ColumnSpec(source_field="ACC_TRDVAL", column="trading_value", dtype="int"),
        ColumnSpec(source_field="MKTCAP", column="market_cap", dtype="int"),
    ),
    key=("date", "index_name"),
)

STOCK_DAILY_SCHEMA: Final[DatasetSchema] = DatasetSchema(
    name="stock_daily",
    endpoint="sto/stk_bydd_trd",
    columns=(
        ColumnSpec(source_field="BAS_DD", column="date", dtype="date", required=True),
        ColumnSpec(source_field="ISU_CD", column="ticker", dtype="str", required=True),
        ColumnSpec(source_field="ISU_NM", column="name", dtype="str"),
        ColumnSpec(source_field="MKT_TP_NM", column="market", dtype="str"),
        ColumnSpec(source_field="SECT_TP_NM", column="sector_type", dtype="str"),
        ColumnSpec(source_field="TDD_CLSPRC", column="close", dtype="float"),
        ColumnSpec(source_field="TDD_OPNPRC", column="open", dtype="float"),
        ColumnSpec(source_field="TDD_HGPRC", column="high", dtype="float"),
        ColumnSpec(source_field="TDD_LWPRC", column="low", dtype="float"),
        ColumnSpec(source_field="ACC_TRDVOL", column="volume", dtype="int"),
        ColumnSpec(source_field="ACC_TRDVAL", column="trading_value", dtype="int"),
        ColumnSpec(source_field="MKTCAP", column="market_cap", dtype="int"),
        ColumnSpec(source_field="LIST_SHRS", column="shares_outstanding", dtype="int"),
    ),
    key=("date", "ticker"),
)
