from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def resolve_family_key(index_key: str, overrides: Mapping[str, str] | None = None) -> str:
    if overrides is not None and index_key in overrides:
        return overrides[index_key]
    return index_key


@dataclass(frozen=True)
class LeverageFamilyMember:
    ticker: str
    name: str
    leverage_multiple: int
    index_key: str


@dataclass(frozen=True)
class LeverageFamily:
    family_key: str
    members: tuple[LeverageFamilyMember, ...]


class LeverageFamilyIndex:
    def __init__(self, families: Mapping[str, LeverageFamily]) -> None:
        self._families: dict[str, LeverageFamily] = dict(families)

    @classmethod
    def build(cls, attributes: Mapping[str, object]) -> LeverageFamilyIndex:
        # attributes is Mapping[str, InstrumentAttributes]
        grouped: dict[str, list[LeverageFamilyMember]] = {}
        for ticker, attr in attributes.items():
            # attr is InstrumentAttributes
            fk = attr.leverage_family_key  # type: ignore[attr-defined]
            nm = attr.name  # type: ignore[attr-defined]
            lev = attr.leverage_multiple  # type: ignore[attr-defined]
            ik = attr.index_key  # type: ignore[attr-defined]
            member = LeverageFamilyMember(ticker=ticker, name=nm, leverage_multiple=lev, index_key=ik)
            grouped.setdefault(fk, []).append(member)
        families: dict[str, LeverageFamily] = {}
        for fk, members in grouped.items():
            # sort by leverage_multiple then ticker for determinism
            sorted_members = sorted(members, key=lambda m: (m.leverage_multiple, m.ticker))
            families[fk] = LeverageFamily(family_key=fk, members=tuple(sorted_members))
        return cls(families)

    def get(self, family_key: str) -> LeverageFamily | None:
        return self._families.get(family_key)
