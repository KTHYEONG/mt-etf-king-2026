from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

_SENTINEL = "__UNKNOWN_INDEX__"


def normalize_index_key(raw: str | None) -> str:
    if raw is None:
        return _SENTINEL
    if not isinstance(raw, str):
        raw = str(raw)
    if raw.strip() == "":
        return _SENTINEL
    # NFKC normalisation
    nfkc = unicodedata.normalize("NFKC", raw)
    # case folding
    folded = nfkc.casefold()
    # whitespace collapsing
    collapsed = " ".join(folded.split())
    if collapsed == "":
        return _SENTINEL
    return collapsed


@dataclass(frozen=True)
class ThemeRule:
    theme: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()


class Taxonomy:
    def __init__(self, rules: Sequence[ThemeRule], fallback: str = "OTHER") -> None:
        self._rules: tuple[ThemeRule, ...] = tuple(rules)
        self._fallback = fallback

    @classmethod
    def from_yaml(cls, path: Path) -> Taxonomy:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # support top-level or nested under 'taxonomy'
        if isinstance(data, dict) and "taxonomy" in data and isinstance(data["taxonomy"], dict):
            data = data["taxonomy"]
        fallback = str(data.get("fallback", "OTHER"))
        raw_rules = data.get("rules", [])
        rules: list[ThemeRule] = []
        for r in raw_rules or []:
            theme = str(r.get("theme", "OTHER"))
            inc = tuple(str(x) for x in (r.get("include") or []))
            exc = tuple(str(x) for x in (r.get("exclude") or []))
            rules.append(ThemeRule(theme=theme, include=inc, exclude=exc))
        return cls(rules=rules, fallback=fallback)

    def classify(self, name: str, index_name: str | None) -> str:
        # pure function of (name, index_name), ordered first-match-wins
        combined = f"{name} {index_name or ''}".casefold()
        for rule in self._rules:
            # check exclude first: if any exclude token appears, skip
            excluded = False
            for ex in rule.exclude:
                if ex.casefold() in combined:
                    excluded = True
                    break
            if excluded:
                continue
            for inc in rule.include:
                if inc.casefold() in combined:
                    return rule.theme
        return self._fallback

    def coverage(self, names: Sequence[tuple[str, str | None]]) -> float:
        if not names:
            return 0.0
        matched = 0
        for n, idx in names:
            if self.classify(n, idx) != self._fallback:
                matched += 1
        return matched / len(names)
