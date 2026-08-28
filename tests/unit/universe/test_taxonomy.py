from __future__ import annotations

from pathlib import Path

from src.universe.taxonomy import Taxonomy, ThemeRule, normalize_index_key


def test_scenario_04_03_normalize_index_key() -> None:
    """SCENARIO-04-03"""
    a = normalize_index_key("코스피 200")
    b = normalize_index_key(" 코스피  200 ")
    assert a == b
    assert normalize_index_key("코스피 200 TR") != a
    sentinel = normalize_index_key(None)
    assert sentinel != a
    assert sentinel == normalize_index_key("")


def test_scenario_04_04_taxonomy_classify_and_coverage() -> None:
    """SCENARIO-04-04"""
    rules = (
        ThemeRule(theme="KOSPI200", include=("코스피 200",), exclude=()),
        ThemeRule(theme="SEMICONDUCTOR", include=("반도체",), exclude=()),
    )
    taxonomy = Taxonomy(rules=rules, fallback="OTHER")
    assert taxonomy.classify("KODEX 200", "코스피 200") == "KOSPI200"
    assert taxonomy.classify("TIGER 반도체", "필라델피아 반도체") == "SEMICONDUCTOR"
    assert taxonomy.classify("UNKNOWN ETF", None) == "OTHER"

    names = [(f"ETF{i}", "코스피 200" if i < 9 else None) for i in range(10)]
    assert taxonomy.coverage(names) == 0.9

    loaded = Taxonomy.from_yaml(Path("configs/taxonomy.yaml"))
    assert loaded.classify("KODEX 200", "코스피 200") != "OTHER"
