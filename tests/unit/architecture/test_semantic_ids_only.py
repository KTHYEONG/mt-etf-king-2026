import re
from pathlib import Path

LEGACY = re.compile(r"""["'](P\d{2}[ABV]?|B[0-5]|M\d{2})["']""")


def test_no_legacy_p_codes_in_src() -> None:
    offenders: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if LEGACY.search(line):
                offenders.append(f"{path}:{lineno}:{line.strip()[:60]}")

    assert offenders == [], f"{len(offenders)} legacy strategy codes remain: " + ", ".join(offenders[:20])
