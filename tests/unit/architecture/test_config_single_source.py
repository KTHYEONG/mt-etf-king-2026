import re
from pathlib import Path

PATTERN = re.compile(r"""(open|Path)\(\s*["']configs/""")


def test_no_cwd_relative_config_reads_in_src() -> None:
    offenders: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        if "__pycache__" in path.parts or path.as_posix() == "src/core/config.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if PATTERN.search(line):
                offenders.append(f"{path}:{lineno}")

    assert offenders == [], "CWD-relative config reads remain: " + ", ".join(offenders)
