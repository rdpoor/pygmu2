#!/usr/bin/env python3
"""
Generate README.md's PE and example tables from the code itself.

The PE table is rendered from pygmu2.__all__ (plus the lazy-import
registry) using each class's first docstring line; the examples table
from examples/*.py module docstrings. Hand-written tables drift — the
audit found ~40 shipped PEs missing and 7 dead example rows — so the
tables are generated and CI asserts README matches (R4: one concept,
one home).

Usage:
    uv run python scripts/gen_readme_tables.py            # rewrite README
    uv run python scripts/gen_readme_tables.py --check    # exit 1 on drift

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

LAZY_EXPORTS = ["BiquadPE", "AudioReaderPE", "SVFilterPE"]

PE_BEGIN = "<!-- BEGIN GENERATED: pe-table (scripts/gen_readme_tables.py) -->"
PE_END = "<!-- END GENERATED: pe-table -->"
EG_BEGIN = "<!-- BEGIN GENERATED: examples-table (scripts/gen_readme_tables.py) -->"
EG_END = "<!-- END GENERATED: examples-table -->"


def first_doc_line(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line.rstrip(".") + "."
    return ""


def pe_table() -> str:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import pygmu2 as pg
    from pygmu2.processing_element import ProcessingElement

    pg.set_sample_rate(44100)
    rows = []
    for name in sorted(set(pg.__all__) | set(LAZY_EXPORTS)):
        try:
            obj = getattr(pg, name)
        except AttributeError:
            continue
        if (
            isinstance(obj, type)
            and issubclass(obj, ProcessingElement)
            and not inspect.isabstract(obj)
        ):
            rows.append(f"| `{name}` | {first_doc_line(obj.__doc__)} |")
    lines = ["| PE | Description |", "|----|-------------|"] + rows
    return "\n".join(lines)


def examples_table() -> str:
    rows = []
    for path in sorted((REPO_ROOT / "examples").glob("*.py")):
        if path.name == "examples_helper.py":
            continue
        try:
            doc = ast.get_docstring(ast.parse(path.read_text()))
        except SyntaxError:
            doc = None
        rows.append(f"| `{path.name}` | {first_doc_line(doc)} |")
    lines = ["| Example | Description |", "|---------|-------------|"] + rows
    return "\n".join(lines)


def splice(text: str, begin: str, end: str, content: str) -> str:
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise SystemExit(f"README.md is missing markers: {begin}")
    return pattern.sub(f"{begin}\n{content}\n{end}", text)


def main() -> int:
    current = README.read_text()
    updated = splice(current, PE_BEGIN, PE_END, pe_table())
    updated = splice(updated, EG_BEGIN, EG_END, examples_table())

    if "--check" in sys.argv:
        if updated != current:
            print(
                "README.md tables are out of date. "
                "Run: uv run python scripts/gen_readme_tables.py"
            )
            return 1
        print("README.md tables are up to date.")
        return 0

    README.write_text(updated)
    print("README.md tables regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
