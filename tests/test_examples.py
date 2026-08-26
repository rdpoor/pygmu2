"""
Examples smoke test (Tier 3 boundary test).

Executes every examples/*.py with __name__ != "__main__" and asserts no
exception. This is the gate that makes an un-propagated rename in src/
fail CI instead of silently breaking the examples (see DESIGN_PHILOSOPHY.md
R5: the consumer is part of the system).

*.py-disabled files are not covered here; resolving them is tracked in
IMPLEMENTATION_PLAN.md P3.4, and CI forbids creating new ones.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

EXAMPLE_FILES = sorted(
    p for p in EXAMPLES_DIR.glob("*.py") if p.name != "examples_helper.py"
)


@pytest.fixture(autouse=True)
def _examples_on_path():
    """Examples import examples_helper as a sibling module."""
    sys.path.insert(0, str(EXAMPLES_DIR))
    yield
    sys.path.remove(str(EXAMPLES_DIR))


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_imports(path):
    """The example's module-level code runs without error.

    __name__ is set to a non-main value so demo menus and playback
    (guarded by ``if __name__ == "__main__":``) do not run.
    """
    source = path.read_text()
    code = compile(source, str(path), "exec")
    module_globals = {"__name__": "__smoke_test__", "__file__": str(path)}
    exec(code, module_globals)
