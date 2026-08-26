"""
Boundary tests (Tier 3): consumers of src/ still work.

Currently: the benchmark suite discovers its configs without error.
Later phases add: README tables match __all__ (P2.4), import hygiene
(P2.5), export completeness (P2.3). See IMPLEMENTATION_PLAN.md.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_benchmark_suite_discovers():
    """`benchmark_pes.py --list` runs to completion (imports and config
    construction both work). This rotted once (RandomPE rename); this
    test keeps it from rotting silently again."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / "benchmark_pes.py"), "--list"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"benchmark_pes.py --list failed (exit {result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    assert "benchmark configurations" in result.stdout
