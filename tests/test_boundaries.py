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


# Files disabled before the overhaul, grandfathered until resolved (P3.4).
# Fix or delete — do NOT add to this list (change protocol: disabling a
# consumer is not a resolution).
_GRANDFATHERED_DISABLED = {
    "examples/13_random.py-disabled",
    "examples/14_trigger.py-disabled",
    "examples/18_adsr.py-disabled",
    "examples/19_sequence.py-disabled",
    "examples/21_sequence_with_durations.py-disabled",
    "examples/24_slice.py-disabled",
    "examples/25_gating.py-disabled",
    "examples/31_trigger.py-disabled",
    "scripts/toy_midi_sampler.py-disabled",
}


def test_no_new_disabled_files():
    """Renaming a broken file to .py-disabled hides breakage from every
    gate; fix it or delete it instead."""
    found = {
        str(p.relative_to(REPO_ROOT))
        for p in REPO_ROOT.rglob("*.py-disabled")
        if ".venv" not in p.parts
    }
    new = found - _GRANDFATHERED_DISABLED
    assert not new, f"New .py-disabled files (fix or delete instead): {sorted(new)}"


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
