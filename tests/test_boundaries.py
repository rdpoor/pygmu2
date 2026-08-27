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


# Complete, working PEs that are deliberately not exported yet. Each entry
# needs a decision (export or delete — no limbo, DESIGN_PHILOSOPHY.md R7);
# tracked as IMPLEMENTATION_PLAN.md P3.2.
_EXPORT_OPT_OUT = {
    "pygmu2.portamento_pe.PortamentoPE",
}

# Names served by the lazy-import registry in __init__.py (part of the
# public surface even though not eagerly imported).
_LAZY_EXPORTS = {"BiquadPE", "BiquadMode", "AudioReaderPE", "SVFilterPE"}


def test_every_public_pe_is_exported():
    """R7: the public surface is the product. A concrete PE class that is
    neither exported nor on the commented opt-out list is unreachable —
    export it or delete it."""
    import importlib
    import inspect

    import pygmu2 as pg
    from pygmu2.processing_element import ProcessingElement

    exported = set(pg.__all__) | _LAZY_EXPORTS
    offenders = []
    for f in sorted((REPO_ROOT / "src" / "pygmu2").glob("*.py")):
        if f.stem.startswith("_"):
            continue
        mod_name = f"pygmu2.{f.stem}"
        mod = importlib.import_module(mod_name)
        for name, obj in list(vars(mod).items()):
            try:
                is_pe = (
                    isinstance(obj, type)
                    and issubclass(obj, ProcessingElement)
                    and obj.__module__ == mod_name
                    and not name.startswith("_")
                    and not inspect.isabstract(obj)
                )
            except TypeError:
                continue
            qualified = f"{mod_name}.{name}"
            if is_pe and name not in exported and qualified not in _EXPORT_OPT_OUT:
                offenders.append(qualified)
    assert not offenders, (
        f"Public PE classes neither exported nor opted out: {offenders}. "
        f"Add to __all__ (and tests/pe_factories.py), or delete the class."
    )


def test_readme_tables_match_code():
    """R4: the README's PE and example tables are generated, not
    hand-written. This asserts they match the code (regenerate with
    `uv run python scripts/gen_readme_tables.py`)."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "gen_readme_tables.py"),
            "--check",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_hygiene():
    """`import pygmu2` must not load the heavy optional dependencies —
    they are served by the lazy registry on first use. Budget is generous
    for CI jitter; the <100 ms target is verified manually (plan P2.5)."""
    code = (
        "import time, sys\n"
        "t0 = time.perf_counter()\n"
        "import pygmu2\n"
        "elapsed_ms = (time.perf_counter() - t0) * 1000\n"
        "heavy = [m for m in ('scipy', 'numba', 'mido', 'miniaudio', 'sounddevice')\n"
        "         if any(k == m or k.startswith(m + '.') for k in sys.modules)]\n"
        "assert not heavy, f'heavy modules loaded eagerly: {heavy}'\n"
        "assert elapsed_ms < 500, f'import pygmu2 took {elapsed_ms:.0f} ms'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
