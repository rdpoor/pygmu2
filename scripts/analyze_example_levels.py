#!/usr/bin/env python3
"""
Analyze peak/RMS levels of every pygmu2 example demo without playing audio.

Each demo function is imported and called; pg.play / pg.play_offline / pg.browse
are monkey-patched to render audio silently to temp WAV files instead of sending
to speakers.  Each captured clip is then measured for peak and RMS dBFS, and a
suggested gain multiplier (to reach --target dBFS peak) is printed.

Usage:
    uv run python scripts/analyze_example_levels.py
    uv run python scripts/analyze_example_levels.py --target -3.0
    uv run python scripts/analyze_example_levels.py --out levels_report.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import pygmu2 as pg  # noqa: E402  (must come after path setup)

# ── Monkey-patch ──────────────────────────────────────────────────────────────
# _clips accumulates temp WAV paths for the *current* demo call.
# Cleared before each demo; drained and cleaned up after.
_clips: list[str] = []


def _fake_play(source, sample_rate=None, **kwargs):
    """Redirect pg.play / pg.play_offline / pg.browse to a silent render."""
    sr = sample_rate or pg.get_sample_rate() or 44100
    try:
        extent = source.extent()
    except Exception as e:
        print(f"    [extent error: {e}]")
        return
    if extent.start is None or extent.end is None:
        print("    [skip: infinite extent]")
        return
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        pg.render_to_file(source, path, sample_rate=sr, extent=extent)
        _clips.append(path)
    except Exception as e:
        print(f"    [render error: {e}]")
        try:
            os.unlink(path)
        except OSError:
            pass


pg.play = _fake_play
pg.play_offline = _fake_play
pg.browse = _fake_play

# ── Level analysis ────────────────────────────────────────────────────────────


def analyze_wav(path: str) -> tuple[float, float]:
    """Return (peak_dBFS, rms_dBFS) for a WAV file."""
    data, _ = sf.read(path, dtype="float32")
    if data.ndim == 1:
        data = data[:, None]
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data**2)))
    peak = max(peak, 1e-10)
    rms = max(rms, 1e-10)
    return 20 * math.log10(peak), 20 * math.log10(rms)


# ── Example discovery ─────────────────────────────────────────────────────────

SKIP_FILES = {"examples_helper.py", "demo_asset_manager.py"}


def load_module(py_file: Path):
    """Import a Python file as a module; return the module or None on error."""
    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
    mod = importlib.util.module_from_spec(spec)
    # Make the module visible to itself as __main__ would see it
    mod.__file__ = str(py_file)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass  # some scripts call sys.exit() at module level
    except Exception as e:
        print(f"  [import error] {py_file.name}: {e}")
        return None
    return mod


# ── Main ──────────────────────────────────────────────────────────────────────


def run_all(target_db: float = -6.0) -> list[tuple]:
    """
    Run every demo in every example file; return list of result rows.

    Each row: (filename, demo_label, peak_dBFS, rms_dBFS, suggested_gain)
    """
    examples_dir = ROOT / "examples"
    rows: list[tuple] = []

    for py_file in sorted(examples_dir.glob("*.py")):
        if py_file.name in SKIP_FILES:
            continue

        print(f"\n── {py_file.name} ──")
        mod = load_module(py_file)
        if mod is None:
            continue

        demos = getattr(mod, "DEMOS", None)
        if not demos:
            print("  [skip] no DEMOS list")
            continue

        for name, fn in demos:
            print(f"  {name} ...", end=" ", flush=True)
            _clips.clear()
            try:
                fn()
            except SystemExit:
                pass
            except Exception as e:
                print(f"\n  [error in {name}]: {e}")

            n = len(_clips)
            if n == 0:
                print("(no audio captured)")
                continue

            print(f"({n} clip{'s' if n != 1 else ''})")
            for idx, path in enumerate(_clips):
                label = name if n == 1 else f"{name} [{idx + 1}]"
                try:
                    peak_db, rms_db = analyze_wav(path)
                    gain = 10 ** ((target_db - peak_db) / 20)
                    rows.append((py_file.name, label, peak_db, rms_db, gain))
                except Exception as e:
                    print(f"    [analysis error: {e}]")
                finally:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            _clips.clear()

    return rows


def print_report(rows: list[tuple], target_db: float, out_file=None):
    col_file = 32
    col_demo = 44
    header = (
        f"{'File':<{col_file}} {'Demo':<{col_demo}}"
        f" {'Peak':>9} {'RMS':>9} {'Sug.gain':>12}"
    )
    sep = "-" * len(header)
    lines = [
        "",
        f"Target peak: {target_db:.1f} dBFS",
        header,
        sep,
    ]
    for fname, demo, peak, rms, gain in rows:
        gain_db = target_db - peak
        lines.append(
            f"{fname:<{col_file}} {demo:<{col_demo}}"
            f" {peak:>8.1f}  {rms:>8.1f}  {gain:>7.3f}×  ({gain_db:+.1f} dB)"
        )
    lines.append(sep)
    lines.append(f"{len(rows)} clips analysed.")
    report = "\n".join(lines)
    print(report)
    if out_file:
        Path(out_file).write_text(report + "\n", encoding="utf-8")
        print(f"\nReport written to {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure peak/RMS levels of all pygmu2 example demos."
    )
    parser.add_argument(
        "--target",
        type=float,
        default=-6.0,
        metavar="dBFS",
        help="Target peak level for suggested gain (default: -6.0)",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="Also write report to this file.",
    )
    args = parser.parse_args()

    rows = run_all(target_db=args.target)
    print_report(rows, target_db=args.target, out_file=args.out)


if __name__ == "__main__":
    main()
