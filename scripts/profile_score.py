#!/usr/bin/env python3
"""
Profile a pygmu2 score (PE graph) to find hot spots.

Uses diagnostics.enable(timing=True) so each PE's render() is timed.
Renders the full extent with NullRenderer (no audio I/O). Do NOT
reset_block() between chunks so timings accumulate over the whole piece.
Prints a per-PE breakdown (total ms, call count, avg ms) sorted by total time.

Usage:
  From your score script, after building the source PE:
    from scripts.profile_score import profile_score
    profile_score(source, sample_rate=44100, chunk_size=4096)

  Or run this module and set SOURCE in the __main__ block.

  uv run python scripts/profile_score.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pygmu2.diagnostics import enable, get_block_report, reset_block


def profile_score(
    source,
    sample_rate: int = 44100,
    chunk_size: int = 4096,
) -> str:
    """
    Render the full extent of source with timing enabled; return report string.

    Args:
        source: Root PE (your score). Must have finite extent.
        sample_rate: Sample rate for configure() and extent.
        chunk_size: Samples per render call (smaller = more calls, smoother stats).

    Returns:
        Per-PE timing report string (same format as get_block_report()).
    """
    extent = source.extent()
    if extent.start is None or extent.end is None:
        raise ValueError(
            "profile_score requires finite extent; "
            f"got start={extent.start}, end={extent.end}"
        )
    duration = extent.end - extent.start
    if duration <= 0:
        raise ValueError(f"profile_score requires positive duration; got {duration}")

    source.configure(sample_rate)
    enable(pull_count=True, timing=True)
    reset_block()

    # Render full extent in chunks; do NOT reset_block() between chunks
    pos = extent.start
    while pos < extent.end:
        chunk = min(chunk_size, extent.end - pos)
        source.render(pos, chunk)
        pos += chunk

    return get_block_report()


if __name__ == "__main__":
    # Example: profile a short sine burst. Replace SOURCE with your score.
    from pygmu2 import set_sample_rate, SinePE, CropPE

    set_sample_rate(44100)
    SOURCE = CropPE(SinePE(440.0), 0, 44100 * 5)  # 5 seconds

    report = profile_score(SOURCE, sample_rate=44100, chunk_size=4096)
    print(report)
