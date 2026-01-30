#!/usr/bin/env python3
"""
Convert MIT KEMAR HRTF .dat files to WAV.

Reads the compact KEMAR dataset (stereo .dat: 16-bit big-endian, 128 taps
per channel, interleaved L/R). Writes 44.1 kHz stereo WAV with the same
base name (e.g. H0e0a.dat -> H0e0a.wav).

Note: compact.zip from MIT may already contain .wav files; in that case
just unzip into examples/audio/kemar and skip this script.

Usage (only if your archive has .dat files):
  1. Download compact.zip from https://sound.media.mit.edu/resources/KEMAR/
  2. Unzip into a directory (e.g. kemar_raw).
  3. Run:
       python scripts/convert_kemar_to_wav.py --input kemar_raw --output examples/audio/kemar

Filenames: H{elevation}e{azimuth}a — elevation = optional '-' + two-digit number (degrees);
azimuth = three-digit number (degrees). E.g. H00e045a, H-40e090a, H90e180a.
Only files matching H*e*a.dat are converted (compact stereo set).

Data: Copyright 1994 MIT Media Laboratory. Free use with citation (Gardner & Martin).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

KEMAR_SAMPLE_RATE = 44100
COMPACT_SAMPLES_PER_CHANNEL = 128  # compact = 256 interleaved samples = 128 L + 128 R


def parse_kemar_filename(path: Path) -> tuple[int, int] | None:
    """Return (elevation, azimuth) if path matches H{el}e{az}a.(dat|wav), else None.
    Elevation = optional '-' + two-digit number; azimuth = three-digit number."""
    name = path.name
    m = re.match(r"^H(-?\d{2})e(\d{3})a\.(dat|wav)$", name, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def read_kemar_dat(path: Path) -> np.ndarray:
    """
    Read one compact KEMAR .dat file.
    Returns (samples, 2) float32 in [-1, 1], channels L, R.
    """
    raw = np.fromfile(path, dtype=">i2")  # big-endian 16-bit
    n = len(raw)
    if n != COMPACT_SAMPLES_PER_CHANNEL * 2:
        raise ValueError(
            f"{path.name}: expected {COMPACT_SAMPLES_PER_CHANNEL * 2} samples (compact stereo), got {n}"
        )
    # Deinterleave: raw is L0,R0,L1,R1,...
    left = raw[0::2].astype(np.float32) / 32768.0
    right = raw[1::2].astype(np.float32) / 32768.0
    return np.stack([left, right], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert MIT KEMAR compact .dat HRTF files to WAV."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Directory containing extracted .dat files (e.g. from compact.zip)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("examples/audio/kemar"),
        help="Output directory for .wav files (default: examples/audio/kemar)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be converted without writing",
    )
    args = parser.parse_args()

    indir = args.input.resolve()
    outdir = args.output.resolve()

    if not indir.is_dir():
        print(f"Error: input is not a directory: {indir}", file=sys.stderr)
        return 1

    dat_files = sorted(indir.glob("H*e*a.dat"))
    if not dat_files:
        print(
            f"No H*e*a.dat files found in {indir}. Unzip compact.zip there.",
            file=sys.stderr,
        )
        return 1

    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for path in dat_files:
        if parse_kemar_filename(path) is None:
            continue
        try:
            data = read_kemar_dat(path)
        except Exception as e:
            print(f"Skip {path.name}: {e}", file=sys.stderr)
            continue
        out_path = outdir / (path.stem + ".wav")
        if args.dry_run:
            print(f"  {path.name} -> {out_path.relative_to(outdir)}")
        else:
            sf.write(out_path, data, KEMAR_SAMPLE_RATE, subtype="PCM_16")
        converted += 1

    print(f"Converted {converted} files to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
