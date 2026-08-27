#!/usr/bin/env python3
"""
Profile BiquadPE vs SVFilterPE using pygmu2.diagnostics.

Builds two equivalent graphs (autowah-style: envelope -> freq control -> lowpass
filter), one using BiquadPE and one using SVFilterPE, renders the same extent
with NullRenderer under diagnostics (per-PE render timing + pull counts), then
prints the per-PE report for each and compares total wall time.

Run from project root: python benchmarks/profile_biquad_vs_svfilter.py

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import sys
import time
from pathlib import Path

# Add src for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pygmu2 as pg
from pygmu2 import (
    SinePE,
    EnvelopePE,
    DetectionMode,
    BiquadPE,
    BiquadMode,
    SVFilterPE,
    TransformPE,
    GainPE,
    CropPE,
    NullRenderer,
)
from pygmu2 import diagnostics

SAMPLE_RATE = 44100
DURATION_SECONDS = 8
BLOCK_SIZE = 1024


def envelope_to_freq(env):
    """Map envelope (0-1) to frequency (100-3000 Hz)."""
    import numpy as np

    env = np.clip(env, 0, 1)
    return 100.0 + (3000.0 - 100.0) * (env**0.5)


def make_biquad_graph():
    """Build autowah-style graph using BiquadPE."""
    source = SinePE(frequency=220.0, amplitude=0.8)
    envelope = EnvelopePE(source, attack=0.005, release=0.05, mode=DetectionMode.PEAK)
    freq_control = TransformPE(envelope, func=envelope_to_freq, name="env_to_freq")
    filtered = BiquadPE(
        source,
        frequency=freq_control,
        q=10.0,
        mode=BiquadMode.LOWPASS,
    )
    return GainPE(filtered, gain=1.0)


def make_svfilter_graph():
    """Build autowah-style graph using SVFilterPE."""
    source = SinePE(frequency=220.0, amplitude=0.8)
    envelope = EnvelopePE(source, attack=0.005, release=0.05, mode=DetectionMode.PEAK)
    freq_control = TransformPE(envelope, func=envelope_to_freq, name="env_to_freq")
    filtered = SVFilterPE(
        source,
        frequency=freq_control,
        q=10.0,
        mode=BiquadMode.LOWPASS,
    )
    return GainPE(filtered, gain=1.0)


def run_profiled(root_pe, duration_samples: int) -> float:
    """Render the full extent in blocks under diagnostics.

    Returns total wall time in seconds; per-PE data accumulates in the
    diagnostics module (print with diagnostics.get_block_report()).
    """
    renderer = NullRenderer(sample_rate=SAMPLE_RATE)
    cropped = CropPE(root_pe, 0, duration_samples)
    renderer.set_source(cropped)
    renderer.start()
    diagnostics.reset_block()
    t0 = time.perf_counter()
    num_blocks = (duration_samples + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(num_blocks):
        start = i * BLOCK_SIZE
        duration = min(BLOCK_SIZE, duration_samples - start)
        if duration <= 0:
            break
        renderer.render(start, duration)
    elapsed = time.perf_counter() - t0
    renderer.stop()
    return elapsed


def main():
    pg.set_sample_rate(SAMPLE_RATE)
    duration_samples = int(DURATION_SECONDS * SAMPLE_RATE)
    diagnostics.enable(pull_count=True, timing=True)

    print("Profiling BiquadPE vs SVFilterPE (autowah-style graph)")
    print(
        f"  Sample rate: {SAMPLE_RATE}, duration: {DURATION_SECONDS}s, block size: {BLOCK_SIZE}"
    )
    print()

    # --- BiquadPE ---
    print("=" * 70)
    print("RUN 1: BiquadPE (envelope -> freq -> BiquadPE lowpass)")
    print("=" * 70)
    t_biquad = run_profiled(make_biquad_graph(), duration_samples)
    print(diagnostics.get_block_report())

    # --- SVFilterPE ---
    print()
    print("=" * 70)
    print("RUN 2: SVFilterPE (envelope -> freq -> SVFilterPE lowpass)")
    print("=" * 70)
    t_svfilter = run_profiled(make_svfilter_graph(), duration_samples)
    print(diagnostics.get_block_report())

    diagnostics.disable()

    # --- Comparison ---
    print()
    print("COMPARISON (total wall time)")
    print("-" * 70)
    print(f"  BiquadPE:   {t_biquad * 1000:>10.2f} ms")
    print(f"  SVFilterPE: {t_svfilter * 1000:>10.2f} ms")
    if t_biquad > 0:
        print(f"  Ratio (SVF/Biquad): {t_svfilter / t_biquad:.2f}x")
    print()


if __name__ == "__main__":
    main()
