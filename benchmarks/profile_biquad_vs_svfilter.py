#!/usr/bin/env python3
"""
Profile BiquadPE vs SVFilterPE using the Renderer's built-in profiling.

Builds two equivalent graphs (autowah-style: envelope -> freq control -> lowpass
filter), one using BiquadPE and one using SVFilterPE, renders the same extent
with NullRenderer and enable_profiling(), then prints the profile report for each.

Run from project root: python benchmarks/profile_biquad_vs_svfilter.py

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import sys
from pathlib import Path

# Add src for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
    Extent,
)

SAMPLE_RATE = 44100
DURATION_SECONDS = 8
BLOCK_SIZE = 1024


def envelope_to_freq(env):
    """Map envelope (0-1) to frequency (100-3000 Hz)."""
    import numpy as np
    env = np.clip(env, 0, 1)
    return 100.0 + (3000.0 - 100.0) * (env ** 0.5)


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


def run_profiled(renderer: NullRenderer, root_pe, duration_samples: int) -> None:
    """Render the full extent in blocks with profiling enabled."""
    cropped = CropPE(root_pe, Extent(0, duration_samples))
    renderer.set_source(cropped)
    renderer.start()
    num_blocks = (duration_samples + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(num_blocks):
        start = i * BLOCK_SIZE
        duration = min(BLOCK_SIZE, duration_samples - start)
        if duration <= 0:
            break
        renderer.render(start, duration)
    renderer.stop()


def main():
    duration_samples = int(DURATION_SECONDS * SAMPLE_RATE)
    renderer = NullRenderer(sample_rate=SAMPLE_RATE)

    print("Profiling BiquadPE vs SVFilterPE (autowah-style graph)")
    print(f"  Sample rate: {SAMPLE_RATE}, duration: {DURATION_SECONDS}s, block size: {BLOCK_SIZE}")
    print()

    # --- BiquadPE ---
    print("=" * 70)
    print("RUN 1: BiquadPE (envelope -> freq -> BiquadPE lowpass)")
    print("=" * 70)
    renderer.enable_profiling()
    run_profiled(renderer, make_biquad_graph(), duration_samples)
    renderer.print_profile_report()
    report_biquad = renderer.get_profile_report()

    # --- SVFilterPE ---
    print()
    print("=" * 70)
    print("RUN 2: SVFilterPE (envelope -> freq -> SVFilterPE lowpass)")
    print("=" * 70)
    renderer.enable_profiling()  # Reset report
    run_profiled(renderer, make_svfilter_graph(), duration_samples)
    renderer.print_profile_report()
    report_svfilter = renderer.get_profile_report()

    # --- Comparison ---
    if report_biquad and report_svfilter:
        t_bq_ms = report_biquad.total_render_time_ns / 1_000_000
        t_sv_ms = report_svfilter.total_render_time_ns / 1_000_000
        print()
        print("COMPARISON (total render time)")
        print("-" * 70)
        print(f"  BiquadPE:   {t_bq_ms:>10.2f} ms")
        print(f"  SVFilterPE: {t_sv_ms:>10.2f} ms")
        if t_bq_ms > 0:
            ratio = t_sv_ms / t_bq_ms
            print(f"  Ratio (SVF/Biquad): {ratio:.2f}x")
        print()


if __name__ == "__main__":
    main()
    sys.exit(0)
