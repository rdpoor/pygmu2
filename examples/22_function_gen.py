#!/usr/bin/env python3
"""
Example 22: FunctionGenPE - naive DSP-like function generator (aliased)

This example demonstrates FunctionGenPE, a deliberately simple oscillator:
- No anti-aliasing (no BLIT/BLEP)
- Useful as a low-level DSP building block

Demos:
1) PWM Rectangle (naive)
2) Saw/Triangle Morph (naive)
3) A/B: FunctionGenPE vs AnalogOscPE at high pitch

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np

from pygmu2 import (
import pygmu2 as pg
pg.set_sample_rate(44100)

    AnalogOscPE,
    AudioRenderer,
    CropPE,
    Extent,
    FunctionGenPE,
    GainPE,
    MixPE,
    PiecewisePE,
    SinePE,
    TransformPE,
    seconds_to_samples,
)


SAMPLE_RATE = 44_100


def demo_pwm_rectangle_naive():
    """
    Classic PWM with the naive rectangle wave.
    """
    print("=== Demo: PWM Rectangle (naive) ===")
    print()

    dur = int(seconds_to_samples(6.0, SAMPLE_RATE))

    duty_lfo_stream = SinePE(frequency=0.25, amplitude=1.0)
    duty_stream = TransformPE(duty_lfo_stream, func=lambda x: 0.5 + 0.45 * x, name="duty_map")

    osc_stream = FunctionGenPE(frequency=110.0, duty_cycle=duty_stream, waveform="rectangle")
    out_stream = GainPE(osc_stream, gain=0.25)
    out_stream = CropPE(out_stream, Extent(0, dur))

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(out_stream)
    with renderer:
        renderer.start()
        renderer.play_extent()


def demo_morph_naive():
    """
    Naive duty-controlled morph:
    duty=0 -> saw up, duty=0.5 -> triangle, duty=1 -> saw down.
    """
    print("=== Demo: Saw/Triangle Morph (naive) ===")
    print()

    dur = int(seconds_to_samples(8.0, SAMPLE_RATE))
    duty_stream = PiecewisePE([(0, 0.0), (dur, 1.0)])

    osc_stream = FunctionGenPE(frequency=220.0, duty_cycle=duty_stream, waveform="sawtooth")
    out_stream = GainPE(osc_stream, gain=0.35)
    out_stream = CropPE(out_stream, Extent(0, dur))

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(out_stream)
    with renderer:
        renderer.start()
        renderer.play_extent()


def demo_ab_high_pitch():
    """
    A/B comparison at high pitch to highlight aliasing:
    - Left: FunctionGenPE (naive, aliased)
    - Right: AnalogOscPE (bandlimited)
    """
    print("=== Demo: A/B at High Pitch (naive vs bandlimited) ===")
    print("Left: FunctionGenPE (naive), Right: AnalogOscPE (bandlimited)")
    print()

    dur = int(seconds_to_samples(6.0, SAMPLE_RATE))

    # Sweep from 500 Hz up near Nyquist
    freq_stream = PiecewisePE([(0, 500.0), (dur, 12_000.0)])

    duty = 0.2

    # Make both sources explicitly 2-channel so AudioRenderer configures a 2-channel stream.
    naive_stream = FunctionGenPE(frequency=freq_stream, duty_cycle=duty, waveform="rectangle", channels=2)
    aa_stream = AnalogOscPE(frequency=freq_stream, duty_cycle=duty, waveform="rectangle", channels=2)

    # Pan by zeroing the opposite channel (keep shape (N,2)).
    left_stream = TransformPE(naive_stream, func=lambda x: np.column_stack([x[:, 0], np.zeros_like(x[:, 0])]), name="pan_left")
    right_stream = TransformPE(aa_stream, func=lambda x: np.column_stack([np.zeros_like(x[:, 0]), x[:, 0]]), name="pan_right")

    stereo_stream = GainPE(MixPE(left_stream, right_stream), gain=0.2)
    stereo_stream = CropPE(stereo_stream, Extent(0, dur))

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(stereo_stream)
    with renderer:
        renderer.start()
        renderer.play_extent()


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "PWM Rectangle (naive)", demo_pwm_rectangle_naive),
        ("2", "Saw/Triangle Morph (naive)", demo_morph_naive),
        ("3", "A/B at High Pitch (naive vs bandlimited)", demo_ab_high_pitch),
        ("a", "All demos", None),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("pygmu2 FunctionGenPE Examples")
        print("----------------------------")
        for key, name, _ in demos:
            print(f"{key}: {name}")
        print()
        choice = input("Choose a demo (1-3 or 'a' for all): ").strip().lower()
        print()

    if choice == "a":
        for _, _, fn in demos:
            if fn is not None:
                fn()
    else:
        for key, _name, fn in demos:
            if key == choice and fn is not None:
                fn()
                break
        else:
            print("Invalid choice.")

