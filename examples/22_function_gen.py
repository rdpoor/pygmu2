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

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

import numpy as np

from pygmu2 import (
    AnalogOscPE,
    AudioRenderer,
    CropPE,
    Extent,
    FunctionGenPE,
    GainPE,
    MixPE,
    RampPE,
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

    duty_lfo = SinePE(frequency=0.25, amplitude=1.0)
    duty = TransformPE(duty_lfo, func=lambda x: 0.5 + 0.45 * x, name="duty_map")

    osc = FunctionGenPE(frequency=110.0, duty_cycle=duty, waveform="rectangle")
    out = GainPE(osc, gain=0.25)
    out = CropPE(out, Extent(0, dur))

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(out)
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
    duty = RampPE(0.0, 1.0, duration=dur)

    osc = FunctionGenPE(frequency=220.0, duty_cycle=duty, waveform="sawtooth")
    out = GainPE(osc, gain=0.35)
    out = CropPE(out, Extent(0, dur))

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(out)
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
    freq = RampPE(500.0, 12_000.0, duration=dur)

    duty = 0.2

    # Make both sources explicitly 2-channel so AudioRenderer configures a 2-channel stream.
    naive = FunctionGenPE(frequency=freq, duty_cycle=duty, waveform="rectangle", channels=2)
    aa = AnalogOscPE(frequency=freq, duty_cycle=duty, waveform="rectangle", channels=2)

    # Pan by zeroing the opposite channel (keep shape (N,2)).
    left = TransformPE(naive, func=lambda x: np.column_stack([x[:, 0], np.zeros_like(x[:, 0])]), name="pan_left")
    right = TransformPE(aa, func=lambda x: np.column_stack([np.zeros_like(x[:, 0]), x[:, 0]]), name="pan_right")

    stereo = GainPE(MixPE(left, right), gain=0.2)
    stereo = CropPE(stereo, Extent(0, dur))

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(stereo)
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

