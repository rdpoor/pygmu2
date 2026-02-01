#!/usr/bin/env python3
"""
Example 33: PiecewisePE - piecewise (sample_index, value) curves

Demonstrates PiecewisePE with LINEAR, STEP, EXPONENTIAL, and SIGMOID
transitions, and ZERO vs HOLD_BOTH extend modes. Uses piecewise curves as
gain envelopes on a sine tone so you can hear the shape.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pygmu2 import (
    AudioRenderer,
    PiecewisePE,
    TransitionType,
    CropPE,
    Extent,
    ExtendMode,
    GainPE,
    SinePE,
    WavWriterPE,
    seconds_to_samples,
)

SAMPLE_RATE = 44100


def stos(seconds: float) -> int:
    return int(seconds_to_samples(seconds, SAMPLE_RATE))


def _play(pe, duration_samples: int) -> None:
    out = CropPE(pe, Extent(0, duration_samples))
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(out)
    with renderer:
        renderer.start()
        renderer.play_extent()


def demo_linear_ramp():
    """Linear ramp 0 -> 1 over 1 second as gain on 440 Hz sine."""
    print("=== Piecewise Example 33: Linear ramp (0->1 over 1 s) ===")
    duration = stos(2.0)
    # Ramp from 0 to 1 over first second, hold 1 for second second
    pw = PiecewisePE(
        [(0, 0.0), (stos(1.0), 1.0), (duration, 1.0)],
        transition_type=TransitionType.LINEAR,
        extend_mode=ExtendMode.HOLD_LAST,
    )
    sine = SinePE(frequency=440.0, amplitude=1.0)
    gated = GainPE(sine, pw)
    _play(gated, duration)


def demo_step():
    """Step envelope: 0, 1, 0.5, 0 over ~2 seconds."""
    print("=== Piecewise Example 33: Step envelope ===")
    duration = stos(2.5)
    pw = PiecewisePE(
        [
            (0, 0.0),
            (stos(0.25), 1.0),
            (stos(1.0), 0.5),
            (stos(1.5), 0.0),
            (duration, 0.0),
        ],
        transition_type=TransitionType.STEP,
        extend_mode=ExtendMode.HOLD_LAST,
    )
    sine = SinePE(frequency=330.0, amplitude=1.0)
    gated = GainPE(sine, pw)
    _play(gated, duration)


def demo_sigmoid_smooth():
    """Sigmoid transition between 0 and 1 (smooth S-curve)."""
    print("=== Piecewise Example 33: Sigmoid (smooth S-curve) ===")
    duration = stos(2.0)
    pw = PiecewisePE(
        [(0, 0.0), (stos(0.5), 1.0), (stos(1.0), 0.0), (duration, 0.0)],
        transition_type=TransitionType.SIGMOID,
        extend_mode=ExtendMode.HOLD_LAST,
    )
    sine = SinePE(frequency=554.0, amplitude=1.0)
    gated = GainPE(sine, pw)
    _play(gated, duration)


def demo_hold_both():
    """ExtendMode.HOLD_BOTH: request before first point holds first value."""
    print("=== Piecewise Example 33: HOLD_BOTH (extend before/after) ===")
    # Points only in middle; crop to include region before and after
    duration = stos(2.0)
    pw = PiecewisePE(
        [(stos(0.5), 0.0), (stos(1.0), 1.0), (stos(1.5), 0.0)],
        transition_type=TransitionType.LINEAR,
        extend_mode=ExtendMode.HOLD_BOTH,
    )
    sine = SinePE(frequency=440.0, amplitude=1.0)
    gated = GainPE(sine, pw)
    _play(gated, duration)


def demo_write_to_file():
    """Write a short piecewise envelope to WAV (for inspection)."""
    print("=== Piecewise Example 33: Write envelope to WAV ===")
    duration = stos(1.0)
    pw = PiecewisePE(
        [(0, 0.0), (stos(0.2), 0.8), (stos(0.6), 0.3), (stos(1.0), 0.0)],
        transition_type=TransitionType.SIGMOID,
        extend_mode=ExtendMode.ZERO,
    )
    out = CropPE(pw, Extent(0, duration))
    writer = WavWriterPE(out, "piecewise_envelope.wav", sample_rate=SAMPLE_RATE)
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(writer)
    with renderer:
        renderer.start()
        renderer.play_extent()
    print("Wrote piecewise_envelope.wav (mono, envelope as sample values)")


def demo_all():
    demo_linear_ramp()
    demo_step()
    demo_sigmoid_smooth()
    demo_hold_both()
    demo_write_to_file()


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "Linear ramp (0->1 as gain)", demo_linear_ramp),
        ("2", "Step envelope", demo_step),
        ("3", "Sigmoid smooth envelope", demo_sigmoid_smooth),
        ("4", "HOLD_BOTH extend", demo_hold_both),
        ("5", "Write envelope to WAV", demo_write_to_file),
        ("a", "All demos", demo_all),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("Example 33: PiecewisePE - piecewise curves")
        print("-------------------------------------------")
        for key, name, _ in demos:
            print(f"  {key}: {name}")
        print()
        choice = input("Choice (1-5 or 'a'): ").strip().lower()

    for key, _name, fn in demos:
        if key == choice:
            fn()
            break
    else:
        print("Invalid choice.")
