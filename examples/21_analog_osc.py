#!/usr/bin/env python3
"""
Example 21: AnalogOscPE - bandlimited PWM + saw/triangle morph oscillator

Demonstrates AnalogOscPE (bandlimited polyBLEP-style oscillator):

1) PWM Rectangle: duty-cycle LFO for classic pulse-width modulation
2) Morphing Saw/Triangle: duty ramps from near 0 to near 1 (saw up -> triangle -> saw down)
3) Subtractive Patch: oscillator into LadderPE lowpass

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pygmu2 import (
    AnalogOscPE,
    AudioRenderer,
    CropPE,
    Extent,
    GainPE,
    LadderMode,
    LadderPE,
    PiecewisePE,
    SinePE,
    TransformPE,
    seconds_to_samples,
)
import pygmu2 as pg
pg.set_sample_rate(44100)



SAMPLE_RATE = 44_100


def demo_pwm_rectangle():
    """
    Classic PWM: pulse wave with a slow duty-cycle LFO.
    """
    print("=== Demo: PWM Rectangle ===")
    print("AnalogOscPE(rectangle) with a duty-cycle LFO")
    print()

    dur_samples = int(seconds_to_samples(6.0, SAMPLE_RATE))

    # Map sine [-1,1] to duty [0.05, 0.95]
    duty_lfo = SinePE(frequency=0.25, amplitude=1.0)
    duty = TransformPE(duty_lfo, func=lambda x: 0.5 + 0.45 * x, name="duty_map")

    osc = AnalogOscPE(frequency=110.0, duty_cycle=duty, waveform="rectangle")
    out = GainPE(osc, gain=0.25)
    out = CropPE(out, 0, (dur_samples) - (0))

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(out)
    with renderer:
        renderer.start()
        renderer.play_extent()


def demo_morphing_saw_triangle():
    """
    Duty-controlled saw/triangle morph:
    duty=0 -> rising saw, duty=0.5 -> triangle, duty=1 -> falling saw.
    """
    print("=== Demo: Morphing Saw/Triangle ===")
    print("AnalogOscPE(sawtooth) with duty ramp: saw up -> triangle -> saw down")
    print()

    dur_samples = int(seconds_to_samples(8.0, SAMPLE_RATE))
    duty = PiecewisePE([(0, 0.05), (dur_samples, 0.95)])

    osc = AnalogOscPE(frequency=220.0, duty_cycle=duty, waveform="sawtooth")
    out = GainPE(osc, gain=0.35)
    out = CropPE(out, 0, (dur_samples) - (0))

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(out)
    with renderer:
        renderer.start()
        renderer.play_extent()


def demo_subtractive_patch():
    """
    A simple subtractive synth patch: oscillator -> ladder lowpass.

    (This is why AnalogOscPE doesn't include a dedicated 'bandwidth' knob:
     patching a filter PE gives more flexibility and feels synth-like.)
    """
    print("=== Demo: Subtractive Patch (Osc -> LadderLPF) ===")
    print()

    dur_samples = int(seconds_to_samples(8.0, SAMPLE_RATE))

    duty_lfo = SinePE(frequency=0.15, amplitude=1.0)
    duty = TransformPE(duty_lfo, func=lambda x: 0.5 + 0.40 * x, name="duty_map")

    osc = AnalogOscPE(frequency=110.0, duty_cycle=duty, waveform="rectangle")

    # Slow cutoff sweep
    cutoff = PiecewisePE([(0, 400.0), (dur_samples, 3200.0)])
    filtered = LadderPE(
        osc,
        mode=LadderMode.LP24,
        frequency=cutoff,
        resonance=0.4,
        drive=1.2,
    )

    out = GainPE(filtered, gain=0.25)
    out = CropPE(out, 0, (dur_samples) - (0))

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(out)
    with renderer:
        renderer.start()
        renderer.play_extent()


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "PWM Rectangle", demo_pwm_rectangle),
        ("2", "Morphing Saw/Triangle", demo_morphing_saw_triangle),
        ("3", "Subtractive Patch (Osc -> LadderLPF)", demo_subtractive_patch),
        ("a", "All demos", None),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("pygmu2 AnalogOscPE Examples")
        print("--------------------------")
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

