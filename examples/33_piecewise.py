#!/usr/bin/env python3
"""
Example 33: PiecewisePE - piecewise (sample_index, value) curves

Uses a piecewise stream to control the pitch of a band-limited sawtooth.
Traces a C major triad (C4, E4, G4, C4) over 8 seconds with 1 s per note.
Each demo uses a different TransitionType so you hear step vs gliding pitch.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pygmu2 import (
    AudioRenderer,
    CropPE,
    Extent,
    ExtendMode,
    FunctionGenPE,
    GainPE,
    PiecewisePE,
    TransformPE,
    TransitionType,
    pitch_to_freq,
)
import pygmu2 as pg
pg.set_sample_rate(44100)


SAMPLE_RATE = 44100
SR = SAMPLE_RATE

# C major triad: (time in samples, MIDI pitch) — 1 s per note
SEGMENTS = [
    (SR * 0, 60),   # C4
    (SR * 1.5, 60),
    (SR * 2, 64),   # E4
    (SR * 3.5, 64),
    (SR * 4, 67),   # G4
    (SR * 5.5, 67),
    (SR * 6, 60),   # C4
    (SR * 7.5, 60),
]
DURATION_SAMPLES = SR * 8


def _play(pe, duration: int = DURATION_SAMPLES) -> None:
    out = CropPE(pe, Extent(0, duration))
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(out)
    with renderer:
        renderer.start()
        renderer.play_extent()


def _make_triad(transition_type: TransitionType):
    """Piecewise MIDI pitch -> frequency -> sawtooth; gain down for comfort."""
    pitch_pe = PiecewisePE(
        SEGMENTS,
        transition_type=transition_type,
        extend_mode=ExtendMode.HOLD_LAST,
    )
    freq_pe = TransformPE(pitch_pe, func=pitch_to_freq, name="pitch_to_freq")
    saw_pe = FunctionGenPE(frequency=freq_pe, duty_cycle=0.5, waveform="sawtooth")
    return GainPE(saw_pe, 0.25)


def demo_step():
    """Step: instant pitch changes (no glide)."""
    print("=== Piecewise 33: STEP (instant pitch changes) ===")
    _play(_make_triad(TransitionType.STEP))


def demo_linear():
    """Linear: constant-rate glide between notes."""
    print("=== Piecewise 33: LINEAR (constant glide between notes) ===")
    _play(_make_triad(TransitionType.LINEAR))


def demo_exponential():
    """Exponential: pitch glides with exponential curve."""
    print("=== Piecewise 33: EXPONENTIAL (exponential pitch glide) ===")
    _play(_make_triad(TransitionType.EXPONENTIAL))


def demo_sigmoid():
    """Sigmoid: S-curve glide (slow at note boundaries)."""
    print("=== Piecewise 33: SIGMOID (S-curve glide) ===")
    _play(_make_triad(TransitionType.SIGMOID))


def demo_constant_power():
    """Constant-power: sin/cos-style curve between notes."""
    print("=== Piecewise 33: CONSTANT_POWER (sin/cos-style glide) ===")
    _play(_make_triad(TransitionType.CONSTANT_POWER))


def demo_all():
    demo_step()
    demo_linear()
    demo_exponential()
    demo_sigmoid()
    demo_constant_power()


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "STEP (instant pitch)", demo_step),
        ("2", "LINEAR (constant glide)", demo_linear),
        ("3", "EXPONENTIAL", demo_exponential),
        ("4", "SIGMOID (S-curve)", demo_sigmoid),
        ("5", "CONSTANT_POWER", demo_constant_power),
        ("a", "All transition types", demo_all),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("Example 33: PiecewisePE — C major triad pitch (sawtooth)")
        print("----------------------------------------------------------")
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
