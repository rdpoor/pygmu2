#!/usr/bin/env python3
"""
Decaying sine tone synthesis — tau refactor.

tau replaces the old duration / rho pair.  db_floor (default −60 dB) sets
the crop point independently of the decay rate.  DecayingSinePE now carries
a finite extent, so CropPE is no longer needed to bound individual notes.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""
import math

from pygmu2 import (
    CropPE,
    DecayingSinePE,
    DelayPE,
    MixPE,
    pitch_to_freq,
)
import pygmu2 as pg
from examples_helper import run_demos

pg.set_sample_rate(44100)
SAMPLE_RATE = 44100


def s2s(seconds):
    return int(round(seconds * SAMPLE_RATE))


def tau_to_samples(tau, db_floor=-60.0):
    """Sample count at which a DecayingSinePE envelope reaches db_floor."""
    return math.ceil(-tau * SAMPLE_RATE * (db_floor / 20.0) * math.log(10))


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_single_tone():
    """One 440 Hz tone, tau = 0.3 s (reaches −60 dB in ≈ 2.1 s)."""
    print("=== Decaying sine: single tone (440 Hz, tau=0.3 s) ===")
    tone = DecayingSinePE(frequency=440.0, amplitude=0.5, tau=0.3)
    print(f"    extent = {tone.extent()}")
    pg.play(tone, SAMPLE_RATE)


def demo_fast_vs_slow_decay():
    """Same pitch: slow decay (tau=0.3 s) followed by fast decay (tau=0.03 s)."""
    print("=== Decaying sine: slow (tau=0.3) then fast (tau=0.03) ===")
    notes = []
    t = 0.0

    tau_slow = 0.3
    notes.append(DelayPE(
        DecayingSinePE(frequency=440.0, amplitude=0.5, tau=tau_slow),
        s2s(t),
    ))
    t += tau_to_samples(tau_slow) / SAMPLE_RATE + 0.25

    tau_fast = 0.03
    notes.append(DelayPE(
        DecayingSinePE(frequency=440.0, amplitude=0.5, tau=tau_fast),
        s2s(t),
    ))
    t += tau_to_samples(tau_fast) / SAMPLE_RATE + 0.25

    mix = MixPE(*notes)
    pg.play(CropPE(mix, 0, s2s(t)), SAMPLE_RATE)


def demo_major_triad_chord():
    """C major triad (C4, E4, G4) struck simultaneously."""
    print("=== Decaying sine: C major triad (C4, E4, G4) ===")
    tau = 0.43              # reaches −60 dB in ≈ 3 s
    voices = [
        DecayingSinePE(frequency=pitch_to_freq(midi), amplitude=0.1, tau=tau)
        for midi in [60, 64, 67]
    ]
    mix = MixPE(*voices)
    pg.play(CropPE(mix, 0, tau_to_samples(tau)), SAMPLE_RATE)


def demo_c_major_arpeggio():
    """C major arpeggio (C4, E4, G4, C5), notes spaced 0.4 s apart, tau=0.15 s."""
    print("=== Decaying sine: C major arpeggio (C4 E4 G4 C5) ===")
    midi_notes    = [60, 64, 67, 72]
    tau           = 0.15
    note_spacing  = 0.4
    notes = []
    t = 0.0
    for midi in midi_notes:
        notes.append(DelayPE(
            DecayingSinePE(frequency=pitch_to_freq(midi), amplitude=0.2, tau=tau),
            s2s(t),
        ))
        t += note_spacing
    mix   = MixPE(*notes)
    total = s2s(t) + tau_to_samples(tau)
    pg.play(CropPE(mix, 0, total), SAMPLE_RATE)


def demo_db_floor():
    """
    Same tau, three db_floor values: −20, −40, −60 dB.

    The decay rate is identical in all three; only the crop point changes.
    Each note is perceptibly shorter than the last, isolating db_floor's
    effect from the decay rate.
    """
    print("=== Decaying sine: db_floor comparison (-20, -40, -60 dB) ===")
    tau    = 0.3
    floors = [-20.0, -40.0, -60.0]
    gap    = 0.5
    notes  = []
    t      = 0.0
    for db_floor in floors:
        notes.append(DelayPE(
            DecayingSinePE(frequency=440.0, amplitude=0.5, tau=tau, db_floor=db_floor),
            s2s(t),
        ))
        t += tau_to_samples(tau, db_floor) / SAMPLE_RATE + gap
    mix = MixPE(*notes)
    pg.play(CropPE(mix, 0, s2s(t)), SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Demo registry
# ---------------------------------------------------------------------------

DEMOS = [
    ("Single tone (440 Hz, tau=0.3 s)", demo_single_tone),
    ("Fast vs slow decay (same pitch)",demo_fast_vs_slow_decay),
    ("C major triad (simultaneous)", demo_major_triad_chord),
    ("C major arpeggio", demo_c_major_arpeggio),
    ("db_floor comparison (-20, -40, -60 dB)", demo_db_floor),
]

if __name__ == "__main__":
    run_demos(DEMOS)
