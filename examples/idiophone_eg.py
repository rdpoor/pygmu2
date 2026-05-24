#!/usr/bin/env python3
"""
Struck bar idiophone synthesis using IdiophonePE.

Each instrument is modelled as a sum of DecayingSinePE partials whose decay
times register-scale with pitch (bass notes ring longer than treble notes).
Instruments: marimba, xylophone, glockenspiel, balafon.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""
from pygmu2 import (
    CropPE,
    DelayPE,
    MixPE,
    pitch_to_freq,
)
import pygmu2 as pg
from pygmu2.idiophone_pe import (
    BALAFON,
    GLOCKENSPIEL,
    MARIMBA,
    XYLOPHONE,
    IdiophonePE,
)
from examples_helper import run_demos

pg.set_sample_rate(44100)
SAMPLE_RATE = 44100


def s2s(seconds):
    return int(round(seconds * SAMPLE_RATE))


# ---------------------------------------------------------------------------
# Note and arpeggio builders
# ---------------------------------------------------------------------------

def _make_note(instrument, frequency, amplitude, crop_seconds):
    """IdiophonePE cropped to crop_seconds — gives the note a finite extent."""
    return CropPE(
        IdiophonePE(instrument, frequency=frequency, amplitude=amplitude),
        0,
        s2s(crop_seconds),
    )


def _make_arpeggio(instrument, midi_notes, note_spacing, crop_seconds, amplitude=0.3):
    """
    Sequential arpeggio: one note every note_spacing seconds, each ringing
    for crop_seconds.  crop_seconds > note_spacing produces natural overlap.
    Returns a MixPE spanning the full duration.
    """
    notes = []
    t = 0.0
    for midi in midi_notes:
        notes.append(
            DelayPE(
                _make_note(instrument, pitch_to_freq(midi), amplitude, crop_seconds),
                s2s(t),
            )
        )
        t += note_spacing
    total = t + crop_seconds
    return CropPE(MixPE(*notes), 0, s2s(total))


# ---------------------------------------------------------------------------
# Scale patterns
# ---------------------------------------------------------------------------

# C diatonic across two octaves: C3 – C5
C_DIATONIC_2OCT = [48, 50, 52, 53, 55, 57, 59,
                   60, 62, 64, 65, 67, 69, 71, 72]

# C pentatonic across two octaves: C3 – C5
C_PENTATONIC_2OCT = [48, 50, 52, 55, 57,
                     60, 62, 64, 67, 69, 72]


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_marimba_arpeggio():
    """
    Marimba: C diatonic, C3–C5.
    Slow spacing lets each note's fundamental decay be audible.
    crop_seconds >> note_spacing so adjacent notes overlap and blend.
    """
    print("=== Marimba: C diatonic arpeggio (C3–C5) ===")
    mix = _make_arpeggio(
        MARIMBA,
        midi_notes=C_DIATONIC_2OCT,
        note_spacing=0.12,
        crop_seconds=1.5,     # tau_mid = 1.0 s at A4; 1.5 s allows full decay
        amplitude=0.25,
    )
    pg.play(mix, SAMPLE_RATE)


def demo_xylophone_arpeggio():
    """
    Xylophone: C diatonic, C4–C6.
    Short decay (tau_mid = 0.3 s) suits faster spacing; notes stay crisp.
    """
    print("=== Xylophone: C diatonic arpeggio (C4–C6) ===")
    midi_notes = [m + 12 for m in C_DIATONIC_2OCT]   # shift up one octave
    mix = _make_arpeggio(
        XYLOPHONE,
        midi_notes=midi_notes,
        note_spacing=0.09,
        crop_seconds=0.45,    # tau_mid = 0.3 s at A4
        amplitude=0.25,
    )
    pg.play(mix, SAMPLE_RATE)


def demo_glockenspiel_arpeggio():
    """
    Glockenspiel: C diatonic, C5–C7.
    Long tau_mid (3 s) and inharmonic overtones (2.756 f0) give the
    characteristic bright shimmer; notes overlap heavily.
    """
    print("=== Glockenspiel: C diatonic arpeggio (C5–C7) ===")
    midi_notes = [m + 24 for m in C_DIATONIC_2OCT]   # shift up two octaves
    mix = _make_arpeggio(
        GLOCKENSPIEL,
        midi_notes=midi_notes,
        note_spacing=0.15,
        crop_seconds=4.0,     # tau_mid = 3.0 s at A4; long ring-out
        amplitude=0.18,
    )
    pg.play(pg.GainPE(mix, gain=1.3), SAMPLE_RATE)


def demo_balafon_arpeggio():
    """
    Balafon: C pentatonic, C3–C5.
    Pentatonic pattern suits the instrument's West African character.
    Decay sits between marimba and xylophone.
    """
    print("=== Balafon: C pentatonic arpeggio (C3–C5) ===")
    mix = _make_arpeggio(
        BALAFON,
        midi_notes=C_PENTATONIC_2OCT,
        note_spacing=0.12,
        crop_seconds=1.2,     # tau_mid = 0.9 s at A4
        amplitude=0.25,
    )
    pg.play(mix, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Demo registry
# ---------------------------------------------------------------------------

DEMOS = [
    ("Marimba arpeggio (C diatonic, C3–C5)",       demo_marimba_arpeggio),
    ("Xylophone arpeggio (C diatonic, C4–C6)",      demo_xylophone_arpeggio),
    ("Glockenspiel arpeggio (C diatonic, C5–C7)",   demo_glockenspiel_arpeggio),
    ("Balafon arpeggio (C pentatonic, C3–C5)",      demo_balafon_arpeggio),
]

if __name__ == "__main__":
    run_demos(DEMOS)
