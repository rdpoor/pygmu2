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
SR = 44100


# ---------------------------------------------------------------------------
# Note and arpeggio builders
# ---------------------------------------------------------------------------


def _make_arpeggio(instrument, midi_notes, note_spacing, amplitude=0.3):
    """
    Sequential arpeggio: one note every note_spacing seconds.
    Each note rings to its natural -60 dB extent.  When note_spacing is
    shorter than a note's decay, adjacent notes overlap naturally.
    Returns a MixPE spanning the full duration.
    """
    notes = []
    t = 0.0
    for midi in midi_notes:
        note = IdiophonePE(
            instrument, frequency=pitch_to_freq(midi), amplitude=amplitude
        )
        notes.append(DelayPE(note, int(round(t * SR))))
        t += note_spacing
    return MixPE(*notes)


def _make_low_high(instrument, midi_low, midi_high, amplitude):
    """Low note followed by high note, no overlap (waits for low to decay to -60 dB)."""
    low = IdiophonePE(
        instrument, frequency=pitch_to_freq(midi_low), amplitude=amplitude
    )
    high = DelayPE(
        IdiophonePE(
            instrument, frequency=pitch_to_freq(midi_high), amplitude=amplitude
        ),
        low.extent().duration,
    )
    return MixPE(low, high)


# ---------------------------------------------------------------------------
# Scale patterns
# ---------------------------------------------------------------------------

# C diatonic across two octaves: C3 – C5
C_DIATONIC_2OCT = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72]

# C pentatonic across two octaves: C3 – C5
C_PENTATONIC_2OCT = [48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72]


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_marimba_low_high():
    """Marimba: lowest note (C3) then highest note (C5), sequential."""
    pg.play(_make_low_high(MARIMBA, midi_low=48, midi_high=72, amplitude=0.25))


def demo_marimba_arpeggio():
    """
    Marimba: C diatonic, C3–C5.
    Slow spacing lets each note's fundamental decay be audible.
    Notes overlap naturally when decay exceeds note_spacing.
    """
    print("=== Marimba: C diatonic arpeggio (C3–C5) ===")
    mix = _make_arpeggio(
        MARIMBA,
        midi_notes=C_DIATONIC_2OCT,
        note_spacing=0.12,
        amplitude=0.2,
    )
    pg.play(mix)


def demo_xylophone_low_high():
    """Xylophone: lowest note (C4) then highest note (C6), sequential."""
    midi_notes = [m + 12 for m in [48, 72]]
    pg.play(
        _make_low_high(
            XYLOPHONE, midi_low=midi_notes[0], midi_high=midi_notes[1], amplitude=0.25
        )
    )


def demo_xylophone_arpeggio():
    """
    Xylophone: C diatonic, C4–C6.
    Short decay (tau_mid = 0.3 s) suits faster spacing; notes stay crisp.
    """
    print("=== Xylophone: C diatonic arpeggio (C4–C6) ===")
    midi_notes = [m + 12 for m in C_DIATONIC_2OCT]  # shift up one octave
    mix = _make_arpeggio(
        XYLOPHONE,
        midi_notes=midi_notes,
        note_spacing=0.12,
        amplitude=0.2,
    )
    pg.play(mix)


def demo_glockenspiel_low_high():
    """Glockenspiel: lowest note (C5) then highest note (C7), sequential."""
    midi_notes = [m + 24 for m in [48, 72]]
    pg.play(
        _make_low_high(
            GLOCKENSPIEL,
            midi_low=midi_notes[0],
            midi_high=midi_notes[1],
            amplitude=0.25,
        )
    )


def demo_glockenspiel_arpeggio():
    """
    Glockenspiel: C diatonic, C5–C7.
    Long tau_mid (3 s) and inharmonic overtones (2.756 f0) give the
    characteristic bright shimmer; notes overlap heavily.
    """
    print("=== Glockenspiel: C diatonic arpeggio (C5–C7) ===")
    midi_notes = [m + 24 for m in C_DIATONIC_2OCT]  # shift up two octaves
    mix = _make_arpeggio(
        GLOCKENSPIEL,
        midi_notes=midi_notes,
        note_spacing=0.12,
        amplitude=0.2,
    )
    pg.play_offline(mix)


def demo_balafon_low_high():
    """Balafon: lowest note (C3) then highest note (C5), sequential."""
    pg.play(_make_low_high(BALAFON, midi_low=48, midi_high=72, amplitude=0.25))


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
        amplitude=0.2,
    )
    pg.play(mix)


# ---------------------------------------------------------------------------
# Demo registry
# ---------------------------------------------------------------------------

DEMOS = [
    ("Marimba: low→high", demo_marimba_low_high),
    ("Marimba arpeggio (C diatonic, C3–C5)", demo_marimba_arpeggio),
    ("Xylophone: low→high", demo_xylophone_low_high),
    ("Xylophone arpeggio (C diatonic, C4–C6)", demo_xylophone_arpeggio),
    ("Glockenspiel: low→high", demo_glockenspiel_low_high),
    ("Glockenspiel arpeggio (C diatonic, C5–C7)", demo_glockenspiel_arpeggio),
    ("Balafon: low→high", demo_balafon_low_high),
    ("Balafon arpeggio (C pentatonic, C3–C5)", demo_balafon_arpeggio),
]

if __name__ == "__main__":
    run_demos(DEMOS)
