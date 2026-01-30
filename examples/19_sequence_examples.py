#!/usr/bin/env python3
"""
Example 19: Sequencing with MixPE, CropPE, SlicePE, DelayPE, RampPE

Demonstrates techniques for sequencing sounds and data streams without
SequencePE: gapless, staccato, legato, and crossfaded (ramped) note sequences
using only MixPE, CropPE, DelayPE, RampPE, and GainPE.

Sections:
  1. Sequencing notes of a C major chord (BlitSawPE) — implemented
  2. Sequencing of .wav sources — TBD
  3. Sequencing of control streams (e.g. pitch) — TBD

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pygmu2 import (
    AudioRenderer,
    BlitSawPE,
    ConstantPE,
    CropPE,
    DelayPE,
    Extent,
    ExtendMode,
    GainPE,
    MixPE,
    RampPE,
    RampType,
    pitch_to_freq,
)

SAMPLE_RATE = 44100

# C major chord: C4, E4, G4 (MIDI)
C4, E4, G4 = 60, 64, 67
C_MAJOR = [C4, E4, G4]


def _make_note(midi: int, amplitude: float = 0.25):
    """One note as BlitSawPE at given MIDI pitch."""
    return BlitSawPE(frequency=pitch_to_freq(midi), amplitude=amplitude)


def _play(pe, duration_samples: int, start: int = 0) -> None:
    """Play pe for duration_samples. If start != 0, crop from start to start + duration_samples (e.g. start=-N for lead-in)."""
    out = CropPE(pe, Extent(start, start + duration_samples))
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(out)
    with renderer:
        renderer.start()
        renderer.play_extent()


# -----------------------------------------------------------------------------
# 1. Sequencing notes of a C major chord (synthesized)
# -----------------------------------------------------------------------------


def demo_gapless_c_major():
    """
    Gapless: C then E then G with no overlap and no gap.

    Technique: Crop each note to its duration, delay E and G by cumulative
    start times, then mix. Only one note is non-zero at any time.
    """
    print("=== C major chord: gapless (C → E → G) ===")
    note_duration = int(1.0 * SAMPLE_RATE)  # 1 second per note
    total = len(C_MAJOR) * note_duration

    notes = []
    for i, midi_pitch in enumerate(C_MAJOR):
        note = _make_note(midi_pitch)
        delayed = DelayPE(
            CropPE(note, Extent(0, note_duration)),
            delay=note_duration * i,
        )
        notes.append(delayed)

    mix = MixPE(*notes)
    _play(mix, total)


def demo_staccato_c_major():
    """
    Staccato: C <silence> E <silence> G.

    Short notes with gaps between. Crop each note to a short duration,
    delay each by its slot start (note_len + gap), then mix.
    """
    print("=== C major chord: staccato (C · E · G) ===")
    note_len = int(0.5 * SAMPLE_RATE)   # 0.5 s note
    gap_len = int(0.2 * SAMPLE_RATE)   # 0.5 s gap
    slot = note_len + gap_len
    total = len(C_MAJOR) * slot

    notes = []
    for i, midi_pitch in enumerate(C_MAJOR):
        note = _make_note(midi_pitch)
        delayed = DelayPE(
            CropPE(note, Extent(0, note_len)),
            delay=slot * i,
        )
        notes.append(delayed)

    mix = MixPE(*notes)
    _play(mix, total)


def demo_legato_c_major():
    """
    Legato: C overlaps partly with E overlaps partly with G.

    Each note starts before the previous ends. Delay each note by its onset,
    no cropping of tails — mix overlaps.
    """
    print("=== C major chord: legato (C overlap E overlap G) ===")
    onset_interval = int(0.5 * SAMPLE_RATE)  # next note starts 0.5 s after previous
    note_duration = int(1.5 * SAMPLE_RATE)   # each note long enough to overlap
    total = int(1.5 * SAMPLE_RATE)           # total output length

    notes = []
    for i, midi_pitch in enumerate(C_MAJOR):
        note = _make_note(midi_pitch)
        delayed = DelayPE(
            CropPE(note, Extent(0, note_duration)),
            delay=onset_interval * i,
        )
        notes.append(delayed)

    mix = MixPE(*notes)
    # Reduce gain when overlapping to avoid clipping
    mix = GainPE(mix, gain=0.5)
    _play(mix, mix.extent().duration)


def _ramped_c_major(ramp_type: str, title: str):
    """
    Shared ramped C major: C crossfades with E crossfades with G.

    Each note lasts note_duration seconds (1.0), each crossfade xfade seconds (0.25).
    Per note: ramp up xfade s, hold (note_duration - xfade) s, ramp down xfade s.
    Same delay and gain formula for every note; ramp_type controls crossfade curve.
    """
    print(title)
    note_duration_sec = 1.0
    xfade_sec = 0.25
    note_duration = int(note_duration_sec * SAMPLE_RATE)
    xfade = int(xfade_sec * SAMPLE_RATE)
    xfade_half = xfade // 2
    d = note_duration

    notes = []
    for i, midi_pitch in enumerate(C_MAJOR):
        note = _make_note(midi_pitch)
        cropped = CropPE(note, Extent(0, note_duration + xfade))
        delayed = DelayPE(cropped, delay=i * d - xfade_half)

        gain = MixPE(
            DelayPE(RampPE(0.0, 1.0, xfade, ramp_type=ramp_type), i * d - xfade_half),
            CropPE(
                ConstantPE(1.0),
                Extent(i * d + xfade_half, (i + 1) * d - xfade_half),
            ),
            DelayPE(RampPE(1.0, 0.0, xfade, ramp_type=ramp_type), (i + 1) * d - xfade_half),
        )

        notes.append(GainPE(delayed, gain=gain))

    mix = MixPE(*notes)
    ext = mix.extent()
    _play(mix, ext.duration, start=ext.start)


def demo_constant_power_ramp():
    """
    Ramped C major using CONSTANT_POWER crossfades (sin/cos).

    Perceived level stays even during crossfades; good for A/B comparison with exponential.
    """
    _ramped_c_major(
        RampType.CONSTANT_POWER,
        "=== C major: ramped with CONSTANT_POWER crossfades (C ⟷ E ⟷ G) ===",
    )


def demo_exponential_ramp():
    """
    Ramped C major using EXPONENTIAL crossfades.

    Exponential curves can sound more natural for some material; compare with constant-power.
    """
    _ramped_c_major(
        RampType.EXPONENTIAL,
        "=== C major: ramped with EXPONENTIAL crossfades (C ⟷ E ⟷ G) ===",
    )


# -----------------------------------------------------------------------------
# 2. Sequencing of .wav sources — TBD
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 3. Sequencing of control streams (e.g. pitch) — TBD
# -----------------------------------------------------------------------------

# ===== MIDI notes
Cm1, Dfm1, Dm1, Efm1, Em1, Fm1, Gfm1, Gm1, Afm1, Am1, Bfm1, Bm1 = range(0, 12)
C0, Df0, D0, Ef0, E0, F0, Gf0, G0, Af0, A0, Bf0, B0 = range(12, 24)
C1, Df1, D1, Ef1, E1, F1, Gf1, G1, Af1, A1, Bf1, B1 = range(24, 36)
C2, Df2, D2, Ef2, E2, F2, Gf2, G2, Af2, A2, Bf2, B2 = range(36, 48)
C3, Df3, D3, Ef3, E3, F3, Gf3, G3, Af3, A3, Bf3, B3 = range(48, 60)
C4, Df4, D4, Ef4, E4, F4, Gf4, G4, Af4, A4, Bf4, B4 = range(60, 72)
C5, Df5, D5, Ef5, E5, F5, Gf5, G5, Af5, A5, Bf5, B5 = range(72, 84)
C6, Df6, D6, Ef6, E6, F6, Gf6, G6, Af6, A6, Bf6, B6 = range(84, 96)

# Sharp equivalents
Asm1, Csm1, Dsm1, Fsm1, Gsm1 = [Bfm1, Dfm1, Efm1, Gfm1, Afm1]
As0, Cs0, Ds0, Fs0, Gs0 = [Bf0, Df0, Ef0, Gf0, Af0]
As1, Cs1, Ds1, Fs1, Gs1 = [Bf1, Df1, Ef1, Gf1, Af1]
As2, Cs2, Ds2, Fs2, Gs2 = [Bf2, Df2, Ef2, Gf2, Af2]
As3, Cs3, Ds3, Fs3, Gs3 = [Bf3, Df3, Ef3, Gf3, Af3]
As4, Cs4, Ds4, Fs4, Gs4 = [Bf4, Df4, Ef4, Gf4, Af4]
As5, Cs5, Ds5, Fs5, Gs5 = [Bf5, Df5, Ef5, Gf5, Af5]
As6, Cs6, Ds6, Fs6, Gs6 = [Bf6, Df6, Ef6, Gf6, Af6]

REST = -1 # an out-of-band value

# ===== Durations, expressed as beats
WHOLE = 4.0
HALF = 2.0
QUARTER = 1.0
EIGHTH = 0.5
SIXTEENTH = 0.25
THIRTY_SECOND = 0.125
DOTTED = 1.5           # multiplicative modifier
TRIPLET = (2.0/3.0)    # multiplicative modifier

# ===== Articulation, scales duration

LEGATO = 1.2     # mild overlap
CONNECTED = 1.0  # notes directly abut one another, no overlap
DETACHED = 0.7   # slight space between notes
STACCATO = 0.5   # shortened notes

moz_k333 = [
    # midi note, duration, expression
    (F5, QUARTER*DOTTED, CONNECTED),
    (D5, EIGHTH, CONNECTED),
    (Bf4, QUARTER, DETACHED),
    (Bf4, QUARTER, DETACHED),
    # 10
    (Ef5, EIGHTH, CONNECTED),
    (F5, THIRTY_SECOND, CONNECTED),
    (Ef5, THIRTY_SECOND, CONNECTED),
    (D5, THIRTY_SECOND, CONNECTED),
    (Ef5, THIRTY_SECOND, CONNECTED),
    (G5, QUARTER, DETACHED),
    (A4, QUARTER, DETACHED),
    (Rest, EIGHTH, DETACHED),
    (A4, EIGHTH, DETACHED),
    # 11
    (C5, EIGHTH, CONNECTED),
    (Bf4, EIGHTH, DETACHED),
    (D5, EIGHTH, CONNECTED),
    (C5, EIGHTH, CONNECTED),
    (Ef5, EIGHTH, CONNECTED),
    (D5, EIGHTH, DETACHED),
    (F5, EIGHTH, CONNECTED),
    (Ef5, EIGHTH, DETACHED),
    # 12
    (D5, QUARTER*DOTTED, CONNECTED),
    (Ef5, SIXTEENTH, CONNECTED),
    (D5, SIXTEENTH, CONNECTED),
    (C5, EIGHTH, CONNECTED),
    (D5, EIGHTH, CONNECTED),
    (Ef5, EIGHTH, CONNECTED),
    (E5, EIGHTH, DETACHED),
    # 13
    (F5, QUARTER*DOTTED, CONNECTED),
    (Ef5, SIXTEENTH*TRIPLET, CONNECTED),
    (D5, SIXTEENTH*TRIPLET, CONNECTED),
    (C5, SIXTEENTH*TRIPLET, DETACHED),
    (Bf4, EIGHTH, STACCATO),
    (Bf4, EIGHTH, STACCATO),
    (Bf4, EIGHTH, STACCATO),
    (Bf4, EIGHTH, STACCATO),
    # 14        
    (Ef5, EIGHTH, CONNECTED),
    (F5, THIRTY_SECOND, CONNECTED),
    (Ef5, THIRTY_SECOND, CONNECTED),
    (D5, THIRTY_SECOND, CONNECTED),
    (Ef5, THIRTY_SECOND, CONNECTED),
    (G5, QUARTER, DETACHED),
    (A4, QUARTER, DETACHED),
    (Rest, EIGHTH, DETACHED),
    (A5, EIGHTH, DETACHED),
    # 15
    (Bf5, EIGHTH*TRIPLET, CONNECTED),
    (F5, EIGHTH*TRIPLET, CONNECTED),
    (D5, EIGHTH*TRIPLET, CONNECTED),
    (G5, EIGHTH*TRIPLET, CONNECTED),
    (Ef5, EIGHTH*TRIPLET, CONNECTED),
    (C5, EIGHTH*TRIPLET, CONNECTED),
    (F5, EIGHTH*TRIPLET, CONNECTED),
    (D5, EIGHTH*TRIPLET, CONNECTED),
    (Bf4, EIGHTH*TRIPLET, CONNECTED),
    (Ef5, EIGHTH*TRIPLET, CONNECTED),
    (C5, EIGHTH*TRIPLET, CONNECTED),
    (A4, EIGHTH*TRIPLET, DETACHED),
    # 16
    (Bf4, WHOLE, CONNECTED),
]


def demo_all_c_major():
    demo_gapless_c_major()
    demo_staccato_c_major()
    demo_legato_c_major()
    demo_constant_power_ramp()
    demo_exponential_ramp()


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "C major: gapless (C → E → G)", demo_gapless_c_major),
        ("2", "C major: staccato (C · E · G)", demo_staccato_c_major),
        ("3", "C major: legato (overlapping)", demo_legato_c_major),
        ("4", "C major: ramped crossfades — CONSTANT_POWER", demo_constant_power_ramp),
        ("5", "C major: ramped crossfades — EXPONENTIAL", demo_exponential_ramp),
        ("a", "All C major demos", demo_all_c_major),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("Sequence examples (MixPE, CropPE, DelayPE, RampPE)")
        print("---------------------------------------------------")
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
