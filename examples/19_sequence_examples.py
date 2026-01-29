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
