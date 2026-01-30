#!/usr/bin/env python3
"""
Example 29: Karplus-Strong plucked string synthesis

Delay line (one period), white noise excitation, feedback through
two-point average with gain rho. Extent is infinite; crop with CropPE
for desired duration. Parameters: frequency, rho.

Optional two-phase decay: pass duration and rho_damping to sustain
with rho for duration samples, then fade faster with rho_damping.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pygmu2 import (
    AudioRenderer,
    CropPE,
    DelayPE,
    Extent,
    KarplusStrongPE,
    MixPE,
    pitch_to_freq,
)
from pygmu2.karplus_strong_pe import rho_for_decay_db
from typing import Optional


SAMPLE_RATE = 44100

def s2s(seconds):
    return int(round(seconds * SAMPLE_RATE))

def _make_note(
    frequency: float = 440.0,
    duration_seconds: float = 1.0,
    amplitude: float = 0.5,
    seed: Optional[int] = None,
    channels: int = 1
    ):
    """A thin wrapper around KarplusStrongPE with a duration parameter"""
    rho=rho_for_decay_db(duration_seconds, frequency, SAMPLE_RATE, db=-30)
    print(f'duration={duration_seconds} => rho={rho}')
    return CropPE(
            KarplusStrongPE(
                frequency=frequency,
                rho=rho,
                amplitude=amplitude,
                seed=1,
            ),
            Extent(0, s2s(duration_seconds))
        )

def _play(pe, duration_samples: int, start: int = 0) -> None:
    out = CropPE(pe, Extent(start, start + duration_samples))
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(out)
    with renderer:
        renderer.start()
        renderer.play_extent()


def demo_single_pluck():
    """One 440 Hz pluck, 1 s, rho=0.996 (slow decay)."""
    print("=== Karplus-Strong: single pluck (440 Hz, 1 s, rho=0.996) ===")
    duration_seconds = 1.0
    ks = _make_note(
        frequency=440.0,
        duration_seconds = duration_seconds,
        amplitude=0.35,
        seed=42,
    )
    _play(ks, s2s(duration_seconds))


def demo_two_phase_decay():
    """One pluck: sustain (rho=0.996) then faster fade (rho_damping=0.94)."""
    print("=== Karplus-Strong: two-phase decay (sustain 0.5, then fade) ===")
    sustain_sec = 1.0
    total_sec = 2.5
    sustain_samp = int(sustain_sec * SAMPLE_RATE)
    total_samp = int(total_sec * SAMPLE_RATE)
    ks = KarplusStrongPE(
        frequency=440.0,
        rho=0.999,
        duration=sustain_samp,
        rho_damping=0.93,
        amplitude=0.35,
        seed=42,
    )
    _play(ks, total_samp, 0)


def demo_high_rho_vs_low_rho():
    """Same pitch: high rho (sustains) then low rho (quick decay), with gap."""
    print("=== Karplus-Strong: high rho (0.995) vs low rho (0.95) ===")

    plucks = []
    t = 0


    frequency: float = 440.0,
    duration_seconds: float = 1.0,
    amplitude: float = 0.5,
    seed: Optional[int] = None,
    channels: int = 1


    duration_seconds = 2.0
    plucks.append(
        DelayPE(
            _make_note(
                frequency=330.0, duration_seconds=duration_seconds), s2s(t)))
    t += duration_seconds

    duration_seconds = 0.25
    plucks.append(
        DelayPE(
            _make_note(
                frequency=330.0, duration_seconds=duration_seconds), s2s(t)))
    t += duration_seconds

    mix = MixPE(*plucks)
    _play(mix, s2s(t))


def demo_c_major_arpeggio():
    """C major arpeggio (C4, E4, G4, C5) as plucks."""
    print("=== Karplus-Strong: C major arpeggio (C4 E4 G4 C5) ===")
    midi_notes = [60, 64, 67, 72]
    duration_seconds = 0.8

    plucks = []
    t = 0
    for _, midi in enumerate(midi_notes):
        plucks.append(
            DelayPE(
                _make_note(
                    frequency=pitch_to_freq(midi), 
                    duration_seconds=duration_seconds), 
                s2s(t)))
        t += duration_seconds

    mix = MixPE(*plucks)
    _play(mix, s2s(t))


def demo_all():
    demo_single_pluck()
    demo_two_phase_decay()
    demo_high_rho_vs_low_rho()
    demo_c_major_arpeggio()


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "Single pluck (440 Hz)", demo_single_pluck),
        ("2", "Two-phase decay (sustain then fade)", demo_two_phase_decay),
        ("3", "High rho vs low rho (same pitch)", demo_high_rho_vs_low_rho),
        ("4", "C major arpeggio", demo_c_major_arpeggio),
        ("a", "All demos", demo_all),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("Example 29: Karplus-Strong plucked string")
        print("----------------------------------------")
        for key, name, _ in demos:
            print(f"  {key}: {name}")
        print()
        choice = input("Choice (1-4 or 'a'): ").strip().lower()

    for key, _name, fn in demos:
        if key == choice:
            fn()
            break
    else:
        print("Invalid choice.")
