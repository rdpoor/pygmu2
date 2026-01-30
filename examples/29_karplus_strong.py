#!/usr/bin/env python3
"""
Example 29: Karplus-Strong plucked string synthesis

Minimal K-S: delay line (one period), white noise excitation,
feedback through two-point average with gain rho. Extent is infinite;
crop with CropPE for desired duration. Parameters: frequency, rho.

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

SAMPLE_RATE = 44100


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
    ks = KarplusStrongPE(
        frequency=440.0,
        rho=0.996,
        amplitude=0.35,
        seed=42,
    )
    _play(ks, int(1.0 * SAMPLE_RATE), 0)


def demo_high_rho_vs_low_rho():
    """Same pitch: high rho (sustains) then low rho (quick decay), with gap."""
    print("=== Karplus-Strong: high rho (0.999) vs low rho (0.98) ===")
    pluck_sec = 1.0
    gap_sec = 1.0
    pluck_samp = int(pluck_sec * SAMPLE_RATE)
    gap_samp = int(gap_sec * SAMPLE_RATE)

    high = KarplusStrongPE(
        frequency=330.0,
        rho=0.999,
        amplitude=0.35,
        seed=1,
    )
    low = KarplusStrongPE(
        frequency=330.0,
        rho=0.98,
        amplitude=0.35,
        seed=1,
    )
    mix = MixPE(
        CropPE(high, Extent(0, pluck_samp)),
        DelayPE(CropPE(low, Extent(0, pluck_samp)), delay=pluck_samp + gap_samp),
    )
    _play(mix, 2 * pluck_samp + gap_samp)


def demo_c_major_arpeggio():
    """C major arpeggio (C4, E4, G4, C5) as plucks."""
    print("=== Karplus-Strong: C major arpeggio (C4 E4 G4 C5) ===")
    midi_notes = [60, 64, 67, 72]
    pluck_sec = 0.8
    gap_sec = 0.2
    pluck_samp = int(pluck_sec * SAMPLE_RATE)
    slot_samp = pluck_samp + int(gap_sec * SAMPLE_RATE)

    plucks = []
    for i, midi in enumerate(midi_notes):
        ks = KarplusStrongPE(
            frequency=pitch_to_freq(midi),
            rho=0.996,
            amplitude=0.3,
            seed=10 + i,
        )
        plucks.append(DelayPE(CropPE(ks, Extent(0, pluck_samp)), delay=i * slot_samp))

    mix = MixPE(*plucks)
    _play(mix, len(midi_notes) * slot_samp)


def demo_all():
    demo_single_pluck()
    demo_high_rho_vs_low_rho()
    demo_c_major_arpeggio()


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "Single pluck (440 Hz)", demo_single_pluck),
        ("2", "High rho vs low rho (same pitch)", demo_high_rho_vs_low_rho),
        ("3", "C major arpeggio", demo_c_major_arpeggio),
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
        choice = input("Choice (1-3 or 'a'): ").strip().lower()

    for key, _name, fn in demos:
        if key == choice:
            fn()
            break
    else:
        print("Invalid choice.")
