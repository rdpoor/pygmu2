#!/usr/bin/env python3
"""
Example 20: TimeWarpPE - variable-speed playback ("tape head")

Demonstrates TimeWarpPE for resampling a source at fixed and time-varying rates.

- Original demo: plays the spoken sample at original speed
- Fixed-rate demo: plays a spoken sample at 1.5x speed
- Accelerating demo: loops a spoken sample and accelerates rate from 0.25x to 5.0x
  over 10 seconds

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pathlib import Path

from pygmu2 import (
    AudioRenderer,
    CropPE,
    Extent,
    GainPE,
    LoopPE,
    RampPE,
    TimeWarpPE,
    WavReaderPE,
    seconds_to_samples,
)


AUDIO_DIR = Path(__file__).parent / "audio"
SPOKEN_PATH = AUDIO_DIR / "spoken_voice.wav"


def demo_original():
    """
    Play a spoken sample at original speed.
    """
    print("=== Demo: Original ===")
    print(f"Source: {SPOKEN_PATH.name}")
    print()

    spoken = WavReaderPE(str(SPOKEN_PATH))
    sample_rate = spoken.file_sample_rate

    output = GainPE(spoken, gain=0.8)

    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(output)

    with renderer:
        renderer.start()
        renderer.play_extent()


def demo_fixed_rate():
    """
    Play a spoken sample at a fixed rate (1.5x).
    """
    print("=== Demo: Fixed Rate (1.5x) ===")
    print(f"Source: {SPOKEN_PATH.name}")
    print()

    spoken = WavReaderPE(str(SPOKEN_PATH))
    sample_rate = spoken.file_sample_rate

    warped = TimeWarpPE(spoken, rate=1.5)
    output = GainPE(warped, gain=0.8)

    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(output)

    with renderer:
        renderer.start()
        renderer.play_extent()


def demo_accelerating_loop():
    """
    Loop the spoken sample and accelerate from 0.5x to 5.0x over 10 seconds.
    """
    print("=== Demo: Accelerating Loop (0.25x -> 5.0x over 10s) ===")
    print(f"Source: {SPOKEN_PATH.name}")
    print()

    spoken = WavReaderPE(str(SPOKEN_PATH))
    sample_rate = spoken.file_sample_rate

    # Loop the entire sample forever (with a small crossfade to reduce clicks)
    looped = LoopPE(spoken, crossfade_seconds=0.01)

    dur_samples = int(seconds_to_samples(10.0, sample_rate))

    # Rate ramp is a PE, so TimeWarpPE's extent becomes finite (matches the rate extent).
    rate = RampPE(0.25, 5.0, duration=dur_samples)

    warped = TimeWarpPE(looped, rate=rate)
    output = GainPE(warped, gain=0.8)
    output = CropPE(output, Extent(0, dur_samples))

    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(output)

    with renderer:
        renderer.start()
        renderer.play_extent()


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "Original", demo_original),
        ("2", "Fixed Rate (1.5x)", demo_fixed_rate),
        ("3", "Accelerating Loop (0.25x -> 5.0x over 10s)", demo_accelerating_loop),
        ("a", "All demos", None),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("pygmu2 TimeWarpPE Examples")
        print("-------------------------")
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

