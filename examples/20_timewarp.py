#!/usr/bin/env python3
"""
Example 20: TimeWarpPE - variable-speed playback ("tape head")

Demonstrates TimeWarpPE for resampling a source at fixed and time-varying rates.

- Original demo: plays the spoken sample at original speed
- Fixed-rate demo: plays a spoken sample at 1.5x speed
- Accelerating demo: loops a spoken sample and accelerates rate from 0.25x to 5.0x
  over 10 seconds

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path

from pygmu2 import (
    AudioRenderer,
    CropPE,
    Extent,
    GainPE,
    LoopPE,
    PiecewisePE,
    TimeWarpPE,
    WavReaderPE,
    seconds_to_samples,
)


AUDIO_DIR = Path(__file__).parent / "audio"
SPOKEN_PATH = AUDIO_DIR / "spoken_voice.wav"
DRUMS_PATH = AUDIO_DIR / "djembe.wav"


def demo_original():
    """
    Play a spoken sample at original speed.
    """
    print("=== Demo: Original ===")
    print(f"Source: {SPOKEN_PATH.name}")
    print()

    spoken_stream = WavReaderPE(str(SPOKEN_PATH))
    sample_rate = spoken_stream.file_sample_rate

    output_stream = GainPE(spoken_stream, gain=0.8)

    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(output_stream)

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

    spoken_stream = WavReaderPE(str(SPOKEN_PATH))
    sample_rate = spoken_stream.file_sample_rate

    warped_stream = TimeWarpPE(spoken_stream, rate=1.5)
    output_stream = GainPE(warped_stream, gain=0.8)

    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(output_stream)

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

    spoken_stream = WavReaderPE(str(SPOKEN_PATH))
    sample_rate = spoken_stream.file_sample_rate

    # Loop the entire sample forever (with a small crossfade to reduce clicks)
    looped_stream = LoopPE(spoken_stream, crossfade_seconds=0.01)

    dur_samples = int(seconds_to_samples(10.0, sample_rate))

    # Rate ramp is a PE, so TimeWarpPE's extent becomes finite (matches the rate extent).
    rate_stream = PiecewisePE([(0, 0.25), (dur_samples, 5.0)])

    warped_stream = TimeWarpPE(looped_stream, rate=rate_stream)
    output_stream = GainPE(warped_stream, gain=0.8)
    output_stream = CropPE(output_stream, Extent(0, dur_samples))

    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(output_stream)

    with renderer:
        renderer.start()
        renderer.play_extent()


def demo_jog_shuttle():
    """
    Play at decreasing speeds, eventually going negative.
    """
    print("=== Demo: Decelerating Playback ===")
    print(f"Source: {DRUMS_PATH.name}")
    print()

    drums_stream = WavReaderPE(str(DRUMS_PATH))
    sample_rate = drums_stream.file_sample_rate

    # Rate ramp is a PE, so TimeWarpPE's extent becomes finite (matches the rate extent).
    demo_length = int(seconds_to_samples(10, sample_rate))
    rate_stream = PiecewisePE([(0, 2.0), (demo_length, -2.0)])

    warped_stream = TimeWarpPE(LoopPE(drums_stream), rate=rate_stream)
    output_stream = GainPE(warped_stream, gain=0.8)
    output_stream = CropPE(output_stream, Extent(0, demo_length))

    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(output_stream)

    with renderer:
        renderer.start()
        renderer.play_extent()

if __name__ == "__main__":
    import sys

    demos = [
        ("1", "Original", demo_original),
        ("2", "Fixed Rate (1.5x)", demo_fixed_rate),
        ("3", "Accelerating Loop (0.25x -> 5.0x over 10s)", demo_accelerating_loop),
        ("4", "Decelerating Playback", demo_jog_shuttle),
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
        choice = input(f"Choose a demo (1-{len(demos)-1} or 'a' for all): ").strip().lower()
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

