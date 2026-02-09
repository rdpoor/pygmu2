"""
Example 06: Autowah - Envelope-controlled filter

Demonstrates using EnvelopePE to control BiquadPE or SVFilterPE frequency,
creating the classic "autowah" or envelope filter effect.

The louder the input, the higher the filter frequency sweeps.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import sys
from pathlib import Path
from pygmu2 import (
    WavReaderPE,
    EnvelopePE,
    DetectionMode,
    BiquadPE,
    BiquadMode,
    SVFilterPE,
    TransformPE,
    GainPE,
    CropPE,
    LoopPE,
    Extent,
)
import pygmu2 as pg
pg.set_sample_rate(44100)


# Path to audio file
AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "djembe.wav"

DURATION_SECONDS = 8


def demo_original():
    """Part 1: Original signal (looped djembe)."""
    print(f"Part 1: Original signal (looped djembe) - {DURATION_SECONDS}s", flush=True)
    source_stream = WavReaderPE(str(WAV_FILE))
    sample_rate = source_stream.file_sample_rate
    duration_samples = int(DURATION_SECONDS * sample_rate)
    looped_stream = LoopPE(source_stream, crossfade_seconds=0.01)
    pg.play(CropPE(looped_stream, 0, duration_samples), sample_rate)


def demo_autowah():
    """Part 2: Autowah effect (envelope-controlled lowpass)."""
    print(f"Part 2: Autowah effect - {DURATION_SECONDS}s", flush=True)
    print("  Envelope follower -> frequency mapping -> lowpass filter", flush=True)

    def envelope_to_freq(env):
        """Map envelope (0-1) to frequency (100-3000 Hz)."""
        import numpy as np
        env = np.clip(env, 0, 1)
        min_freq = 100.0
        max_freq = 3000.0
        return min_freq + (max_freq - min_freq) * (env ** 0.5)

    source_stream = WavReaderPE(str(WAV_FILE))
    sample_rate = source_stream.file_sample_rate
    duration_samples = int(DURATION_SECONDS * sample_rate)
    looped_stream = LoopPE(source_stream, crossfade_seconds=0.01)

    envelope_stream = EnvelopePE(
        looped_stream,
        attack=0.005,
        release=0.05,
        mode=DetectionMode.PEAK
    )
    freq_control_stream = TransformPE(
        envelope_stream, func=envelope_to_freq, name="env_to_freq"
    )
    filtered_stream = BiquadPE(
        looped_stream,
        frequency=freq_control_stream,
        q=10.0,
        mode=BiquadMode.LOWPASS
    )
    output_stream = GainPE(filtered_stream, gain=1.0)
    pg.play(CropPE(output_stream, 0, duration_samples), sample_rate)


def demo_svfilter_autowah():
    """Autowah using SVFilterPE (state variable filter) instead of BiquadPE."""
    print(f"Part 3: Autowah with SVFilterPE - {DURATION_SECONDS}s", flush=True)
    print("  Envelope follower -> frequency mapping -> SVF lowpass", flush=True)

    def envelope_to_freq(env):
        """Map envelope (0-1) to frequency (100-3000 Hz)."""
        import numpy as np
        env = np.clip(env, 0, 1)
        min_freq = 100.0
        max_freq = 3000.0
        return min_freq + (max_freq - min_freq) * (env ** 0.5)

    source_stream = WavReaderPE(str(WAV_FILE))
    sample_rate = source_stream.file_sample_rate
    duration_samples = int(DURATION_SECONDS * sample_rate)
    looped_stream = LoopPE(source_stream, crossfade_seconds=0.01)

    envelope_stream = EnvelopePE(
        looped_stream,
        attack=0.005,
        release=0.05,
        mode=DetectionMode.PEAK
    )
    freq_control_stream = TransformPE(
        envelope_stream, func=envelope_to_freq, name="env_to_freq"
    )
    filtered_stream = SVFilterPE(
        looped_stream,
        frequency=freq_control_stream,
        q=10.0,
        mode=BiquadMode.LOWPASS
    )
    output_stream = GainPE(filtered_stream, gain=1.0)
    pg.play(CropPE(output_stream, 0, duration_samples), sample_rate)


def demo_all():
    """Run original, biquad autowah, then SVFilter autowah."""
    demo_original()
    demo_autowah()
    demo_svfilter_autowah()


if __name__ == "__main__":
    demos = [
        ("1", "Original signal (looped djembe)", demo_original),
        ("2", "Autowah effect (BiquadPE lowpass)", demo_autowah),
        ("3", "Autowah effect (SVFilterPE lowpass)", demo_svfilter_autowah),
        ("a", "All autowah demos", demo_all),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("Autowah examples (EnvelopePE, BiquadPE, SVFilterPE, TransformPE)")
        print("-----------------------------------------------------------------")
        for key, name, _ in demos:
            print(f"  {key}: {name}")
        print()
        choice = input(f"Choice (1-3 or 'a'): ").strip().lower()

    for key, _name, fn in demos:
        if key == choice:
            fn()
            break
    else:
        print("Invalid choice.")
