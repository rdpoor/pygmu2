"""
Example 15: Reverse Pitch Echo - block-based reverse playback

Demonstrates ReversePitchEchoPE with pitch shift and feedback.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import (
    CropPE,
    GainPE,
    MixPE,
    ReversePitchEchoPE,
    WavReaderPE,
)
import pygmu2 as pg
from examples_helper import run_demos
pg.set_sample_rate(44100)


AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "spoken_voice.wav"

DURATION_SECONDS = 8

print("=== pygmu2 Example 15: Reverse Pitch Echo ===", flush=True)
print(f"Loading: {WAV_FILE}", flush=True)

source_stream = WavReaderPE(str(WAV_FILE))
sample_rate = source_stream.file_sample_rate or 44100
# Align the global rate to the file's native rate before any downstream
# PEs are constructed (CropPE, GainPE, ReversePitchEchoPE, ...).
pg.set_sample_rate(sample_rate)
duration_samples = int(DURATION_SECONDS * sample_rate)

def original_signal():
    # --- Part 1: Dry ---
    print(f"\nPart 1: Dry signal - {DURATION_SECONDS}s", flush=True)
    dry_stream = CropPE(source_stream, 0, (duration_samples) - (0))

    pg.play(pg.GainPE(dry_stream, gain=2.14))

def wet_only():
    # --- Part 2: Wet only ---
    print("\nPart 2: Reverse pitch echo (wet only)", flush=True)
    wet_stream = ReversePitchEchoPE(
        source_stream,
        block_seconds=0.12,
        pitch_ratio=0.75,
        feedback=0.6,
        alternate_direction=1.0,
    )
    wet_stream = GainPE(wet_stream, gain=0.8)
    wet_out_stream = CropPE(wet_stream, 0, (duration_samples) - (0))

    pg.play(pg.GainPE(wet_out_stream, gain=2.28))

def wet_plus_dry():
    # --- Part 3: Dry + wet mix ---
    print("\nPart 3: Reverse pitch echo mixed with dry", flush=True)
    wet_mix_stream = ReversePitchEchoPE(
        source_stream,
        block_seconds=0.12,
        pitch_ratio=0.75,
        feedback=0.6,
        alternate_direction=1.0,
    )
    mixed_stream = MixPE(GainPE(source_stream, gain=0.5), GainPE(wet_mix_stream, gain=0.5))
    mixed_out_stream = CropPE(mixed_stream, 0, (duration_samples) - (0))

    pg.play(pg.GainPE(mixed_out_stream, gain=2.83))

DEMOS = [
    ("Original signal", original_signal),
    ("Wet only", wet_only),
    ("Wet plus dry", wet_plus_dry),
]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    run_demos(DEMOS)
