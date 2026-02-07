"""
Example 15: Reverse Pitch Echo - block-based reverse playback

Demonstrates ReversePitchEchoPE with pitch shift and feedback.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import (
    AudioRenderer,
    CropPE,
    Extent,
    GainPE,
    MixPE,
    ReversePitchEchoPE,
    WavReaderPE,
)
import pygmu2 as pg
pg.set_sample_rate(44100)


AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "spoken_voice.wav"

DURATION_SECONDS = 8

print("=== pygmu2 Example 15: Reverse Pitch Echo ===", flush=True)
print(f"Loading: {WAV_FILE}", flush=True)

source_stream = WavReaderPE(str(WAV_FILE))
sample_rate = source_stream.file_sample_rate or 44100
duration_samples = int(DURATION_SECONDS * sample_rate)

# --- Part 1: Dry ---
print(f"\nPart 1: Dry signal - {DURATION_SECONDS}s", flush=True)
dry_stream = CropPE(source_stream, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(dry_stream)

with renderer:
    renderer.start()
    renderer.play_extent()

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
wet_out_stream = CropPE(wet_stream, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(wet_out_stream)

with renderer:
    renderer.start()
    renderer.play_extent()

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
mixed_out_stream = CropPE(mixed_stream, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(mixed_out_stream)

with renderer:
    renderer.start()
    renderer.play_extent()

print("\nDone!", flush=True)
