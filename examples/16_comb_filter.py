"""
Example 14: Comb Filter - pitched resonance

Demonstrates CombPE tuned to different frequencies.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import (
import pygmu2 as pg
pg.set_sample_rate(44100)

    AudioRenderer,
    CombPE,
    CropPE,
    Extent,
    GainPE,
    WavReaderPE,
)

AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "djembe.wav"

DURATION_SECONDS = 6

print("=== pygmu2 Example 14: Comb Filter ===", flush=True)
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

# --- Part 2: Comb tuned to 220 Hz ---
print("\nPart 2: Comb filter (220 Hz, feedback 0.7)", flush=True)
comb_220_stream = CombPE(source_stream, frequency=220.0, feedback=0.7)
comb_220_stream = GainPE(comb_220_stream, gain=0.7)
comb_220_out_stream = CropPE(comb_220_stream, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(comb_220_out_stream)

with renderer:
    renderer.start()
    renderer.play_extent()

# --- Part 3: Comb tuned to 440 Hz ---
print("\nPart 3: Comb filter (440 Hz, feedback 0.9)", flush=True)
comb_440_stream = CombPE(source_stream, frequency=440.0, feedback=0.9)
comb_440_stream = GainPE(comb_440_stream, gain=0.7)
comb_440_out_stream = CropPE(comb_440_stream, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(comb_440_out_stream)

with renderer:
    renderer.start()
    renderer.play_extent()

print("\nDone!", flush=True)
