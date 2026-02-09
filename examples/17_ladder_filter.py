"""
Example 13: Ladder Filter - Moog-style ladder responses

Demonstrates LadderPE with resonance, drive, and a cutoff sweep.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import (
    CropPE,
    GainPE,
    LadderPE,
    LadderMode,
    PiecewisePE,
    WavReaderPE,
)
import pygmu2 as pg
pg.set_sample_rate(44100)


AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "choir.wav"

DURATION_SECONDS = 8

print("=== pygmu2 Example 13: Ladder Filter ===", flush=True)
print(f"Loading: {WAV_FILE}", flush=True)

source_stream = WavReaderPE(str(WAV_FILE))
sample_rate = source_stream.file_sample_rate or 44100
duration_samples = int(DURATION_SECONDS * sample_rate)

# --- Part 1: Dry ---
print(f"\nPart 1: Dry signal - {DURATION_SECONDS}s", flush=True)
dry_stream = CropPE(source_stream, 0, (duration_samples) - (0))

pg.play(dry_stream, sample_rate)

# --- Part 2: Resonant lowpass ---
print("\nPart 2: Ladder lowpass (800 Hz, resonance 0.6)", flush=True)
lowpass_stream = LadderPE(source_stream, frequency=800.0, resonance=0.6, mode=LadderMode.LP24, drive=1.5)
lowpass_stream = GainPE(lowpass_stream, gain=0.8)
lowpass_out_stream = CropPE(lowpass_stream, 0, (duration_samples) - (0))

pg.play(lowpass_out_stream, sample_rate)

# --- Part 3: Cutoff sweep ---
print("\nPart 3: Ladder sweep (200 Hz -> 4 kHz)", flush=True)
cutoff_sweep_stream = PiecewisePE([(0, 200.0), (duration_samples, 4000.0)])
sweep_stream = LadderPE(source_stream, frequency=cutoff_sweep_stream, resonance=0.3, mode=LadderMode.LP12, drive=1.2)
sweep_stream = GainPE(sweep_stream, gain=0.8)
sweep_out_stream = CropPE(sweep_stream, 0, (duration_samples) - (0))

pg.play(sweep_out_stream, sample_rate)

print("\nDone!", flush=True)
