"""
Example 04: Filtering - Biquad filter with frequency sweep

Demonstrates BiquadPE with a sweeping cutoff frequency,
creating a classic filter sweep effect.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import (
    WavReaderPE,
    BiquadPE,
    BiquadMode,
    RampPE,
    CropPE,
    AudioRenderer,
    Extent,
)

# Path to audio file
AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "faun.wav"

DURATION_SECONDS = 8

LO_FREQUENCY = 100.0
HI_FREQUENCY = 2500.0
Q = 8.0

print("=== pygmu2 Example 04: Filtering ===", flush=True)
print(f"Loading: {WAV_FILE}", flush=True)

# Load orchestral source
source = WavReaderPE(str(WAV_FILE))
sample_rate = source.file_sample_rate
duration_samples = int(DURATION_SECONDS * sample_rate)

# --- Part 1: Lowpass filter sweep (low to high) ---
print(f"\nPart 1: Lowpass sweep {LO_FREQUENCY}Hz -> {HI_FREQUENCY}Hz - {DURATION_SECONDS}s, Q={Q}", flush=True)

# Create frequency sweep from 200Hz to 8000Hz
freq_sweep_up = RampPE(LO_FREQUENCY, HI_FREQUENCY, duration=duration_samples)

# Apply lowpass filter with sweeping frequency
filtered_up = BiquadPE(
    source,
    frequency=freq_sweep_up,
    q=Q,
    mode=BiquadMode.LOWPASS
)
output_up = CropPE(filtered_up, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(output_up)

with renderer:
    renderer.start()
    renderer.play_extent()

# --- Part 2: Lowpass filter sweep (high to low) ---
print(f"\nPart 2: Lowpass sweep {HI_FREQUENCY}Hz -> {LO_FREQUENCY}Hz - {DURATION_SECONDS}s, Q={Q}", flush=True)

# Create frequency sweep from 8000Hz to 200Hz
freq_sweep_down = RampPE(HI_FREQUENCY, LO_FREQUENCY, duration=duration_samples)

# Apply lowpass filter with sweeping frequency
filtered_down = BiquadPE(
    source,
    frequency=freq_sweep_down,
    q=Q,
    mode=BiquadMode.LOWPASS
)
output_down = CropPE(filtered_down, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(output_down)

with renderer:
    renderer.start()
    renderer.play_extent()

# --- Part 3: Resonant bandpass sweep ---
print(f"\nPart 3: Resonant bandpass sweep 300Hz -> 3000Hz (Q=5) - {DURATION_SECONDS}s", flush=True)

freq_sweep_bp = RampPE(300.0, 3000.0, duration=duration_samples)

filtered_bp = BiquadPE(
    source,
    frequency=freq_sweep_bp,
    q=5.0,  # Resonant
    mode=BiquadMode.BANDPASS
)
output_bp = CropPE(filtered_bp, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(output_bp)

with renderer:
    renderer.start()
    renderer.play_extent()

print("\nDone!", flush=True)
