"""
Example 04: Filtering - Biquad filter with frequency sweep

Demonstrates BiquadPE with a sweeping cutoff frequency,
creating a classic filter sweep effect.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path
import pygmu2 as pg
from examples_helper import run_demos

pg.set_sample_rate(44100)

from pygmu2 import (
    WavReaderPE,
    BiquadPE,
    BiquadMode,
    PiecewisePE,
    CropPE,
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
source_stream = WavReaderPE(str(WAV_FILE))
sample_rate = source_stream.file_sample_rate
pg.set_sample_rate(sample_rate)
duration_samples = int(DURATION_SECONDS * sample_rate)


def lowpass_sweep_low_to_high():
    # --- Part 1: Lowpass filter sweep (low to high) ---
    print(
        f"\nPart 1: Lowpass sweep {LO_FREQUENCY}Hz -> {HI_FREQUENCY}Hz - {DURATION_SECONDS}s, Q={Q}",
        flush=True,
    )

    # Create frequency sweep from 200Hz to 8000Hz
    freq_sweep_up_stream = PiecewisePE(
        [(0, LO_FREQUENCY), (duration_samples, HI_FREQUENCY)]
    )

    # Apply lowpass filter with sweeping frequency
    filtered_up_stream = BiquadPE(
        source_stream, frequency=freq_sweep_up_stream, q=Q, mode=BiquadMode.LOWPASS
    )
    output_up_stream = CropPE(filtered_up_stream, 0, (duration_samples) - (0))

    pg.play(pg.GainPE(output_up_stream, gain=0.52))


def lowpass_sweep_high_to_low():
    # --- Part 2: Lowpass filter sweep (high to low) ---
    print(
        f"\nPart 2: Lowpass sweep {HI_FREQUENCY}Hz -> {LO_FREQUENCY}Hz - {DURATION_SECONDS}s, Q={Q}",
        flush=True,
    )

    # Create frequency sweep from 8000Hz to 200Hz
    freq_sweep_down_stream = PiecewisePE(
        [(0, HI_FREQUENCY), (duration_samples, LO_FREQUENCY)]
    )

    # Apply lowpass filter with sweeping frequency
    filtered_down_stream = BiquadPE(
        source_stream, frequency=freq_sweep_down_stream, q=Q, mode=BiquadMode.LOWPASS
    )
    output_down_stream = CropPE(filtered_down_stream, 0, (duration_samples) - (0))

    pg.play(pg.GainPE(output_down_stream, gain=0.50))


def resonant_bandpass_sweep():
    # --- Part 3: Resonant bandpass sweep ---
    print(
        f"\nPart 3: Resonant bandpass sweep 300Hz -> 3000Hz (Q=5) - {DURATION_SECONDS}s",
        flush=True,
    )

    freq_sweep_bp_stream = PiecewisePE([(0, 300.0), (duration_samples, 3000.0)])

    filtered_bp_stream = BiquadPE(
        source_stream,
        frequency=freq_sweep_bp_stream,
        q=5.0,  # Resonant
        mode=BiquadMode.BANDPASS,
    )
    output_bp_stream = CropPE(filtered_bp_stream, 0, (duration_samples) - (0))

    pg.play(pg.GainPE(output_bp_stream, gain=4.11))

    print("\nDone!", flush=True)


DEMOS = [
    ("Lowpass sweep low to high", lowpass_sweep_low_to_high),
    ("Lowpass sweep high to low", lowpass_sweep_high_to_low),
    ("Resonant bandpass sweep", resonant_bandpass_sweep),
]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    run_demos(DEMOS)
