"""
Example 03: Looping - Repeating audio segments

Demonstrates LoopPE with and without crossfade on a percussion sample.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path

from pygmu2 import WavReaderPE, LoopPE, CropPE, Extent
import pygmu2 as pg
from examples_helper import run_demos
pg.set_sample_rate(44100)


# Path to audio file
AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "djembe.wav"

DURATION_SECONDS = 8

source_stream = WavReaderPE(str(WAV_FILE))
sample_rate = source_stream.file_sample_rate
duration_samples = int(DURATION_SECONDS * sample_rate)

def loop_without_crossfade():
    print("=== pygmu2 Example 03: Looping ===", flush=True)
    print(f"Loading: {WAV_FILE}", flush=True)

    extent = source_stream.extent()
    loop_length = extent.end - extent.start
    print(f"  Original duration: {loop_length / sample_rate:.2f}s ({loop_length} samples)", flush=True)

    # --- Part 1: Basic loop (no crossfade) ---
    print(f"\nPart 1: Basic loop (no crossfade) - {DURATION_SECONDS}s", flush=True)

    looped_basic_stream = LoopPE(source_stream)
    output_basic_stream = CropPE(looped_basic_stream, 0, (duration_samples) - (0))

    pg.play(output_basic_stream, sample_rate=sample_rate)
    print("\nDone!", flush=True)

def loop_with_crossfade():
    # --- Part 2: Smooth loop (with crossfade) ---
    print(f"\nPart 2: Smooth loop (20ms crossfade) - {DURATION_SECONDS}s", flush=True)

    looped_smooth_stream = LoopPE(source_stream, crossfade_seconds=0.02)  # 20ms crossfade
    output_smooth_stream = CropPE(looped_smooth_stream, 0, (duration_samples) - (0))

    pg.play(output_smooth_stream, sample_rate=sample_rate)
    print("\nDone!", flush=True)

DEMOS = [
    ("Play loop without crossfade", loop_without_crossfade),
    ("Play loop with crossfade", loop_with_crossfade),
]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    run_demos(DEMOS)
