"""
Example 03: Looping - Repeating audio segments

Demonstrates LoopPE with and without crossfade on a percussion sample.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

import time
from pathlib import Path
from pygmu2 import WavReaderPE, LoopPE, CropPE, AudioRenderer, Extent

# Path to audio file
AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "djembe.wav"

DURATION_SECONDS = 8

print("=== pygmu2 Example 03: Looping ===", flush=True)
print(f"Loading: {WAV_FILE}", flush=True)

# Load the percussion sample
source = WavReaderPE(str(WAV_FILE))
sample_rate = source.file_sample_rate
duration_samples = int(DURATION_SECONDS * sample_rate)

extent = source.extent()
loop_length = extent.end - extent.start
print(f"  Original duration: {loop_length / sample_rate:.2f}s ({loop_length} samples)", flush=True)

# --- Part 1: Basic loop (no crossfade) ---
print(f"\nPart 1: Basic loop (no crossfade) - {DURATION_SECONDS}s", flush=True)

looped_basic = LoopPE(source)
output_basic = CropPE(looped_basic, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(output_basic)

t0 = time.perf_counter()
with renderer:
    renderer.start()
    renderer.play_extent()
t1 = time.perf_counter()
print(f"  Elapsed: {t1 - t0:.2f}s (expected ~{DURATION_SECONDS}s)", flush=True)

# --- Part 2: Smooth loop (with crossfade) ---
print(f"\nPart 2: Smooth loop (20ms crossfade) - {DURATION_SECONDS}s", flush=True)

looped_smooth = LoopPE(source, crossfade_seconds=0.02)  # 20ms crossfade
output_smooth = CropPE(looped_smooth, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(output_smooth)

t0 = time.perf_counter()
with renderer:
    renderer.start()
    renderer.play_extent()
t1 = time.perf_counter()
print(f"  Elapsed: {t1 - t0:.2f}s (expected ~{DURATION_SECONDS}s)", flush=True)

print("\nDone!", flush=True)
