"""
Example 02: Play WAV - Loading and playing audio files

Loads a WAV file and plays it through the audio output.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import WavReaderPE, AudioRenderer

# Path to audio file (relative to this script)
AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "faun.wav"

print("=== pygmu2 Example 02: Play WAV ===", flush=True)
print(f"Loading: {WAV_FILE}", flush=True)

# Load the WAV file
source = WavReaderPE(str(WAV_FILE))

# Get file info (file_sample_rate is available before configuration)
file_sr = source.file_sample_rate
print(f"  Channels: {source.channel_count()}", flush=True)
print(f"  Sample rate: {file_sr} Hz", flush=True)

# Create renderer matching the file's sample rate
renderer = AudioRenderer(sample_rate=file_sr)
renderer.set_source(source)

# Now we can get extent (after set_source configures the graph)
extent = source.extent()
duration_samples = extent.end - extent.start
duration_seconds = duration_samples / file_sr
print(f"  Duration: {duration_seconds:.2f} seconds ({duration_samples} samples)", flush=True)

print(f"Playing...", flush=True)

with renderer:
    renderer.start()
    renderer.play_extent()

print("Done!", flush=True)
