"""
Example 08: Write to File - Offline rendering to WAV

Demonstrates rendering audio to a file instead of playing it,
useful for batch processing, pre-rendering, or creating samples.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import (
    SinePE,
    MixPE,
    GainPE,
    CropPE,
    WavWriterPE,
    WavReaderPE,
    NullRenderer,
    AudioRenderer,
    Extent,
    pitch_to_freq,
)

SAMPLE_RATE = 44100
DURATION_SECONDS = 4
DURATION_SAMPLES = int(DURATION_SECONDS * SAMPLE_RATE)

# Output file path
OUTPUT_DIR = Path(__file__).parent / "audio"
OUTPUT_FILE = OUTPUT_DIR / "output_triad.wav"

print("=== pygmu2 Example 08: Write to File ===", flush=True)

# Create the same C major triad as example 01
C4, E4, G4 = 60, 64, 67

print(f"Creating C major triad...", flush=True)
print(f"  C4: {pitch_to_freq(C4):.1f} Hz", flush=True)
print(f"  E4: {pitch_to_freq(E4):.1f} Hz", flush=True)
print(f"  G4: {pitch_to_freq(G4):.1f} Hz", flush=True)

sine_c_stream = SinePE(frequency=pitch_to_freq(C4), amplitude=0.3)
sine_e_stream = SinePE(frequency=pitch_to_freq(E4), amplitude=0.3)
sine_g_stream = SinePE(frequency=pitch_to_freq(G4), amplitude=0.3)

mixed_stream = MixPE(sine_c_stream, sine_e_stream, sine_g_stream)
gained_stream = GainPE(mixed_stream, gain=0.5)
cropped_stream = CropPE(gained_stream, Extent(0, DURATION_SAMPLES))

# Wrap in WavWriterPE to write to file
# WavWriterPE passes audio through while also writing to disk
output_stream = WavWriterPE(cropped_stream, str(OUTPUT_FILE))

print(f"\nRendering to: {OUTPUT_FILE}", flush=True)
print(f"  Duration: {DURATION_SECONDS} seconds", flush=True)
print(f"  Sample rate: {SAMPLE_RATE} Hz", flush=True)

# Use NullRenderer for offline rendering (no audio output)
renderer = NullRenderer(sample_rate=SAMPLE_RATE)
renderer.set_source(output_stream)

with renderer:
    renderer.start()
    # Render the entire extent
    extent = output_stream.extent()
    renderer.render(extent.start, extent.end - extent.start)

print(f"\nFile written successfully!", flush=True)
print(f"  Size: {OUTPUT_FILE.stat().st_size:,} bytes", flush=True)

# --- Play back the written file ---
print(f"\nPlaying back: {OUTPUT_FILE}", flush=True)

playback_stream = WavReaderPE(str(OUTPUT_FILE))
playback_renderer = AudioRenderer(sample_rate=playback_stream.file_sample_rate)
playback_renderer.set_source(playback_stream)

with playback_renderer:
    playback_renderer.start()
    playback_renderer.play_extent()

# Clean up: delete the temporary output file
OUTPUT_FILE.unlink()
print(f"\nCleaned up: {OUTPUT_FILE.name}", flush=True)

print("\nDone!", flush=True)
