"""
Create PingPong.wav - Create an Impulse Response for convolution that 
"ping pongs" a mono signal into left and right chennels

Creates a stereo WAV file with:
- t = 1 beat delay: unit impulse in left channel only
- t = 2 beat delay: unit impulse in right channel only

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import (
    DiracPE,
    DelayPE,
    MixPE,
    SpatialPE,
    SpatialAdapter,
    SpatialLinear,
    WavWriterPE,
    NullRenderer,
    CropPE,
    Extent,
    seconds_to_samples,
)

SAMPLE_RATE = 44100
BEATS_PER_MINUTE = 92  # chosen for acoustic_drums.wav
SECONDS_PER_BEAT = 60.0 / BEATS_PER_MINUTE
# DURATION_SECONDS = 1.5  # Slightly longer than 1.0s to capture all impulses
# DURATION_SAMPLES = int(DURATION_SECONDS * SAMPLE_RATE)

# Output file path
OUTPUT_DIR = Path(__file__).parent / "audio"
OUTPUT_FILE = OUTPUT_DIR / "PingPong.wav"

print("=== Creating PingPong.wav ===", flush=True)

# Create unit impulses
impulse = DiracPE(channels=1)

# t = 1 beat: unit impulse in left channel only
delay_05 = int(round(seconds_to_samples(SECONDS_PER_BEAT, SAMPLE_RATE)))
impulse_t05_delayed = DelayPE(impulse, delay=delay_05)
impulse_t05_left = SpatialPE(impulse_t05_delayed, method=SpatialLinear(azimuth=-90.0))  # All left

# t = 1.0: unit impulse in right channel only
delay_10 = int(round(seconds_to_samples(2 * SECONDS_PER_BEAT, SAMPLE_RATE)))
impulse_t10_delayed = DelayPE(impulse, delay=delay_10)
impulse_t10_right = SpatialPE(impulse_t10_delayed, method=SpatialLinear(azimuth=90.0))  # All right

# Mix the impulses
mix_stream = MixPE(impulse_t05_left, impulse_t10_right)

# Crop to finite length (leave a few extra samples at the end)
mix_stream = CropPE(mix_stream, Extent(0, delay_10+5))

# Write to WAV file
output_stream = WavWriterPE(mix_stream, str(OUTPUT_FILE))

print(f"Rendering Impulse Response file to: {OUTPUT_FILE}", flush=True)
print(f"  Sample rate: {SAMPLE_RATE} Hz", flush=True)
print(f"  Duration: {mix_stream.extent()} seconds", flush=True)
print(f"  Channels: 2 (stereo)", flush=True)
print(f"  Impulses:", flush=True)
print(f"    t=first beat: left channel only", flush=True)
print(f"    t=second beat: right channel only", flush=True)

# Use NullRenderer for offline rendering
renderer = NullRenderer(sample_rate=SAMPLE_RATE)
renderer.set_source(output_stream)

with renderer:
    renderer.start()
    # Render the entire extent
    extent = output_stream.extent()
    renderer.render(extent.start, extent.end - extent.start)

print(f"\nFile written successfully!", flush=True)
print(f"  Size: {OUTPUT_FILE.stat().st_size:,} bytes", flush=True)
print(f"  Location: {OUTPUT_FILE.absolute()}", flush=True)
