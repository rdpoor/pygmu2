"""
Example 05: Flanging - Time-varying delay effect

Demonstrates DelayPE with a sine-modulated delay time,
creating classic flanging and chorus effects.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import (
    WavReaderPE,
    DelayPE,
    SinePE,
    MixPE,
    GainPE,
    CropPE,
    ConstantPE,
    AudioRenderer,
    Extent,
)

# Path to audio file
AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "faun.wav"

DURATION_SECONDS = 8

print("=== pygmu2 Example 05: Flanging ===", flush=True)
print(f"Loading: {WAV_FILE}", flush=True)

# Load orchestral source
source = WavReaderPE(str(WAV_FILE))
sample_rate = source.file_sample_rate
duration_samples = int(DURATION_SECONDS * sample_rate)

# --- Part 1: Original sound (dry) ---
print(f"\nPart 1: Original sound (dry) - {DURATION_SECONDS}s", flush=True)

output1 = CropPE(source, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(output1)

with renderer:
    renderer.start()
    renderer.play_extent()

# --- Part 2: Subtle chorus effect ---
print(f"\nPart 2: Chorus effect (0.25Hz, 20-25ms delay) - {DURATION_SECONDS}s", flush=True)

# Chorus uses longer delay times with slow, subtle modulation
chorus_rate = 0.25  # Hz - very slow for subtle thickening
chorus_depth_ms = 2.5  # +/- 2.5ms - subtle variation
chorus_center_ms = 22.0  # Center delay for "doubling" effect

chorus_depth_samples = chorus_depth_ms * sample_rate / 1000
chorus_center_samples = chorus_center_ms * sample_rate / 1000

lfo_chorus = SinePE(frequency=chorus_rate, amplitude=chorus_depth_samples)
center_chorus = ConstantPE(chorus_center_samples)
delay_signal_chorus = MixPE(center_chorus, lfo_chorus)

delayed_chorus = DelayPE(source, delay=delay_signal_chorus)

dry_chorus = GainPE(source, gain=0.6)
wet_chorus = GainPE(delayed_chorus, gain=0.4)
chorused = MixPE(dry_chorus, wet_chorus)

output2 = CropPE(chorused, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(output2)

with renderer:
    renderer.start()
    renderer.play_extent()

# --- Part 3: Classic flanging ---
print(f"\nPart 3: Classic flanging (0.5Hz, 0-10ms delay) - {DURATION_SECONDS}s", flush=True)

# Flanging uses a slowly oscillating short delay (0-10ms)
# delay_samples = base_delay + depth * sin(rate * t)
# At 48000 Hz: 10ms = 480 samples

flange_rate = 0.5  # Hz - slow sweep
flange_depth_ms = 5.0  # +/- 5ms around center
flange_center_ms = 5.0  # Center delay

flange_depth_samples = flange_depth_ms * sample_rate / 1000
flange_center_samples = flange_center_ms * sample_rate / 1000

# Create LFO for delay modulation
lfo_flange = SinePE(frequency=flange_rate, amplitude=flange_depth_samples)

# Add center offset: center + lfo
center_flange = ConstantPE(flange_center_samples)
delay_signal_flange = MixPE(center_flange, lfo_flange)

# Apply variable delay
delayed_flange = DelayPE(source, delay=delay_signal_flange)

# Mix dry and wet signals (50/50)
dry_flange = GainPE(source, gain=0.5)
wet_flange = GainPE(delayed_flange, gain=0.5)
flanged = MixPE(dry_flange, wet_flange)

output3 = CropPE(flanged, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(output3)

with renderer:
    renderer.start()
    renderer.play_extent()

print("\nDone!", flush=True)
