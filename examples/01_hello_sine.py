"""
Example 01: Hello Sine - Introduction to pygmu2

Creates a major triad from three sine waves, applies gain,
crops to 8 seconds, and plays through the audio output.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import pygmu2 as pg
pg.set_sample_rate(44100)

from pygmu2 import (
    SinePE,
    MixPE,
    GainPE,
    CropPE,
    AudioRenderer,
    Extent,
    pitch_to_freq,
)

# Configuration
SAMPLE_RATE = 44100
DURATION_SECONDS = 8
DURATION_SAMPLES = int(DURATION_SECONDS * SAMPLE_RATE)

# MIDI note numbers for C major triad (C4, E4, G4)
C4 = 60
E4 = 64
G4 = 67

print("=== pygmu2 Example 01: Hello Sine ===", flush=True)
print(f"Creating a C major triad ({DURATION_SECONDS}s)", flush=True)

# Create three sine oscillators at the triad frequencies
print(f"  C4: {pitch_to_freq(C4):.1f} Hz", flush=True)
print(f"  E4: {pitch_to_freq(E4):.1f} Hz", flush=True)
print(f"  G4: {pitch_to_freq(G4):.1f} Hz", flush=True)

sine_c_stream = SinePE(frequency=pitch_to_freq(C4), amplitude=0.3)
sine_e_stream = SinePE(frequency=pitch_to_freq(E4), amplitude=0.3)
sine_g_stream = SinePE(frequency=pitch_to_freq(G4), amplitude=0.3)

# Mix the three sines together
mixed_stream = MixPE(sine_c_stream, sine_e_stream, sine_g_stream)

# Apply overall gain (reduce to avoid clipping)
gained_stream = GainPE(mixed_stream, gain=0.5)

# Crop to desired duration
output_stream = CropPE(gained_stream, 0, (DURATION_SAMPLES) - (0))

# Create audio renderer and play
print(f"Playing for {DURATION_SECONDS} seconds...", flush=True)
renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
renderer.set_source(output_stream)

with renderer:
    renderer.start()
    renderer.play_extent()

print("Done!", flush=True)
