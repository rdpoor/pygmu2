"""
Example 07: Soft Clipping - TransformPE with saturation

Demonstrates TransformPE for waveshaping/distortion effects
using np.tanh for soft clipping (tube-like saturation).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
from pygmu2 import (
    SinePE,
    MixPE,
    GainPE,
    TransformPE,
    CropPE,
    AudioRenderer,
    Extent,
    pitch_to_freq,
)

SAMPLE_RATE = 44100
DURATION_SECONDS = 4
DURATION_SAMPLES = int(DURATION_SECONDS * SAMPLE_RATE)

print("=== pygmu2 Example 07: Soft Clipping ===", flush=True)

# Create a rich source: stacked fifths with slight detuning
# Detuning creates beating/internal motion that sounds great with saturation
print("Creating source: detuned stacked fifths chord", flush=True)

A2 = 45  # Low A
E3 = 52.05  # Fifth above (detuned)
A3 = 57.1  # Octave (detuned)
E4 = 64.15  # Another fifth (detuned)

sines = [
    SinePE(frequency=pitch_to_freq(A2) + 0.3, amplitude=0.4),
    SinePE(frequency=pitch_to_freq(E3) - 0.5, amplitude=0.3),
    SinePE(frequency=pitch_to_freq(A3) + 0.7, amplitude=0.25),
    SinePE(frequency=pitch_to_freq(E4) - 0.4, amplitude=0.2),
]

source = MixPE(*sines)

# --- Part 1: Clean signal ---
print(f"\nPart 1: Clean signal (no clipping) - {DURATION_SECONDS}s", flush=True)

clean = GainPE(source, gain=0.5)
clean_out = CropPE(clean, Extent(0, DURATION_SAMPLES))

renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
renderer.set_source(clean_out)

with renderer:
    renderer.start()
    renderer.play_extent()

# --- Part 2: Light saturation ---
print(f"\nPart 2: Light saturation (1.5x drive into tanh) - {DURATION_SECONDS}s", flush=True)

# Drive the signal harder, then soft clip
driven_light = GainPE(source, gain=1.5)
saturated_light = TransformPE(driven_light, func=np.tanh, name="tanh")
# Compensate for level reduction
output_light = GainPE(saturated_light, gain=0.6)
output_light = CropPE(output_light, Extent(0, DURATION_SAMPLES))

renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
renderer.set_source(output_light)

with renderer:
    renderer.start()
    renderer.play_extent()

# --- Part 3: Heavy saturation ---
print(f"\nPart 3: Heavy saturation (4x drive into tanh) - {DURATION_SECONDS}s", flush=True)

driven_heavy = GainPE(source, gain=4.0)
saturated_heavy = TransformPE(driven_heavy, func=np.tanh, name="tanh")
output_heavy = GainPE(saturated_heavy, gain=0.5)
output_heavy = CropPE(output_heavy, Extent(0, DURATION_SAMPLES))

renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
renderer.set_source(output_heavy)

with renderer:
    renderer.start()
    renderer.play_extent()

# --- Part 4: Asymmetric clipping (more "character") ---
print(f"\nPart 4: Asymmetric clipping - {DURATION_SECONDS}s", flush=True)


def asymmetric_clip(x):
    """Asymmetric soft clipping - different positive/negative response."""
    # Positive half: gentle clipping
    # Negative half: hard clipping (creates even harmonics)
    pos = np.tanh(x * 1.0)
    neg = np.tanh(x * 5.0)  # Much harder negative clipping
    return np.where(x >= 0, pos, neg)


driven_asym = GainPE(source, gain=3.0)
saturated_asym = TransformPE(driven_asym, func=asymmetric_clip, name="asymmetric")
output_asym = GainPE(saturated_asym, gain=0.6)
output_asym = CropPE(output_asym, Extent(0, DURATION_SAMPLES))

renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
renderer.set_source(output_asym)

with renderer:
    renderer.start()
    renderer.play_extent()

print("\nDone!", flush=True)
