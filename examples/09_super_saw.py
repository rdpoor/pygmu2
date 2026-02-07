"""
Example 09: Super Saw - Rich, detuned unison oscillator

Demonstrates SuperSawPE, which creates the classic "supersaw" sound
by mixing multiple slightly-detuned sawtooth oscillators together.

This sound is a staple of trance, EDM, and synth pads.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pygmu2 import (
import pygmu2 as pg
pg.set_sample_rate(44100)

    SuperSawPE,
    SinePE,
    GainPE,
    CropPE,
    AudioRenderer,
    Extent,
    pitch_to_freq,
)

# Try to import BiquadPE (requires scipy)
try:
    from pygmu2 import BiquadPE, BiquadMode
    HAS_BIQUAD = True
except ImportError:
    HAS_BIQUAD = False
    print("Note: BiquadPE not available (scipy not installed)")

# Configuration
SAMPLE_RATE = 44100
DURATION_SECONDS = 6
DURATION_SAMPLES = int(DURATION_SECONDS * SAMPLE_RATE)

# MIDI notes for a minor chord (A minor: A3, C4, E4)
A3 = 57  # 220 Hz
C4 = 60  # 261.6 Hz
E4 = 64  # 329.6 Hz

print("=== pygmu2 Example 09: Super Saw ===", flush=True)
print(flush=True)

# --- Demo 1: Basic supersaw ---
print("Demo 1: Basic supersaw chord (A minor)", flush=True)
print(f"  Voices: 7, Detune: 20 cents", flush=True)

# Create three supersaw oscillators for a chord
saw_a_stream = SuperSawPE(
    frequency=pitch_to_freq(A3),
    amplitude=0.3,
    voices=7,
    detune_cents=20.0,
    mix_mode='center_heavy',
)

saw_c_stream = SuperSawPE(
    frequency=pitch_to_freq(C4),
    amplitude=0.3,
    voices=7,
    detune_cents=20.0,
    mix_mode='center_heavy',
)

saw_e_stream = SuperSawPE(
    frequency=pitch_to_freq(E4),
    amplitude=0.3,
    voices=7,
    detune_cents=20.0,
    mix_mode='center_heavy',
)

# Mix the chord together
from pygmu2 import MixPE
chord_stream = MixPE(saw_a_stream, saw_c_stream, saw_e_stream)

# Apply overall gain to avoid clipping
output_stream = GainPE(chord_stream, gain=0.4)

# Optionally add a lowpass filter for a smoother sound
if HAS_BIQUAD:
    print("  Adding lowpass filter at 3000 Hz", flush=True)
    output_stream = BiquadPE(output_stream, mode=BiquadMode.LOWPASS, frequency=3000.0, q=0.7)

# Crop to desired duration
output_stream = CropPE(output_stream, Extent(0, DURATION_SAMPLES))

# Play
print(f"Playing for {DURATION_SECONDS} seconds...", flush=True)
renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
renderer.set_source(output_stream)

with renderer:
    renderer.start()
    renderer.play_extent()

print(flush=True)

# --- Demo 2: Comparing detune amounts ---
print("Demo 2: Comparing detune amounts", flush=True)
print("  Playing: no detune -> light detune -> heavy detune", flush=True)

detune_amounts = [0.0, 15.0, 50.0]
short_duration = int(2 * SAMPLE_RATE)  # 2 seconds each

for detune in detune_amounts:
    label = "mono" if detune == 0 else f"{detune} cents"
    print(f"  Detune: {label}", flush=True)
    
    saw = SuperSawPE(
        frequency=pitch_to_freq(A3),
        amplitude=0.5,
        voices=7,
        detune_cents=detune,
    )
    
    if HAS_BIQUAD:
        saw = BiquadPE(saw, mode=BiquadMode.LOWPASS, frequency=2500.0, q=0.7)
    
    output = GainPE(saw, gain=0.5)
    output = CropPE(output, Extent(0, short_duration))
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(output_stream)
    
    with renderer:
        renderer.start()
        renderer.play_extent()

print(flush=True)

# --- Demo 3: Mix modes comparison ---
print("Demo 3: Comparing mix modes", flush=True)
print("  Playing: equal -> center_heavy -> linear", flush=True)

mix_modes = ['equal', 'center_heavy', 'linear']

for mode in mix_modes:
    print(f"  Mix mode: {mode}", flush=True)
    
    saw = SuperSawPE(
        frequency=pitch_to_freq(E4),
        amplitude=0.5,
        voices=7,
        detune_cents=25.0,
        mix_mode=mode,
    )
    
    if HAS_BIQUAD:
        saw = BiquadPE(saw, mode=BiquadMode.LOWPASS, frequency=4000.0, q=0.5)
    
    output = GainPE(saw, gain=0.5)
    output = CropPE(output, Extent(0, short_duration))
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(output_stream)
    
    with renderer:
        renderer.start()
        renderer.play_extent()

print()
print("Done!", flush=True)
print()
print("Tips for using SuperSawPE:")
print("  - 7 voices with 15-25 cents detune is a good starting point")
print("  - Use a lowpass filter to tame the high frequencies")
print("  - Stack multiple notes for rich chord pads")
print("  - 'center_heavy' mix mode gives a more focused sound")
print("  - Higher voice counts (9-11) add richness but cost more CPU", flush=True)
