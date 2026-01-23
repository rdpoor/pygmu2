"""
Example 06: Autowah - Envelope-controlled filter

Demonstrates using EnvelopePE to control BiquadPE frequency,
creating the classic "autowah" or envelope filter effect.

The louder the input, the higher the filter frequency sweeps.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import (
    WavReaderPE,
    EnvelopePE,
    DetectionMode,
    BiquadPE,
    BiquadMode,
    TransformPE,
    GainPE,
    CropPE,
    LoopPE,
    AudioRenderer,
    Extent,
)

# Path to audio file
AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "djembe.wav"

DURATION_SECONDS = 8

print("=== pygmu2 Example 06: Autowah ===", flush=True)
print(f"Loading: {WAV_FILE}", flush=True)

# Load percussion source and loop it
source = WavReaderPE(str(WAV_FILE))
sample_rate = source.file_sample_rate
duration_samples = int(DURATION_SECONDS * sample_rate)

# Loop the djembe sample
looped = LoopPE(source, crossfade=0.01)

# --- Part 1: Original signal ---
print(f"\nPart 1: Original signal (looped djembe) - {DURATION_SECONDS}s", flush=True)

output1 = CropPE(looped, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(output1)

with renderer:
    renderer.start()
    renderer.play_extent()

# --- Part 2: Autowah effect ---
print(f"\nPart 2: Autowah effect - {DURATION_SECONDS}s", flush=True)
print("  Envelope follower -> frequency mapping -> lowpass filter", flush=True)

# 1. Extract envelope from the audio
#    Fast attack to catch transients, medium release for smooth sweep
envelope = EnvelopePE(
    looped,
    attack=0.005,   # 5ms attack - catch percussive hits
    release=0.05,   # 50ms release - smooth decay
    mode=DetectionMode.PEAK
)


# 2. Map envelope (0-1 range) to filter frequency
def envelope_to_freq(env):
    """Map envelope (0-1) to frequency (100-3000 Hz)."""
    import numpy as np
    # Clamp envelope to 0-1 range
    env = np.clip(env, 0, 1)
    # Exponential mapping sounds more musical
    # Low envelope -> low freq, high envelope -> high freq
    min_freq = 100.0
    max_freq = 3000.0
    # Use power curve for more dramatic sweep
    return min_freq + (max_freq - min_freq) * (env ** 0.5)


freq_control = TransformPE(envelope, func=envelope_to_freq, name="env_to_freq")

# 3. Apply lowpass filter controlled by envelope
filtered = BiquadPE(
    looped,
    frequency=freq_control,
    q=10.0,  # Resonant for that "wah" character
    mode=BiquadMode.LOWPASS
)

# 4. Output
output2 = GainPE(filtered, gain=1.0)
output2 = CropPE(output2, Extent(0, duration_samples))

renderer = AudioRenderer(sample_rate=sample_rate)
renderer.set_source(output2)

with renderer:
    renderer.start()
    renderer.play_extent()

print("\nDone!", flush=True)
