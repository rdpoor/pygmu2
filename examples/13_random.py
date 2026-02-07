"""
Example 13: RandomPE - Musical Randomness

Demonstrates various modes and uses of RandomPE for generating
random notes, modulating parameters, and random walks.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pygmu2 import (
    AudioRenderer,
    BlitSawPE,
    ConstantPE,
    GainPE,
    MixPE,
    RandomPE,
    RandomMode,
    SinePE,
    TransformPE,
    pitch_to_freq,
)
import pygmu2 as pg
pg.set_sample_rate(44100)


# Configuration
SAMPLE_RATE = 44100


def demo_sample_hold_notes():
    """
    Random notes from C Major scale using SAMPLE_HOLD.
    """
    print("=== Random Notes (C Major Scale) ===")
    print("Random notes from C Major scale triggered by a 4 Hz clock.")
    print()
    
    # 4 Hz trigger (16th notes at 120 BPM)
    clock_stream = SinePE(frequency=4.0)
    
    # C Major scale (C4 to C5)
    c_major_pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    c_major_freqs = [pitch_to_freq(p) for p in c_major_pitches]
    
    # Generate random indices (0 to 7)
    # We generate slightly outside [0, 7] to ensure coverage after floor()
    random_index_stream = RandomPE(
        min_value=0,
        max_value=len(c_major_freqs) - 0.01,
        mode=RandomMode.SAMPLE_HOLD,
        trigger=clock_stream,
        seed=42
    )
    
    # Map index to frequency using TransformPE
    # This acts as a "quantizer" or "sequencer"
    def map_to_scale(indices):
        import numpy as np
        # Convert scale frequencies to numpy array for efficient indexing
        scale = np.array(c_major_freqs)
        
        # Convert float indices to integers
        idx = indices.astype(int)
        
        # Clamp to be safe
        idx = np.clip(idx, 0, len(scale) - 1)
        
        # Look up frequencies using numpy indexing
        # Reshape to ensure output matches input shape (samples, 1)
        # Note: idx is (N, 1), scale[idx] results in (N, 1)
        return scale[idx].reshape(indices.shape)

    scale_freq_stream = TransformPE(random_index_stream, map_to_scale)
    
    # Oscillator with quantized frequency
    synth_stream = BlitSawPE(frequency=scale_freq_stream, amplitude=0.4)
    
    # Adjust the gain
    output_stream = GainPE(synth_stream, gain=0.5)
    
    with AudioRenderer(sample_rate=SAMPLE_RATE) as renderer:
        renderer.set_source(output_stream)
        renderer.start()
        print("Playing for 4 seconds...")
        renderer.play_range(0, int(4.0 * SAMPLE_RATE))
    print()


def demo_smooth_modulation():
    """
    Smooth parameter modulation (vibrato and tremolo).
    """
    print("=== Smooth Modulation ===")
    print("Random wandering for vibrato depth and tremolo rate.")
    print()
    
    # Base frequency
    base_freq = 220.0
    
    # Random vibrato (frequency modulation)
    # Slow wandering between 2 and 10 Hz of depth
    vibrato_depth_stream = RandomPE(
        rate=0.5,
        min_value=2,
        max_value=15,
        mode=RandomMode.SMOOTH
    )
    
    # Vibrato LFO
    vibrato_lfo_stream = GainPE(SinePE(frequency=6.0), gain=vibrato_depth_stream)
    
    # Random tremolo rate (amplitude modulation)
    # Rate wanders between 1 Hz and 8 Hz
    trem_rate_stream = RandomPE(
        rate=0.2,
        min_value=1,
        max_value=8,
        mode=RandomMode.SMOOTH
    )
    
    # Tremolo LFO (normalized 0.5 to 1.0)
    trem_lfo_stream = MixPE(
        GainPE(SinePE(frequency=trem_rate_stream), gain=0.25),
        ConstantPE(0.75)
    )
    
    # Main oscillator
    synth_stream = BlitSawPE(
        frequency=MixPE(ConstantPE(base_freq), vibrato_lfo_stream),
        amplitude=0.5
    )
    
    output_stream = GainPE(synth_stream, gain=trem_lfo_stream)
    
    with AudioRenderer(sample_rate=SAMPLE_RATE) as renderer:
        renderer.set_source(output_stream)
        renderer.start()
        print("Playing for 5 seconds...")
        renderer.play_range(0, int(5.0 * SAMPLE_RATE))
    print()


def demo_random_walk():
    """
    Random walk for drifting pitch.
    """
    print("=== Random Walk ===")
    print("Pitch gently drifting up and down.")
    print()
    
    # Random walk around a center frequency
    # We'll use small steps (slew) for a gentle drift
    drift_stream = RandomPE(
        min_value=-100,
        max_value=100,
        mode=RandomMode.WALK,
        slew=0.0005,  # Moderate drift (0.05 Hz per sample max step)
        seed=123
    )
    
    frequency_stream = MixPE(ConstantPE(330.0), drift_stream)
    
    synth_stream = SinePE(frequency=frequency_stream, amplitude=0.4)
    
    with AudioRenderer(sample_rate=SAMPLE_RATE) as renderer:
        renderer.set_source(synth_stream)
        renderer.start()
        print("Playing for 5 seconds...")
        renderer.play_range(0, int(5.0 * SAMPLE_RATE))
    print()


if __name__ == "__main__":
    print("pygmu2 Randomness Examples")
    print("--------------------------")
    
    demos = [
        ("1", "Random Notes (C Major Scale)", demo_sample_hold_notes),
        ("2", "Smooth Modulation (Vibrato/Tremolo)", demo_smooth_modulation),
        ("3", "Random Walk (Drifting Pitch)", demo_random_walk),
        ("a", "All demos", None),
    ]
    
    for key, name, _ in demos:
        print(f"{key}: {name}")
    print()
    
    choice = input("Choose a demo (1-3 or 'a' for all): ").strip().lower()
    print()
    
    if choice == 'a':
        for _, _, func in demos[:-1]:
            func()
    else:
        for key, _, func in demos:
            if choice == key:
                func()
                break
        else:
            print("Invalid choice.")
