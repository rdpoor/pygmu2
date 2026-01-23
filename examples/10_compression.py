#!/usr/bin/env python3
"""
Example 10: Compression, Limiting, and Gating

Demonstrates the easy-to-use CompressorPE, LimiterPE, and GatePE
for common dynamics processing tasks.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

import sys
sys.path.insert(0, 'src')

from pygmu2 import (
    AudioRenderer,
    BlitSawPE,
    CompressorPE,
    ConstantPE,
    CropPE,
    Extent,
    GainPE,
    GatePE,
    LimiterPE,
    MixPE,
    RampPE,
    SinePE,
    SuperSawPE,
)


def demo_basic_compression():
    """
    Demonstrate basic compression on a synth pad.
    """
    print("=== Basic Compression ===")
    print("A rich synth pad with compression to control dynamics.")
    print()
    
    sample_rate = 44100
    duration = 3.0  # seconds
    
    # Create a synth pad with some dynamics (volume swell)
    freq = 220.0  # A3
    
    # Volume envelope: swell up then sustain
    swell = CropPE(
        RampPE(0.3, 1.0, duration=int(1.5 * sample_rate)),
        Extent(0, int(duration * sample_rate))
    )
    
    # Rich supersaw pad
    pad = GainPE(
        SuperSawPE(frequency=freq, voices=5, detune_cents=15),
        gain=swell
    )
    
    # Add some higher octave for brightness
    pad_bright = MixPE(
        pad,
        GainPE(SuperSawPE(frequency=freq * 2, voices=3, detune_cents=10), gain=0.3),
    )
    
    # Compress to even out the dynamics
    compressed = CompressorPE(
        pad_bright,
        threshold=-12,      # Start compressing at -12dB
        ratio=4,            # 4:1 compression
        attack=0.02,        # 20ms attack
        release=0.2,        # 200ms release
        knee=6,             # Soft knee for smooth transition
        makeup_gain="auto", # Auto makeup gain
    )
    
    print(f"Threshold: -12dB, Ratio: 4:1")
    print(f"Attack: 20ms, Release: 200ms, Knee: 6dB")
    print(f"Auto makeup gain: {compressed.makeup_gain:.1f}dB")
    print()
    
    # Play
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(compressed)
        renderer.start()
        renderer.play_range(0, int(duration * sample_rate))
    print()


def demo_limiter():
    """
    Demonstrate brick-wall limiting.
    """
    print("=== Brick-Wall Limiter ===")
    print("Preventing peaks from exceeding -1dB ceiling.")
    print()
    
    sample_rate = 44100
    duration = 3.0
    
    # Create a loud sawtooth signal
    saw = BlitSawPE(frequency=110.0)
    
    # Make it intentionally too loud
    loud = GainPE(saw, gain=1.5)
    
    # Apply limiter
    limited = LimiterPE(
        loud,
        ceiling=-1.0,      # -1dB ceiling
        release=0.05,      # 50ms release
        lookahead=0.005,   # 5ms lookahead for transparent limiting
    )
    
    print(f"Input: 1.5x amplitude (clipping without limiter)")
    print(f"Ceiling: -1dB, Release: 50ms, Lookahead: 5ms")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(limited)
        renderer.start()
        renderer.play_range(0, int(duration * sample_rate))
    print()


def demo_noise_gate():
    """
    Demonstrate noise gating.
    """
    print("=== Noise Gate ===")
    print("Gating a signal to remove quiet passages.")
    print()
    
    sample_rate = 44100
    duration = 4.0
    
    # Create a signal that fades in and out
    signal = SinePE(frequency=440.0)
    
    # Amplitude envelope: quiet -> loud -> quiet -> loud (slow cycle)
    amp_env = MixPE(
        GainPE(SinePE(frequency=0.5), gain=0.4),  # Slow modulation
        ConstantPE(0.5),  # Offset to keep it positive
    )
    
    modulated = GainPE(signal, gain=amp_env)
    
    # Apply gate - quiet parts will be silenced
    gated = GatePE(
        modulated,
        threshold=-12,     # Gate threshold
        attack=0.001,      # 1ms attack (fast open)
        release=0.05,      # 50ms release
        range=-60,         # -60dB attenuation when gated
    )
    
    print(f"Signal amplitude cycles between 0.1 and 0.9")
    print(f"Threshold: -12dB, Attack: 1ms, Release: 50ms")
    print(f"Range: -60dB (attenuation when gated)")
    print("You should hear the signal cut out during quiet portions.")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(gated)
        renderer.start()
        renderer.play_range(0, int(duration * sample_rate))
    print()


def demo_parallel_compression():
    """
    Demonstrate parallel (New York) compression.
    """
    print("=== Parallel Compression ===")
    print("Mixing dry signal with heavily compressed signal for punch + dynamics.")
    print()
    
    sample_rate = 44100
    duration = 3.0
    
    # Source: dynamic synth with amplitude modulation
    source = GainPE(
        SuperSawPE(frequency=165.0, voices=5),  # E3
        gain=MixPE(
            GainPE(SinePE(frequency=1.0), gain=0.3),  # Slow modulation
            ConstantPE(0.6),  # Base level
        )
    )
    
    # Heavy compression (squash it!)
    compressed = CompressorPE(
        source,
        threshold=-30,     # Very low threshold
        ratio=10,          # Heavy compression
        attack=0.005,
        release=0.1,
        makeup_gain="auto",
    )
    
    # Mix dry + wet (parallel compression)
    parallel = MixPE(
        GainPE(source, gain=0.6),      # Dry (60%)
        GainPE(compressed, gain=0.4),  # Compressed (40%)
    )
    
    print("Mixing 60% dry + 40% heavily compressed")
    print(f"Compression: threshold=-30dB, ratio=10:1")
    print("Result: Punchy transients from dry + sustained body from compressed")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(parallel)
        renderer.start()
        renderer.play_range(0, int(duration * sample_rate))
    print()


def demo_comparison():
    """
    Compare dry vs compressed signal.
    """
    print("=== Dry vs Compressed Comparison ===")
    print("First: dry signal, Then: compressed signal")
    print()
    
    sample_rate = 44100
    duration = 2.0
    
    # Dynamic source
    source = GainPE(
        SuperSawPE(frequency=220.0, voices=5),
        gain=MixPE(
            GainPE(SinePE(frequency=2.0), gain=0.4),
            ConstantPE(0.5),
        )
    )
    
    compressed = CompressorPE(
        source,
        threshold=-15,
        ratio=6,
        attack=0.01,
        release=0.15,
        makeup_gain="auto",
    )
    
    print("Playing DRY signal...")
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(source)
        renderer.start()
        renderer.play_range(0, int(duration * sample_rate))
    
    print("Playing COMPRESSED signal...")
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(compressed)
        renderer.start()
        renderer.play_range(0, int(duration * sample_rate))
    
    print()


if __name__ == "__main__":
    print("pygmu2 Compression Examples")
    print("=" * 50)
    print()
    
    demos = [
        ("1", "Basic Compression", demo_basic_compression),
        ("2", "Brick-Wall Limiter", demo_limiter),
        ("3", "Noise Gate", demo_noise_gate),
        ("4", "Parallel Compression", demo_parallel_compression),
        ("5", "Dry vs Compressed", demo_comparison),
        ("a", "All demos", None),
    ]
    
    print("Available demos:")
    for key, name, _ in demos:
        print(f"  {key}: {name}")
    print()
    
    choice = input("Choose a demo (1-5 or 'a' for all): ").strip().lower()
    print()
    
    if choice == "a":
        for key, name, func in demos[:-1]:
            func()
    else:
        for key, name, func in demos:
            if key == choice and func:
                func()
                break
        else:
            print(f"Unknown choice: {choice}")
