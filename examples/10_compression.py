#!/usr/bin/env python3
"""
Example 10: Compression, Limiting, and Gating

Demonstrates the easy-to-use CompressorPE, LimiterPE, and GatePE
for common dynamics processing tasks.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path
import sys
import pygmu2 as pg
pg.set_sample_rate(44100)

sys.path.insert(0, 'src')

from pygmu2 import (
    AudioRenderer,
    CompressorPE,
    Extent,
    GainPE,
    GatePE,
    LimiterPE,
    LoopPE,
    MixPE,
    WavReaderPE,
    AudioLibrary,
)

# library = AudioLibrary.from_url("https://software.tomandandy.com/strudel.json")
# sound_path = library.resolve("bigBeat")
AUDIO_DIR = Path(__file__).parent / "audio"
sound_path = AUDIO_DIR / "acoustic_drums.wav"

def demo_basic_compression():
    """
    Compare dry vs compressed signal.
    """
    print("=== Dry vs Compressed Comparison ===")
    print("First: dry signal, Then: compressed signal")
    print()
    
    source = WavReaderPE(sound_path)
    sample_rate = source.file_sample_rate
    looped_drums = LoopPE(source, count=2, crossfade_seconds=0.002)
    
    compressed = CompressorPE(
        looped_drums,
        threshold=-24,      # Start compressing at -24dB
        ratio=6,            # 6:1 compression
        attack=0.02,        # 20ms attack
        release=0.2,        # 200ms release
        knee=6,             # Soft knee for smooth transition
        makeup_gain="auto", # Auto makeup gain
    )
    
    print("Playing DRY signal...")
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(looped_drums)
        renderer.start()
        renderer.play_extent()
    
    print("Playing COMPRESSED signal...")
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(compressed)
        renderer.start()
        renderer.play_extent()
    
    print()

def demo_limiter():
    """
    Demonstrate brick-wall limiting.
    """
    print("=== Brick-Wall Limiter ===")
    print("Preventing peaks from exceeding -6dB ceiling.")
    print()
    
    source_stream = WavReaderPE(sound_path)
    sample_rate = source_stream.file_sample_rate
    looped_drums_stream = LoopPE(source_stream, count=2, crossfade_seconds=0.002)

    # Make it intentionally too loud
    loud_stream = GainPE(looped_drums_stream, gain=10.0)
    
    # Apply limiter
    limited_stream = LimiterPE(
        loud_stream,
        ceiling=-12.0,     # -12dB ceiling
        release=0.05,      # 50ms release
        lookahead=0.005,   # 5ms lookahead for transparent limiting
    )
    
    print(f"Input: 10x amplitude (clipping without limiter)")
    print(f"Ceiling: -6dB, Release: 50ms, Lookahead: 5ms")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(limited_stream)
        renderer.start()
        renderer.play_extent()
    print()


def demo_noise_gate():
    """
    Demonstrate noise gating.
    """
    print("=== Noise Gate ===")
    print("Gating a signal to remove quiet passages.")
    print()
    
    source_stream = WavReaderPE(sound_path)
    sample_rate = source_stream.file_sample_rate
    looped_drums_stream = LoopPE(source_stream, count=2, crossfade_seconds=0.002)
    
    # Apply gate - quiet parts will be silenced
    gated_stream = GatePE(
        looped_drums_stream,
        threshold=-16,     # Gate threshold
        attack=0.001,      # 1ms attack (fast open)
        release=0.1 ,      # 1ms release
        range=-30,         # -30dB attenuation when gated
    )
    
    print(f"Threshold: -16dB, Attack: 1ms, Release: 100ms")
    print(f"Range: -30dB (attenuation when gated)")
    print("You should hear the signal cut out during quiet portions.")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(gated_stream)
        renderer.start()
        renderer.play_extent()
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
    
    source_stream = WavReaderPE(sound_path)
    sample_rate = source_stream.file_sample_rate
    looped_drums_stream = LoopPE(source_stream, count=2, crossfade_seconds=0.002)
    
    # Apply limiter to squash it
    limited_stream = LimiterPE(
        GainPE(looped_drums_stream, gain=10.0),
        ceiling=-6.0,      # -3dB ceiling
        release=0.05,      # 50ms release
        lookahead=0.005,   # 5ms lookahead for transparent limiting
    )
    
    # Mix dry + wet (parallel compression)
    parallel_stream = MixPE(
        GainPE(looped_drums_stream, gain=0.6),      # Dry (60%)
        GainPE(limited_stream, gain=0.4),     # Limited (40%)
    )
    
    print("Mixing 60% dry + 40% heavily compressed")
    print(f"Compression: threshold=-30dB, ratio=10:1")
    print("Result: Punchy transients from dry + sustained body from compressed")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(parallel_stream)
        renderer.start()
        renderer.play_extent()
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
        ("a", "All demos", None),
    ]
    
    print("Available demos:")
    for key, name, _ in demos:
        print(f"  {key}: {name}")
    print()
    
    choice = input("Choose a demo (1-4 or 'a' for all): ").strip().lower()
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
