#!/usr/bin/env python3
"""
Example 11: Advanced Dynamics with DynamicsPE

Demonstrates the flexible DynamicsPE for advanced dynamics processing
including sidechain compression and custom envelope routing.

For simple compression/limiting/gating, see 10_compression.py.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

import sys
sys.path.insert(0, 'src')

from pygmu2 import (
    AudioRenderer,
    BlitSawPE,
    ConstantPE,
    CropPE,
    DynamicsPE,
    DynamicsMode,
    EnvelopePE,
    Extent,
    GainPE,
    MixPE,
    RampPE,
    SinePE,
    SuperSawPE,
)


def demo_sidechain_ducking():
    """
    Demonstrate sidechain compression (ducking).
    
    Classic EDM technique: bass ducks when kick hits.
    """
    print("=== Sidechain Compression (Ducking) ===")
    print("Bass ducks when 'kick' hits - classic EDM pumping effect.")
    print()
    
    sample_rate = 44100
    duration = 4.0
    
    # Create a "kick drum" - low frequency pulse on the beat
    # 2 Hz = 120 BPM (one hit every 0.5 seconds)
    beat_freq = 2.0
    
    # Kick: sine burst with quick decay
    # Use a half-wave rectified sine to get pulses
    kick_pulse = GainPE(
        SinePE(frequency=beat_freq),
        gain=1.0
    )
    # Rectify to get positive pulses only
    kick_trigger = GainPE(
        SinePE(frequency=60.0),  # Low kick frequency
        gain=kick_pulse  # Amplitude follows the pulse
    )
    
    # Bass synth - constant droning tone
    bass = GainPE(
        BlitSawPE(frequency=55.0),  # A1
        gain=0.7
    )
    
    # Create envelope follower from kick
    # Fast attack to catch transients, moderate release for pumping
    kick_env = EnvelopePE(
        kick_trigger,
        attack=0.001,   # 1ms - instant attack
        release=0.15,   # 150ms - creates the "pump"
    )
    
    # Sidechain compress: bass is ducked by kick envelope
    ducked_bass = DynamicsPE(
        bass,           # Audio to process
        kick_env,       # Control signal (from kick)
        threshold=-20,  # Start ducking at -20dB
        ratio=8,        # Strong ducking
        makeup_gain=0,
        mode=DynamicsMode.COMPRESS,
    )
    
    # Mix kick and ducked bass
    mix = MixPE(
        GainPE(kick_trigger, gain=0.4),  # Kick (quieter)
        GainPE(ducked_bass, gain=0.8),   # Ducked bass
    )
    
    print("Kick triggers compression on bass")
    print("Threshold: -20dB, Ratio: 8:1, Release: 150ms")
    print("Listen for the 'pumping' effect on the bass.")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(mix)
        renderer.start()
        renderer.play_range(0, int(duration * sample_rate))
    print()


def demo_sidechain_ducking_voice():
    """
    Voice ducking - music ducks when voice is present.
    
    Common in podcasts, radio, and announcements.
    """
    print("=== Voice Ducking (Podcast Style) ===")
    print("Music ducks when 'voice' is present.")
    print()
    
    sample_rate = 44100
    duration = 5.0
    
    # Simulate "voice" - a tone that comes and goes
    # In real use, this would be an actual voice track
    voice_freq = 300.0  # Approximate voice fundamental
    
    # Voice envelope: silent -> speaking -> silent -> speaking
    voice_env = CropPE(
        MixPE(
            # On/off pattern
            GainPE(SinePE(frequency=0.4), gain=0.5),
            ConstantPE(0.5),
        ),
        Extent(0, int(duration * sample_rate))
    )
    
    # "Voice" signal
    voice = GainPE(
        SinePE(frequency=voice_freq),
        gain=voice_env
    )
    
    # Background music - rich pad
    music = GainPE(
        SuperSawPE(frequency=110.0, voices=5, detune_cents=10),
        gain=0.5
    )
    
    # Create envelope from voice
    voice_detector = EnvelopePE(
        voice,
        attack=0.05,    # 50ms attack (not too fast)
        release=0.5,    # 500ms release (smooth return)
    )
    
    # Duck music when voice is present
    ducked_music = DynamicsPE(
        music,
        voice_detector,
        threshold=-20,
        ratio=4,        # Moderate ducking
        makeup_gain=0,
    )
    
    # Mix voice and ducked music
    mix = MixPE(
        GainPE(voice, gain=0.8),
        GainPE(ducked_music, gain=0.6),
    )
    
    print("Voice detector controls music level")
    print("Attack: 50ms, Release: 500ms (smooth transitions)")
    print("Music ducks during 'voice' portions.")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(mix)
        renderer.start()
        renderer.play_range(0, int(duration * sample_rate))
    print()


def demo_expander():
    """
    Demonstrate expansion (opposite of compression).
    
    Makes quiet parts quieter, increasing dynamic range.
    """
    print("=== Expander ===")
    print("Expansion: quiet parts become even quieter.")
    print()
    
    sample_rate = 44100
    duration = 3.0
    
    # Signal with varying dynamics
    signal = SinePE(frequency=440.0)
    
    # Amplitude modulation: varying between 0.2 and 1.0
    amp_env = MixPE(
        GainPE(SinePE(frequency=0.5), gain=0.4),
        ConstantPE(0.6),
    )
    
    source = GainPE(signal, gain=amp_env)
    
    # Create envelope follower
    env = EnvelopePE(source, attack=0.01, release=0.1)
    
    # Expand: reduce gain below threshold
    expanded = DynamicsPE(
        source,
        env,
        threshold=-6,      # Expand below -6dB
        ratio=2.0,         # 2:1 expansion (1dB below threshold -> 2dB reduction)
        mode=DynamicsMode.EXPAND,
        makeup_gain=0,
    )
    
    print("Threshold: -6dB, Ratio: 2:1 expansion")
    print("Quiet parts become even quieter.")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(expanded)
        renderer.start()
        renderer.play_range(0, int(duration * sample_rate))
    print()


def demo_multiband_concept():
    """
    Demonstrate the concept of multiband processing.
    
    Note: This is a simplified demonstration. Real multiband
    compression would use proper crossover filters.
    """
    print("=== Multiband Compression Concept ===")
    print("Different compression settings for low and high frequencies.")
    print()
    
    sample_rate = 44100
    duration = 3.0
    
    # Create a complex signal with both low and high content
    low_content = GainPE(
        BlitSawPE(frequency=80.0),
        gain=0.6
    )
    high_content = GainPE(
        SinePE(frequency=2000.0),
        gain=0.3
    )
    
    source = MixPE(low_content, high_content)
    
    # In a real multiband compressor, we'd split with crossover filters.
    # Here we'll just demonstrate compressing the full signal with
    # settings that would be typical for "multiband-like" results.
    
    # Envelope with moderate settings
    env = EnvelopePE(source, attack=0.01, release=0.15)
    
    # Compress with soft knee for smooth response across spectrum
    compressed = DynamicsPE(
        source,
        env,
        threshold=-15,
        ratio=3,
        knee=12,  # Very soft knee
        makeup_gain="auto",
    )
    
    print("Full-band compression with soft knee (12dB)")
    print("Soft knee provides smooth transition across levels")
    print("(True multiband would use crossover filters)")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(compressed)
        renderer.start()
        renderer.play_range(0, int(duration * sample_rate))
    print()


def demo_custom_envelope():
    """
    Demonstrate using a custom envelope shape for compression.
    
    Instead of following the input, use a completely separate control signal.
    """
    print("=== Custom Envelope Control ===")
    print("Using an LFO as the control signal for rhythmic compression.")
    print()
    
    sample_rate = 44100
    duration = 4.0
    
    # Audio source: sustained pad
    pad = GainPE(
        SuperSawPE(frequency=220.0, voices=5, detune_cents=15),
        gain=0.7
    )
    
    # Custom control signal: slow triangle-ish LFO
    # This creates rhythmic "breathing" independent of input level
    lfo_freq = 0.5  # 0.5 Hz = 2 second cycle
    
    # Use sine as control (rectified to stay positive)
    control = MixPE(
        GainPE(SinePE(frequency=lfo_freq), gain=0.4),
        ConstantPE(0.5),  # Keep it positive
    )
    
    # Apply compression driven by LFO, not by input
    rhythmic = DynamicsPE(
        pad,
        control,
        threshold=-6,
        ratio=4,
        makeup_gain=0,
    )
    
    print("Control signal: 0.5 Hz LFO (not the audio input)")
    print("Creates rhythmic 'breathing' effect")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(rhythmic)
        renderer.start()
        renderer.play_range(0, int(duration * sample_rate))
    print()


if __name__ == "__main__":
    print("pygmu2 Advanced Dynamics Examples (DynamicsPE)")
    print("=" * 50)
    print()
    print("DynamicsPE allows flexible routing of control signals,")
    print("enabling sidechain compression and creative effects.")
    print()
    
    demos = [
        ("1", "Sidechain Ducking (EDM)", demo_sidechain_ducking),
        ("2", "Voice Ducking (Podcast)", demo_sidechain_ducking_voice),
        ("3", "Expander", demo_expander),
        ("4", "Multiband Concept", demo_multiband_concept),
        ("5", "Custom Envelope Control", demo_custom_envelope),
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
