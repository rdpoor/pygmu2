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
from pathlib import Path

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
    LoopPE,
    MixPE,
    RampPE,
    SinePE,
    SuperSawPE,
    WavReaderPE,
)

# Path to audio file (relative to this script)
AUDIO_DIR = Path(__file__).parent / "audio"
BASS_FILE = AUDIO_DIR / "bass.wav"  # 90 bpm
BASS_BPM = 90
KICK_FILE = AUDIO_DIR / "kick.wav"
MUSIC_FILE = AUDIO_DIR / "choir.wav"
VOICE_FILE = AUDIO_DIR / "spoken_voice.wav"

def demo_sidechain_ducking():
    """
    Demonstrate sidechain compression (ducking).
    
    Classic EDM technique: bass ducks when kick hits.
    """
    print("=== Sidechain Compression (Ducking) ===")
    print("Bass ducks when 'kick' hits - classic EDM pumping effect.")
    print()
    
    bass_stream = WavReaderPE(str(BASS_FILE))
    sample_rate = bass_stream.file_sample_rate
    kick_stream = WavReaderPE(str(KICK_FILE))

    # loop the kick to keep time with the bass
    samples_per_beat = (60/BASS_BPM) * sample_rate
    kick_pulse_stream = LoopPE(kick_stream, loop_end=int(samples_per_beat))
        
    # Create envelope follower from kick
    # Fast attack to catch transients, moderate release for pumping
    kick_env_stream = EnvelopePE(
        kick_pulse_stream,
        attack=0.001,   # 1ms - instant attack
        release=0.15,   # 150ms - creates the "pump"
    )
    
    # Sidechain compress: bass is ducked by kick envelope
    ducked_bass_stream = DynamicsPE(
        bass_stream,           # Audio to process
        kick_env_stream,       # Control signal (from kick)
        threshold=-20,  # Start ducking at -20dB
        ratio=8,        # Strong ducking
        makeup_gain=0,
        mode=DynamicsMode.COMPRESS,
    )
    
    # Mix kick and ducked bass
    mix_stream = MixPE(
        GainPE(kick_pulse_stream, gain=0.2),  # Kick (quieter)
        GainPE(ducked_bass_stream, gain=0.8),   # Ducked bass
    )
    
    print("Kick triggers compression on bass")
    print("Threshold: -20dB, Ratio: 8:1, Release: 150ms")
    print("Listen for the 'pumping' effect on the bass.")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(mix_stream)
        renderer.start()
        renderer.play_range(0, bass_stream.extent().end)
    print()


def demo_sidechain_ducking_voice():
    """
    Voice ducking - music ducks when voice is present.
    
    Common in podcasts, radio, and announcements.
    """
    print("=== Voice Ducking (Podcast Style) ===")
    print("Music ducks when 'voice' is present.")
    print()
    
    music_stream = WavReaderPE(str(MUSIC_FILE))
    voice_stream = WavReaderPE(str(VOICE_FILE))
    sample_rate = voice_stream.file_sample_rate
    
    # Create envelope from voice
    voice_detector_stream = EnvelopePE(
        voice_stream,
        attack=0.05,    # 50ms attack (not too fast)
        release=0.15,    # 150ms release (smooth return)
    )
    
    # Duck music when voice is present
    ducked_music_stream = DynamicsPE(
        music_stream,
        voice_detector_stream,
        threshold=-40,
        ratio=4,        # Moderate ducking
        makeup_gain=0,
    )
    
    # Mix voice and ducked music
    mix_stream = MixPE(
        GainPE(voice_stream, gain=0.4),
        GainPE(ducked_music_stream, gain=0.6),
    )
    
    print("Voice detector controls music level")
    print("Attack: 50ms, Release: 150ms (smooth transitions)")
    print("Music ducks during voice portions.")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(mix_stream)
        renderer.start()
        renderer.play_extent()
    print()


def demo_expander():
    """
    Demonstrate expansion (opposite of compression).
    
    Makes quiet parts quieter, increasing dynamic range.
    """
    print("=== Expander ===")
    print("Expansion: quiet parts become even quieter.")
    print()

    source_stream = WavReaderPE(str(BASS_FILE))
    sample_rate = source_stream.file_sample_rate
    
    # Create envelope follower
    env_stream = EnvelopePE(source_stream, attack=0.01, release=0.1)
    
    # Expand: reduce gain below threshold
    expanded_stream = DynamicsPE(
        source_stream,
        env_stream,
        threshold=-6,      # Expand below -6dB
        ratio=4.0,         # 4:1 expansion (1dB below threshold -> 4dB reduction)
        mode=DynamicsMode.EXPAND,
        makeup_gain=0,
    )
    
    print("Threshold: -6dB, Ratio: 4:1 expansion")
    print("Quiet parts become even quieter.")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(expanded_stream)
        renderer.start()
        renderer.play_extent()
    print()



def demo_custom_envelope():
    """
    Demonstrate using a custom envelope shape for compression.
    
    Instead of following the input, use a completely separate control signal.
    """
    print("=== Custom Envelope Control ===")
    print("Using an LFO as the control signal for rhythmic compression.")
    print()
    
    pad_stream = WavReaderPE(str(MUSIC_FILE))
    sample_rate = pad_stream.file_sample_rate
    
    # Custom control signal: slow triangle-ish LFO
    # This creates rhythmic "breathing" independent of input level
    lfo_freq = 3

    # Use sine as control (rectified to stay positive)
    control_stream = MixPE(
        GainPE(SinePE(frequency=lfo_freq), gain=0.4),
        ConstantPE(0.5),  # Keep it positive
    )
    
    # Apply compression driven by LFO, not by input
    rhythmic_stream = DynamicsPE(
        pad_stream,
        control_stream,
        threshold=-6,
        ratio=4,
        makeup_gain=0,
    )
    
    print(f"Control signal: {lfo_freq} Hz LFO (not the audio input)")
    print("Creates rhythmic 'breathing' effect")
    print()
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(rhythmic_stream)
        renderer.start()
        renderer.play_extent()
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
        ("4", "Custom Envelope Control", demo_custom_envelope),
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
