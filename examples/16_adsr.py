"""
Example 16: AdsrPE - Attack-Decay-Sustain-Release Envelope

Demonstrates various uses of AdsrPE for shaping sounds with
envelope generators.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pygmu2 import (
    AudioRenderer,
    AdsrPE,
    BlitSawPE,
    ConstantPE,
    CropPE,
    Extent,
    GainPE,
    MixPE,
    NullRenderer,
    ResetPE,
    SequencePE,
    SinePE,
    SuperSawPE,
    TransformPE,
    pitch_to_freq,
    seconds_to_samples,
)

# Configuration
SAMPLE_RATE = 44100

# MIDI notes for a minor chord (A minor: A3, C4, E4)
A3 = 57  # 220 Hz
C4 = 60  # 261.6 Hz
E4 = 64  # 329.6 Hz

def stos(seconds: float) -> int:
    """
    Convert seconds to sample count (as integer).
    """
    return int(seconds_to_samples(seconds, SAMPLE_RATE))

def make_gate(duration: float):
    """
    Create a PE that goes high for a specified number of seconds.
    """
    return CropPE(ConstantPE(1.0), Extent(0, stos(duration)))

def demo_basic_envelope():
    """
    Basic ADSR envelope shaping a simple oscillator.
    """
    print("=== Basic ADSR Envelope ===")
    print("Apply classic ADSR to classic pad sound")
    
    pad = SuperSawPE(
        frequency=pitch_to_freq(A3),
        amplitude=0.5,
        voices=3,
        detune_cents=15,
    )

    # pad = BlitSawPE(frequency=pitch_to_freq(A3))    

    # make a sequence of gates.  The first ones are long enough for
    # attack, decay and release to complete.  The last one forces a
    # truncated release.
    gates = SequencePE(
        [
            (make_gate(1.00), stos(0.0)),   # Long enough
            (make_gate(0.50), stos(1.5)),   # Long enough
            (make_gate(0.25), stos(3.0)),   # Almost long enough
            (make_gate(0.13), stos(4.5)),   # Forces release before decay finishes
        ])

    # Reset oscillator on each gate to prevent desynchronization
    reset_pad = ResetPE(pad, trigger=gates)
    
    adsr = AdsrPE(gates, 
                  attack=stos(0.005), 
                  decay=stos(0.25), 
                  sustain_level=0.2, 
                  release=stos(0.05))

    enveloped = GainPE(GainPE(reset_pad, adsr), gain=0.5)
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(enveloped)
    
    with renderer:
        renderer.start()
        renderer.play_range(stos(0.0), stos(8.0))

def demo_gated_sequence():
    """
    Multiple notes triggered by a gate sequence.
    """
    print("=== Gated Sequence ===")
    print("Description of this demo.")
    print()
    
    # TODO: Implement demo
    pass


def demo_retrigger_behavior():
    """
    Demonstrates re-triggering behavior during release.
    """
    print("=== Re-trigger Behavior ===")
    print("Description of this demo.")
    print()
    
    # TODO: Implement demo
    pass


if __name__ == "__main__":
    print("pygmu2 ADSR Examples")
    print("--------------------")
    
    demos = [
        ("1", "Basic ADSR Envelope", demo_basic_envelope),
        ("2", "Gated Sequence", demo_gated_sequence),
        ("3", "Re-trigger Behavior", demo_retrigger_behavior),
        ("a", "All demos", None),
    ]
    
    for key, name, _ in demos:
        print(f"{key}: {name}")
    print()
    
    choice = input("Choose a demo (0-3 or 'a' for all): ").strip().lower()
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
