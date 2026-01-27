"""
Example 18: AdsrPE - Attack-Decay-Sustain-Release Envelope

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
    DelayPE,
    Extent,
    GainPE,
    MixPE,
    NullRenderer,
    ResetPE,
    SequencePE,
    SinePE,
    SuperSawPE,
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

    pad_stream = SuperSawPE(
        frequency=pitch_to_freq(A3),
        amplitude=0.5,
        voices=3,
        detune_cents=15,
    )

    # pad_stream = BlitSawPE(frequency=pitch_to_freq(A3))

    # make a sequence of gates.  The first ones are long enough for
    # attack, decay and release to complete.  The last one forces a
    # truncated release.
    gates_stream = SequencePE(
        [
            (make_gate(1.00), stos(0.0)),  # Long enough
            (make_gate(0.50), stos(1.5)),  # Long enough
            (make_gate(0.25), stos(3.0)),  # Almost long enough
            (make_gate(0.13), stos(4.5)),  # Forces release before decay finishes
        ]
    )

    # Reset oscillator on each gate to prevent desynchronization
    reset_pad_stream = ResetPE(pad_stream, trigger=gates_stream)

    adsr_stream = AdsrPE(
        gates_stream,
        attack_samples=stos(0.005),
        decay_samples=stos(0.25),
        sustain_level=0.2,
        release_samples=stos(0.05),
    )

    enveloped_stream = GainPE(GainPE(reset_pad_stream, adsr_stream), gain=0.5)
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(enveloped_stream)

    with renderer:
        renderer.start()
        renderer.play_range(stos(0.0), stos(8.0))


def demo_axel_f():
    """
    Sing along if you know it.
    """
    from itertools import accumulate

    print("=== Gated Sequence ===")
    print("The song that made Roland Juno famous")
    print()
    BPM = 116
    SECONDS_PER_BEAT = 60.0 / BPM
    SAMPLES_PER_BEAT = SECONDS_PER_BEAT * SAMPLE_RATE

    def _samp(x: float) -> int:
        """Convert a (possibly fractional) sample count to int."""
        return int(round(x))

    # durations (in samples): Whole, Half, Quater, Eigth, Sixteenth, dotted
    dW = (SAMPLES_PER_BEAT * 4.0)
    dH = (SAMPLES_PER_BEAT * 2.0)
    dQ = (SAMPLES_PER_BEAT * 1.0)
    dE = (SAMPLES_PER_BEAT / 2.0)
    dS = (SAMPLES_PER_BEAT / 4.0)
    dDOT = 1.5

    # pitches (in frequency)
    pA, pBf, pB, pC, pDf, pD, pEf, pE, pF, pGf, pG, pAf = pitch_to_freq(range(69 - 12, 69))
    pAs, pCs, pDs, pFs, pGs = [pBf, pDf, pEf, pGf, pAf]
    # octaves
    o2, o3, o4, o5, o6 = [0.25, 0.5, 1.0, 2.0, 4.0]

    # theme is an array of (frequency (Hz), duration (samples), legato) triples
    # A legato of 1 means elide with next note.  Shorter than 1 means start a
    # new attack on the next note
    theme = [
        (pF, dQ, 0.7),
        (pAf, dE * dDOT, 0.7),
        (pF, dE, 0.4),
        (pF, dS, 0.4),
        (pBf * o5, dE, 0.4),
        (pF, dE, 0.7),
        (pEf, dE, 0.7),
        (pF, dQ, 0.7),
        (pC * o5, dE * dDOT, 0.7),
        (pF, dE, 0.4),
        (pF, dS, 0.4),
        (pDf * o5, dE, 0.4),
        (pC * o5, dE, 0.7),
        (pAf, dE, 0.7),
        (pF, dE, 0.7),
        (pC * o5, dE, 0.7),
        (pF * o5, dE, 0.7),
        (pF, dS, 0.7),
        (pEf, dE, 0.4),
        (pEf, dS, 0.4),
        (pC, dE, 0.7),
        (pG, dE, 0.7),
        (pF, dE + dW, 0.7),
    ]

    start = 0
    notes = []
    for freq, dur, legato in theme:
        # For each note, allocate an ADSR and a SuperSawPE with a delay and 
        # append on the notes[] list.  We will pass this list to MixPE() to
        # render the whole piece.
        #
        # About "gate" signals: a gate is on when its value is greater than 0.
        # The ADSR takes a gate input: rising gate initiates attack, falling
        # edge initiates decay.  In this case, the length of the gate is the 
        # duration of the note, scaled by the legato.  (A short legato will make 
        # a more staccato note.)
        gate_len = max(1, _samp(dur * legato))

        # Create a gate signal is 1.0 for the duration gate_len
        gate_stream = CropPE(ConstantPE(1.0), Extent(0, gate_len))

        # Apply the gate to the ADSR to generate the envelope signal
        envelope_stream = AdsrPE(
            gate_stream,
            attack_samples=stos(0.04),
            decay_samples=stos(0.2),
            sustain_level=0.5,
            release_samples=stos(0.1),
        )

        # Generate the Roland-esque detuned note
        pad_stream = SuperSawPE(frequency=freq, detune_cents=8.0, mix_mode="center_heavy")
        
        # === Subtle but important: ===
        # ResetPE sends pad.reset_start() at the onset of each new gate signal.
        # SuperSaw needs this so all of its detuned oscillators get restarted 
        # on each note - without this, the attack is mushy sounding. To hear the
        # difference, comment out this next line...
        pad_stream = ResetPE(pad_stream, gate_stream)

        # Take the synth note we've just created, apply the ADSR envelope, 
        # delay it to its designated start time and add it to the list of notes.
        notes.append(DelayPE(GainPE(pad_stream, envelope_stream), int(start)))

        # bump start to the next start time.
        start += int(dur)

    # mix all notes and attenuate some
    mix_stream = GainPE(MixPE(*notes), 0.5)

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    # Optional: Crop mix to a finite extent so we can stream it chunk-by-chunk. 
    # It works without this, but that would compute the entire piece before it 
    # starts to play, leading to a long lag before you hear anything.
    renderer.set_source(CropPE(mix_stream, Extent(0, int(start))))

    with renderer:
        renderer.start()
        # play_range() would render the whole buffer before playback, leading to
        # a long delay, so we use play_extent() instead to start rendering right
        # away.
        renderer.play_extent(chunk_size=renderer.blocksize)


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
        ("2", "Gated Sequence", demo_axel_f),
        ("3", "Re-trigger Behavior", demo_retrigger_behavior),
        ("a", "All demos", None),
    ]

    for key, name, _ in demos:
        print(f"{key}: {name}")
    print()

    choice = input("Choose a demo (0-3 or 'a' for all): ").strip().lower()
    print()

    if choice == "a":
        for _, _, func in demos[:-1]:
            func()
    else:
        for key, _, func in demos:
            if choice == key:
                func()
                break
        else:
            print("Invalid choice.")
