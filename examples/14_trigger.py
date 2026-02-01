import numpy as np

from pygmu2 import (
    AudioRenderer,
    DiracPE,
    DelayPE,
    MixPE,
    PiecewisePE,
    SinePE,
    TriggerPE,
    TriggerMode,
    WavReaderPE,
    WavetablePE,
    InterpolationMode,
    OutOfBoundsMode,
    WindowPE,
    LoopPE,
    WindowMode,
    seconds_to_samples,
    GainPE,
    Extent,
    ArrayPE
)

def demo_one_shot_trigger():
    """
    Demonstrate ONE_SHOT triggering.
    
    A bass line starts playing. After a short delay, a Dirac pulse triggers
    a second sound (a drum beat) mixed in with the first.
    """
    print("=== Demo: ONE_SHOT Triggering ===")
    print("Playing a bass line, then triggering a drum beat...")

    # Load sounds
    bass_stream = WavReaderPE("examples/audio/bass.wav")
    drums_stream = WavReaderPE("examples/audio/acoustic_drums.wav")
    
    # Create a trigger: A delayed Dirac pulse
    # The Dirac pulse happens at t=0, but we delay it by 1 second.
    # So the trigger happens at t=1.0s
    trigger_pulse_stream = DiracPE()
    # Assume 44100 Hz for the example, though real usage should match renderer.
    # Or just use an integer for samples directly.
    # Let's assume standard rate for the example calculation.
    SAMPLE_RATE = 44100
    trigger_stream = DelayPE(trigger_pulse_stream, delay=seconds_to_samples(1.0, SAMPLE_RATE))
    
    # Wrap the drums in a TriggerPE
    # The drums will only start playing when the trigger arrives (at t=1.0s)
    triggered_drums_stream = TriggerPE(drums_stream, trigger_stream, trigger_mode=TriggerMode.ONE_SHOT)
    
    # Mix the bass (playing immediately) with the triggered drums
    # We delay the triggered drums slightly less than the trigger time simply to show 
    # that the sound starts RELATIVE to the trigger time.
    # Actually, TriggerPE output is 0 until trigger time. So we just mix them.
    mix_stream = MixPE(bass_stream, triggered_drums_stream)
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(mix_stream)
    
    with renderer:
        renderer.start()
        renderer.play_range(0, seconds_to_samples(4.0, SAMPLE_RATE).astype(int))


def demo_gated_retrigger():
    """
    Demonstrate GATED triggering.

    A sine LFO (2 Hz -> 20 Hz over 5 s) gates a djembe sample. Sample plays
    while LFO > 0 and stops when LFO <= 0. GATED does not retrigger when the
    gate goes high again (one segment per run).
    """
    print("\n=== Demo: GATED ===")
    print("Sample plays while LFO > 0, stops when LFO <= 0 (no retrigger).")

    sample_stream = WavReaderPE("examples/audio/djembe.wav")
    sample_rate = 44100
    dur = int(seconds_to_samples(5.0, sample_rate))

    # LFO frequency ramps 2 Hz -> 20 Hz over the demo duration
    lfo_freq = PiecewisePE([(0, 2.0), (dur, 20.0)])
    trigger_lfo = SinePE(frequency=lfo_freq)
    stutter_stream = TriggerPE(sample_stream, trigger_lfo, trigger_mode=TriggerMode.GATED)

    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(stutter_stream)
    with renderer:
        renderer.start()
        renderer.play_range(0, dur)

def demo_gated_rhythm():
    """
    Demonstrate RETRIGGER mode for rhythmic patterns.

    An 8-step pattern (1 0 1 1 0 1 0 0) at 2 Hz gates a choir pad. Each time
    the pattern goes high, the pad retriggers from the start (RETRIGGER mode).
    """
    print("\n=== Demo: RETRIGGER Rhythm ===")
    print("Choir pad retriggers from the start on each pattern step (RETRIGGER).")
    
    choir_stream = WavReaderPE("examples/audio/choir.wav")
    sample_rate = choir_stream.file_sample_rate
    
    # Create a rhythmic pattern: 1 0 1 1 0 1 0 0 (1=On, 0=Off)
    # We'll use a WavetablePE to loop this pattern.
    pattern = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    # Note: TriggerPE uses >0 for ON. So -1 is OFF.
    
    # Create a sequencer by driving a WavetablePE with a phasor (PiecewisePE)
    # 1. Create the data source (the sequence pattern)
    pattern_source_stream = ArrayPE(pattern)
    
    # 2. Create the indexer (phasor): ramp 0..8 over one cycle, loop at 2 Hz
    cycle_samples = int(sample_rate / 2.0)

    # Create a single ramp from 0 to 8 (length of pattern)
    one_cycle_stream = PiecewisePE([(0, 0.0), (cycle_samples, len(pattern))])
    
    # Loop it indefinitely to create a phasor
    phasor_stream = LoopPE(one_cycle_stream)
    
    # 3. Create the Wavetable Lookup
    sequencer_stream = WavetablePE(
        wavetable=pattern_source_stream,
        indexer=phasor_stream,
        interpolation=InterpolationMode.LINEAR,
        out_of_bounds=OutOfBoundsMode.WRAP
    )
    
    gated_pad_stream = TriggerPE(choir_stream, sequencer_stream, trigger_mode=TriggerMode.RETRIGGER)
    
    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(gated_pad_stream)
    
    with renderer:
        renderer.start()
        renderer.play_range(0, seconds_to_samples(4.0, sample_rate).astype(int))

if __name__ == "__main__":
    try:
        demo_one_shot_trigger()
        demo_gated_retrigger()
        demo_gated_rhythm()
    except KeyboardInterrupt:
        pass
