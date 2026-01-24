import numpy as np

from pygmu2 import (
    AudioRenderer,
    DiracPE,
    DelayPE,
    MixPE,
    RampPE,
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
    bass = WavReaderPE("examples/audio/bass.wav")
    drums = WavReaderPE("examples/audio/acoustic_drums.wav")
    
    # Create a trigger: A delayed Dirac pulse
    # The Dirac pulse happens at t=0, but we delay it by 1 second.
    # So the trigger happens at t=1.0s
    trigger_pulse = DiracPE()
    # Assume 44100 Hz for the example, though real usage should match renderer.
    # Or just use an integer for samples directly.
    # Let's assume standard rate for the example calculation.
    SAMPLE_RATE = 44100
    trigger = DelayPE(trigger_pulse, delay=seconds_to_samples(1.0, SAMPLE_RATE))
    
    # Wrap the drums in a TriggerPE
    # The drums will only start playing when the trigger arrives (at t=1.0s)
    triggered_drums = TriggerPE(drums, trigger, mode=TriggerMode.ONE_SHOT)
    
    # Mix the bass (playing immediately) with the triggered drums
    # We delay the triggered drums slightly less than the trigger time simply to show 
    # that the sound starts RELATIVE to the trigger time.
    # Actually, TriggerPE output is 0 until trigger time. So we just mix them.
    mix = MixPE(bass, triggered_drums)
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(mix)
    
    with renderer:
        renderer.start()
        renderer.play_range(0, seconds_to_samples(4.0, SAMPLE_RATE).astype(int))


def demo_gated_retrigger():
    """
    Demonstrate GATED triggering with restart.
    
    A sine wave with increasing frequency acts as a trigger source.
    Every time the sine wave crosses zero (goes positive), it restarts
    a percussive sample (djembe). As the frequency increases, the
    re-triggering gets faster, creating a "stutter" or "drill" effect.
    """
    print("\n=== Demo: GATED Retriggering ===")
    print("Using an LFO to repeatedly trigger a sample with increasing speed...")

    # Load a short percussive sound
    sample = WavReaderPE("examples/audio/djembe.wav")
    
    # We want to re-trigger this sample rhythmically.
    # We'll use a Sine wave as the LFO (Low Frequency Oscillator).
    # Its frequency will ramp up from 2 Hz to 20 Hz over 5 seconds.
    
    # 1. Create the frequency ramp
    freq_ramp = WindowPE(
        SinePE(frequency=0.0), # Dummy input, not used by WindowPE directly this way usually? 
        # Wait, WindowPE is for envelopes. Let's use linear interpolation or just a Sweep.
        # We can use a Wavetable or just simple math if we had a RampPE.
        # Let's use a SinePE to modulate the frequency of another SinePE (FM).
        # Or better, just a helper for linear ramp.
        # Actually we have RampPE? Let's check imports. No RampPE imported.
        # We can use a very slow SinePE as an LFO for frequency.
    )
    # Re-writing the frequency ramp concept using what we have.
    # We want f(t) to go from 2 to 20.
    # We can use a SinePE at 0Hz (DC offset) ? No, SinePE frequency is constant.
    # Wait, SinePE frequency CAN be a PE!
    
    # LFO Frequency control: A ramp from 2 to 20 over 5 seconds.
    # Since we don't have a simple Line/Ramp PE in the standard import list above (let me check),
    # I'll implement a quick ramp using a very slow sine wave segment or just add RampPE to imports if available.
    # I see RampPE in __init__.py from previous context. I will add it to imports.
        
    # Assume 44100 Hz
    SAMPLE_RATE = 44100
    lfo_freq = RampPE(start_value=2.0, end_value=20.0, duration=int(seconds_to_samples(5.0, SAMPLE_RATE)))
    
    # The Trigger LFO
    trigger_lfo = SinePE(frequency=lfo_freq)
    
    # The Gated Trigger
    # When LFO > 0, it plays. When LFO <= 0, it stops (and resets for next time).
    # This chops the sample, playing only the first half-cycle duration of the LFO.
    # As LFO speeds up, we hear shorter and shorter chunks of the start of the sample.
    stutter = TriggerPE(sample, trigger_lfo, mode=TriggerMode.GATED)
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(stutter)
    
    with renderer:
        renderer.start()
        renderer.play_range(0, seconds_to_samples(5.0, SAMPLE_RATE).astype(int))

def demo_gated_rhythm():
    """
    Demonstrate GATED mode for rhythmic patterns.
    
    We create a simple sequencer using a WavetablePE as a control signal
    to gate a continuous sound (like a choir or pad).
    """
    print("\n=== Demo: GATED Rhythm ===")
    print("Gating a choir pad with a rhythmic control pattern...")
    
    choir = WavReaderPE("examples/audio/choir.wav")
    sample_rate = choir.file_sample_rate
    
    # Create a rhythmic pattern: 1 0 1 1 0 1 0 0 (1=On, 0=Off)
    # We'll use a WavetablePE to loop this pattern.
    pattern = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    # Note: TriggerPE uses >0 for ON. So -1 is OFF.
    
    # Create a sequencer by driving a WavetablePE with a phasor (RampPE)
    # 1. Create the data source (the sequence pattern)
    pattern_source = ArrayPE(pattern)
    
    # 2. Create the indexer (Phasor)
    # We want to scan through the 8-step pattern at 2 Hz.
    # Cycle duration = sample_rate / 2.0
    
    from pygmu2 import RampPE, LoopPE
    
    cycle_samples = int(sample_rate / 2.0)
    
    # Create a single ramp from 0 to 8 (length of pattern)
    one_cycle = RampPE(start_value=0.0, end_value=len(pattern), duration=cycle_samples)
    
    # Loop it indefinitely to create a phasor
    phasor = LoopPE(one_cycle)
    
    # 3. Create the Wavetable Lookup
    sequencer = WavetablePE(
        wavetable=pattern_source,
        indexer=phasor,
        interpolation=InterpolationMode.LINEAR,
        out_of_bounds=OutOfBoundsMode.WRAP
    )
    
    gated_pad = TriggerPE(choir, sequencer, mode=TriggerMode.GATED)
    
    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(gated_pad)
    
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
