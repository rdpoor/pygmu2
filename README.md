# pygmu2

A Python framework for generating and processing digital audio, with a bias towards generating music.

## Overview

pygmu2 provides a flexible, composable architecture for building audio processing pipelines. Audio is generated on-demand through a directed acyclic graph (DAG) of Processing Elements (PEs), enabling efficient processing of long or infinite streams.

**Key Features:**
- Lazy evaluation: audio generated on-demand
- Composable design: PEs connect to form complex audio graphs
- Rich library of oscillators, filters, effects, and dynamics processors
- Alternative temperament support (12-ET, 19-ET, just intonation, Pythagorean, custom)
- Cross-platform audio playback via `sounddevice`
- WAV file I/O via `soundfile`

## Installation

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager:

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run commands in the virtual environment
uv run python examples/hello_sine_eg.py
```

## Quick Start

```python
import pygmu2 as pg
from pygmu2 import SinePE

# Set global sample rate before constructing any PEs
pg.set_sample_rate(44100)

# Create a 440 Hz sine wave
sine_stream = SinePE(frequency=440.0, amplitude=0.5)

# Play through speakers
with pg.AudioRenderer() as renderer:
    renderer.set_source(sine_stream)
    renderer.start()
    renderer.play_range(0, 44100 * 3)  # Play 3 seconds
```

## Core Concepts

### Sample Rate

The global sample rate must be set before constructing any Processing Elements:

```python
import pygmu2 as pg
pg.set_sample_rate(44100)   # 44.1 kHz — call this first
```

Every PE captures the sample rate at construction time. If `set_sample_rate()` has not been called, `ProcessingElement.__new__()` raises a `RuntimeError`. Renderers (`AudioRenderer`, `NullRenderer`) also read the global rate by default.

Use the convenience functions in `pg.utils` for common tasks — they read the global rate automatically:

```python
pg.play(source)                        # real-time playback
pg.play_offline(source)                # render to WAV then play
pg.browse(source)                      # render to WAV, open in jog/shuttle player
pg.render_to_file(source, "out.wav")   # render to WAV file
```

### Processing Elements (PEs)

A **Processing Element** is the fundamental building block. Each PE generates or transforms audio:

```python
# Source PE: generates audio
sine_stream = SinePE(frequency=440.0)

# Transform PE: processes audio from another PE
quieter_stream = GainPE(sine_stream, gain=0.5)

# Combine PEs: mix multiple sources
mix_stream = MixPE(sine1_stream, sine2_stream, sine3_stream)
```

### Snippets

A **Snippet** is a chunk of audio samples with a start position:

```python
snippet = pe.render(start=0, duration=1024)
# snippet.data: numpy array of shape (samples, channels)
# snippet.start: sample index where this chunk begins
```

### Extents

An **Extent** defines the temporal bounds of a PE's output:

```python
# Finite extent (e.g., a WAV file)
crop_stream = CropPE(source_stream, 0, 44100)  # First second only

# Infinite extent (e.g., oscillators)
sine_stream = SinePE(frequency=440.0)  # Extent(None, None) - plays forever
```

### Renderers

A **Renderer** pulls audio from the PE graph. The sample rate is read from the global `set_sample_rate()` unless explicitly provided:

```python
# Play to speakers
with AudioRenderer() as renderer:
    renderer.set_source(my_pe)
    renderer.start()
    renderer.play_range(0, 44100 * 5)

# Or render silently (for testing/processing)
renderer = NullRenderer()
```

## Available Processing Elements

The table below is generated from the code (`scripts/gen_readme_tables.py`);
descriptions are each PE's first docstring line. Do not edit by hand — CI
checks that it matches the source.

<!-- BEGIN GENERATED: pe-table (scripts/gen_readme_tables.py) -->
| PE | Description |
|----|-------------|
| `AdsrGatedPE` | Gate-driven ADSR envelope generator. |
| `AdsrTriggeredPE` | Trigger-driven one-shot ADSR envelope generator. |
| `AnalogOscPE` | Analog-style oscillator: PWM rectangle + duty-controlled saw/triangle. |
| `ArrayPE` | A SourcePE that outputs values from a provided array. |
| `AudioReaderPE` | A SourcePE that decodes a compressed audio file (MP3, FLAC, OGG, WAV). |
| `BiquadPE` | Second-order IIR (biquad) filter. |
| `BlitSawPE` | Band-limited sawtooth oscillator using BLIT synthesis. |
| `CachePE` | Single-entry render cache for a source PE. |
| `CombPE` | Feedback comb filter tuned by a target frequency. |
| `CompressorPE` | All-in-one audio compressor with integrated envelope follower. |
| `ConstantPE` | A SourcePE that outputs a constant value. |
| `ControlPE` | A SourcePE whose output value can be changed at any time from any thread. |
| `ConvolvePE` | Streaming convolution: y = x * h. |
| `CropPE` | A ProcessingElement that imposes a time window on its input. |
| `DecayingSinePE` | Exponentially decaying sine tone. |
| `DelayPE` | A ProcessingElement that delays its input by a specified amount. |
| `DiracPE` | A SourcePE that outputs a unit impulse (Dirac delta in discrete time). |
| `DynamicsPE` | Flexible dynamics processor that applies compression, limiting,. |
| `EnvelopePE` | Causal envelope follower with attack/release dynamics and optional lookahead. |
| `ExpanderPE` | Downward expander / noise gate — attenuates signal below threshold. |
| `GainPE` | A ProcessingElement that applies gain (amplitude scaling) to its input. |
| `GateToTriggerPE` | Converts a GateSignal to a TriggerSignal by emitting +1 at each. |
| `HoldPE` | Follow `source` while `control` > 0; hold the last value while it is 0. |
| `IdentityPE` | A SourcePE that outputs the sample index as the sample value. |
| `IdiophonePE` | Struck bar idiophone synthesis PE. |
| `KarplusStrongPE` | Plucked string using the classic Karplus-Strong algorithm. |
| `LadderPE` | Moog-style ladder filter with non-linear saturation. |
| `LimiterPE` | Brick-wall limiter — prevents signal from exceeding a ceiling level. |
| `LoopPE` | Repeat a segment of audio from the source. |
| `MagFreqPE` | PE that modifies the magnitude and phase of a source in the frequency. |
| `MeltysynthPE` | Source PE that renders meltysynth SoundFont synthesis into stereo Snippets. |
| `MidiInPE` | Source PE that receives MIDI input via Mido and exposes messages via callback. |
| `MixPE` | A ProcessingElement that mixes (adds) multiple PE outputs together. |
| `MovingAveragePE` | Pure box-filter low-pass via a sliding window mean. |
| `NoisePE` | Noise generator. |
| `NotesPE` | Render a list of Notes from a single source sample. |
| `PeriodicGatePE` | A GateSignal that emits a periodic rectangular gate (0/1), with fixed or. |
| `PiecewisePE` | A SourcePE that outputs a piecewise curve defined by (sample_index, value) points. |
| `RandomGatePE` | Poisson-process toggle gate. |
| `RandomSelectPE` | On each positive trigger event, randomly selects one of N input PEs, then. |
| `RandomValuePE` | Continuously wandering random voltage in [0, 1]. |
| `ResamplePE` | Pure constant-rate resampling of a source PE. |
| `ReverbPE` | Convolution reverb with a wet/dry mix control. |
| `ReversePitchEchoPE` | Pitch-shifted reverse echo effect. |
| `RingModulatorPE` | A ProcessingElement that ring-modulates a carrier signal with a modulator signal. |
| `SVFilterPE` | Second-order state variable filter with the same API as BiquadPE. |
| `ScheduledGatePE` | Convert note (start, duration) pairs into a gate signal: 1.0 while a. |
| `SequencePE` | Schedule PEs at specific start times. |
| `SignalToGatePE` | Schmitt-trigger gate: converts an analog signal to a gate signal. |
| `SinePE` | A ProcessingElement that generates a sine wave. |
| `SlewLimiterPE` | Slew-rate limiter for control signals. |
| `SlicePE` | Extract a region from a source and shift it to start at time 0. |
| `SpatialPE` | Spatial audio processing and channel conversion PE. |
| `SuperSawPE` | Detuned unison sawtooth oscillator for warm, analog-like sounds. |
| `TimeWarpPE` | Resample a source at a time-varying rate. |
| `TralfamPE` | PE that spreads a finite source's spectrum randomly across its time span. |
| `TransformPE` | Apply an arbitrary transformation function to audio samples. |
| `TriggerRestartPE` | Trigger-controlled restart/time-remap. |
| `WavReaderPE` | A SourcePE that reads audio samples from a WAV file. |
| `WavWriterPE` | A ProcessingElement that writes audio to a WAV file as a side effect. |
| `WavetablePE` | Wavetable lookup synthesis with interpolation. |
| `WindowPE` | Bidirectional windowed statistics - computes statistics over a symmetric. |
<!-- END GENERATED: pe-table -->

## Examples

The `examples/` directory contains runnable demos:

```bash
# Using uv:
uv run python examples/hello_sine_eg.py

```

<!-- BEGIN GENERATED: examples-table (scripts/gen_readme_tables.py) -->
| Example | Description |
|---------|-------------|
| `adsr_eg.py` | adsr_eg.py  ADSR demos using the new GateSignal / TriggerSignal ADSR classes. |
| `analog_osc_eg.py` | Example 21: AnalogOscPE - bandlimited PWM + saw/triangle morph oscillator. |
| `asset_manager_eg.py` | demo_asset_mgr.py. |
| `audio_library_eg.py` | Strudel Audio Library - Lazy downloading and playback. |
| `audio_reader_eg.py` | audio_reader_eg.py. |
| `audio_slew_rate_limit_eg.py` | Example: Slew Rate Limiting on a stringed instrument. |
| `autowah_eg.py` | Autowah - Envelope-controlled filter. |
| `bwv1007_eg.py` | Fun with Tralfam.  And Bach.  And Yo Yo Ma. |
| `comb_filter_eg.py` | Comb Filter - pitched resonance. |
| `compression_eg.py` | Example 10: Compression, Limiting, and Gating. |
| `convolution_eg.py` | Example 23: ConvolvePE - convolution reverb (room impulse responses). |
| `decaying_sine_eg.py` | Decaying sine tone synthesis — tau refactor. |
| `dynamics_eg.py` | Example 11: Advanced Dynamics with DynamicsPE. |
| `envelope_filter_eg.py` | envelope_filter_eg.py — Envelope-controlled filter: louder hits sound brighter. |
| `filtering_eg.py` | Filtering - Biquad filter with frequency sweep. |
| `flanging_eg.py` | Flanging - Time-varying delay effect. |
| `fold_4k_eg.py` | what does it sound like when you ring modulate a sound with high frequency. |
| `function_gen_aliasing_eg.py` | AnalogOscPE antialias=False - the naive (aliased) oscillator mode. |
| `function_generator_eg.py` | Function generator outputs for teaching. |
| `hello_sine_eg.py` | Hello Sine - Introduction to pygmu2. |
| `idiophone_eg.py` | Struck bar idiophone synthesis using IdiophonePE. |
| `im_lucky_eg.py` | I'm Lucky -- a pygmu2 re-creation of Thomas Dolby's synth part in Joan. |
| `karplus_strong_eg.py` | Example 29: Karplus-Strong plucked string synthesis. |
| `ladder_filter_eg.py` | Ladder Filter - Moog-style ladder responses. |
| `looping_eg.py` | Looping - Repeating audio segments. |
| `mag_freq_eg.py` | mag_freq_eg.py — FFT-domain magnitude and phase manipulation via MagFreqPE. |
| `notes_eg.py` | notes_eg.py — NotesPE: play MIDI notes from a source sample. |
| `piecewise_eg.py` | Example 33: PiecewisePE - piecewise (sample_index, value) curves. |
| `play_wav_eg.py` | Play WAV - Loading and playing audio files. |
| `random_gate_eg.py` | Example: Random Gate - Poisson-process toggle gate. |
| `random_select_eg.py` | RandomSelectPE example (new TriggerSignal/GateSignal conventions):. |
| `random_step2_eg.py` | Example: Random Step 2 — Musical ratios with parallel stepped LPF and slew. |
| `random_step_eg.py` | Example: Random Step - Poisson sample-and-hold random generator. |
| `random_trigger_eg.py` | Example: random triggers, derived from a random gate. |
| `random_value_eg.py` | Example: RandomValuePE - continuously wandering random voltage generator. |
| `reverb_eg.py` | reverb_eg.py. |
| `reverse_pitch_echo_eg.py` | Reverse Pitch Echo - block-based reverse playback. |
| `ring_modulator_eg.py` | Example: Ring Modulator — Sideband synthesis and vocal morphing. |
| `sequence_eg.py` | SequencePE example showing OVERLAP and NON_OVERLAP modes with audio material. |
| `sequencing_techniques_eg.py` | Example 19: Sequencing with MixPE, CropPE, SlicePE, DelayPE, PiecewisePE. |
| `slice_eg.py` | SlicePE - quick snippet audition framework. |
| `soft_clipping_eg.py` | Soft Clipping - TransformPE with saturation. |
| `spatial_eg.py` | Spatial Audio - Panning and Channel Conversion. |
| `sum_of_sines_eg.py` | Example: Demo sine summation for popular waveforms (square, triangle, saw, pulse). |
| `super_saw_eg.py` | Example: Super Saw - Rich, detuned unison oscillator. |
| `temperaments_eg.py` | Alternative Temperaments Example. |
| `template_eg.py` | Starting template for new pygmu2 examples. |
| `timewarp_eg.py` | Example 20: TimeWarpPE - variable-speed playback ("tape head"). |
| `tralfam_eg.py` | tralfam_eg.py. |
| `wargle_eg.py` | etude #12: a funny sounding instrument. |
| `write_to_file_eg.py` | Write to File - Offline rendering to WAV. |
<!-- END GENERATED: examples-table -->

## Modulation and Automation

Many PE parameters accept either a constant value or another PE. Both of
SinePE's main parameters are modulatable, which makes vibrato and tremolo
mirror images of each other — drive `frequency` with an LFO for one,
`amplitude` for the other:

```python
# Vibrato: frequency = 440 ± 10 Hz, wobbling at 5 Hz
vibrato_lfo = SinePE(frequency=5.0, amplitude=10.0)
vibrato_stream = SinePE(frequency=MixPE(ConstantPE(440.0), vibrato_lfo))

# Tremolo: amplitude = 0.7 ± 0.3, pulsing at 4 Hz
tremolo_lfo = SinePE(frequency=4.0, amplitude=0.3)
tremolo_stream = SinePE(frequency=440.0, amplitude=MixPE(ConstantPE(0.7), tremolo_lfo))
```

The `MixPE(ConstantPE(center), lfo)` idiom adds a DC offset to a bipolar
LFO — here it keeps the tremolo gain within [0.4, 1.0], so the signal
never inverts or fully gates. The same pattern modulates any PE-valued
parameter: filter frequency, gain, delay time, and so on.

## Alternative Temperaments

pygmu2 supports multiple tuning systems (temperaments) and reference frequencies:

### Temperaments

```python
from pygmu2 import (
    pitch_to_freq,
    EqualTemperament,
    JustIntonation,
    PythagoreanTuning,
    set_temperament
)

# Use 19-tone equal temperament
et19 = EqualTemperament(19)
freq = pitch_to_freq(69, temperament=et19)  # A4 in 19-ET

# Use 5-limit just intonation (pure harmonic ratios)
ji = JustIntonation()
freq = pitch_to_freq(64, temperament=ji)  # E4 with pure major third

# Use Pythagorean tuning (based on perfect 3:2 fifths)
pyth = PythagoreanTuning()
freq = pitch_to_freq(67, temperament=pyth)  # G4 with pure fifth

# Set a global default temperament
set_temperament(EqualTemperament(19))
freq = pitch_to_freq(60)  # Now uses 19-ET globally
```

**Available Temperaments:**
- `EqualTemperament(divisions)` - N-tone equal temperament (12-ET, 19-ET, 24-ET, etc.)
- `JustIntonation(ratios)` - Just intonation with pure harmonic ratios
- `PythagoreanTuning()` - 3-limit tuning based on perfect fifths
- `CustomTemperament(...)` - Define your own tuning system

### Reference Frequency

Change the reference pitch (A4 defaults to 440 Hz):

```python
from pygmu2 import (
    set_reference_frequency,
    set_concert_pitch,
    set_verdi_tuning,
    set_baroque_pitch,
    pitch_to_freq
)

# A4 = 432 Hz (Verdi/philosophical pitch)
set_verdi_tuning()
freq = pitch_to_freq(69)  # 432.0 Hz

# A4 = 415 Hz (Baroque pitch)
set_baroque_pitch()
freq = pitch_to_freq(69)  # 415.0 Hz

# A4 = 440 Hz (concert pitch, default)
set_concert_pitch()

# Custom reference frequency
set_reference_frequency(442.0)  # Some orchestras tune to A=442
```

See `examples/temperaments_eg.py` for a detailed demonstration.

## Running Tests

```bash
# Using uv
uv run pytest
uv run pytest --cov=src --cov-report=html  # With coverage

```

## Troubleshooting

### SSL Certificate Errors (macOS)

If you see `ssl.SSLCertVerificationError` when using `AudioLibrary.from_url()`, this is a common issue with Python installed from python.org on macOS. Fix it by running:

```bash
# Option 1: Run the certificate installer (in Finder)
# Applications → Python 3.x → "Install Certificates.command"

# Option 2: Install certifi
uv add certifi   # or: pip install certifi
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, architecture details, and how to create new Processing Elements.

## License

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License - see [LICENSE](LICENSE) for details.
