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
uv run python examples/01_hello_sine.py
```

### Using pipenv

```bash
# Install pipenv if needed
pip install pipenv

# Install dependencies
pipenv install

# Activate environment
pipenv shell
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
| `AnalogOscPE` | Bandlimited analog-style oscillator (PWM rectangle + duty-controlled saw/triangle morph). |
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
| `CropPE` | A ProcessingElement that limits its input to a specified range. |
| `DecayingSinePE` | Exponentially decaying sine tone. |
| `DelayPE` | A ProcessingElement that delays its input by a specified amount. |
| `DiracPE` | A SourcePE that outputs a unit impulse (Dirac delta in discrete time). |
| `DynamicsPE` | Flexible dynamics processor that applies compression, limiting,. |
| `EnvelopePE` | Causal envelope follower with attack/release dynamics and optional lookahead. |
| `ExpanderPE` | Downward expander / noise gate — attenuates signal below threshold. |
| `FunctionGenPE` | Naive function generator (no anti-aliasing). |
| `GainPE` | A ProcessingElement that applies gain (amplitude scaling) to its input. |
| `GateToTriggerPE` | Converts a GateSignal to a TriggerSignal by emitting +1 at each. |
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
| `PeriodicGate` | A GateSignal that emits a periodic rectangular gate (0/1), with fixed or. |
| `PeriodicTrigger` | A TriggerSignal that emits +1 impulses periodically. |
| `PiecewisePE` | A SourcePE that outputs a piecewise curve defined by (sample_index, value) points. |
| `RandomGatePE` | Poisson-process toggle gate. |
| `RandomSelectPE` | On each positive trigger event, randomly selects one of N input PEs, then. |
| `RandomStepPE` | Poisson sample-and-hold random generator. |
| `RandomTriggerPE` | Poisson-process trigger generator. |
| `RandomValuePE` | Continuously wandering random voltage in [0, 1]. |
| `ResamplePE` | Pure constant-rate resampling of a source PE. |
| `ReverbPE` | Convolution reverb with a wet/dry mix control. |
| `ReversePitchEchoPE` | Pitch-shifted reverse echo effect. |
| `RingModulatorPE` | A ProcessingElement that ring-modulates a carrier signal with a modulator signal. |
| `SVFilterPE` | Second-order state variable filter with the same API as BiquadPE. |
| `SampleHoldPE` | Sample-and-Hold processing element. |
| `ScheduledGatePE` | Convert note durations into gate signals, specifically for feeding into an. |
| `SequencePE` | Schedule PEs at specific start times. |
| `SetExtentPE` | Force a PE to a specified extent, padding or truncating as needed. |
| `SignalToGatePE` | Schmitt-trigger gate: converts an analog signal to a gate signal. |
| `SinePE` | A ProcessingElement that generates a sine wave. |
| `SlewLimiterPE` | Slew-rate limiter for control signals. |
| `SlicePE` | Extract a region from a source and shift it to start at time 0. |
| `SpatialPE` | Spatial audio processing and channel conversion PE. |
| `SuperSawPE` | Detuned unison sawtooth oscillator for warm, analog-like sounds. |
| `TimeWarpPE` | Resample a source at a time-varying rate. |
| `TrackHoldPE` | Track-and-Hold processing element. |
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
uv run python examples/01_hello_sine.py

# Using pipenv:
pipenv run python examples/01_hello_sine.py
```

<!-- BEGIN GENERATED: examples-table (scripts/gen_readme_tables.py) -->
| Example | Description |
|---------|-------------|
| `00_template_eg.py` | 00_template_eg.py. |
| `01_hello_sine.py` | Example 01: Hello Sine - Introduction to pygmu2. |
| `02_play_wav.py` | Example 02: Play WAV - Loading and playing audio files. |
| `03_looping.py` | Example 03: Looping - Repeating audio segments. |
| `04_filtering.py` | Example 04: Filtering - Biquad filter with frequency sweep. |
| `05_flanging.py` | Example 05: Flanging - Time-varying delay effect. |
| `06_autowah.py` | Example 06: Autowah - Envelope-controlled filter. |
| `07_soft_clipping.py` | Example 07: Soft Clipping - TransformPE with saturation. |
| `08_write_to_file.py` | Example 08: Write to File - Offline rendering to WAV. |
| `10_compression.py` | Example 10: Compression, Limiting, and Gating. |
| `11_dynamics.py` | Example 11: Advanced Dynamics with DynamicsPE. |
| `12_audio_library.py` | Example 12: Strudel Audio Library - Lazy downloading and playback. |
| `15_reverse_pitch_echo.py` | Example 15: Reverse Pitch Echo - block-based reverse playback. |
| `16_comb_filter.py` | Example 16: Comb Filter - pitched resonance. |
| `17_ladder_filter.py` | Example 17: Ladder Filter - Moog-style ladder responses. |
| `19_sequence_examples.py` | Example 19: Sequencing with MixPE, CropPE, SlicePE, DelayPE, PiecewisePE. |
| `20_alternative_temperaments.py` | Alternative Temperaments Example. |
| `20_timewarp.py` | Example 20: TimeWarpPE - variable-speed playback ("tape head"). |
| `21_analog_osc.py` | Example 21: AnalogOscPE - bandlimited PWM + saw/triangle morph oscillator. |
| `22_function_gen.py` | Example 22: FunctionGenPE - naive DSP-like function generator (aliased). |
| `23_convolution.py` | Example 23: ConvolvePE - convolution reverb (room impulse responses). |
| `24_slice.py` | Example 24: SlicePE - quick snippet audition framework. |
| `27_spatial.py` | Example 27: Spatial Audio - Panning and Channel Conversion. |
| `29_karplus_strong.py` | Example 29: Karplus-Strong plucked string synthesis. |
| `33_piecewise.py` | Example 33: PiecewisePE - piecewise (sample_index, value) curves. |
| `37_sequence_eg.py` | 37_sequence_eg.py. |
| `adsr_eg.py` | adsr_eg.py  ADSR demos using the new GateSignal / TriggerSignal ADSR classes. |
| `audio_reader_eg.py` | audio_reader_eg.py. |
| `audio_slew_rate_limit_eg.py` | Example: Slew Rate Limiting on a stringed instrument. |
| `bwv1007_eg.py` | Fun with Tralfam.  And Bach.  And Yo Yo Ma. |
| `decaying_sine_eg.py` | Decaying sine tone synthesis — tau refactor. |
| `demo_asset_manager.py` | demo_asset_mgr.py. |
| `envelope_filter_eg.py` | envelope_filter_eg.py — Envelope-controlled filter: louder hits sound brighter. |
| `fold_4K_test.py` | what does it sound like when you ring modulate a sound with high frequency. |
| `function_generator_eg.py` | Function generator outputs for teaching. |
| `idiophone_eg.py` | Struck bar idiophone synthesis using IdiophonePE. |
| `im_lucky.py` | I'm Lucky -- a pygmu2 re-creation of Thomas Dolby's synth part in Joan. |
| `mag_freq_eg.py` | mag_freq_eg.py — FFT-domain magnitude and phase manipulation via MagFreqPE. |
| `notes_eg.py` | notes_eg.py — NotesPE: play MIDI notes from a source sample. |
| `random_gate_eg.py` | Example: Random Gate - Poisson-process toggle gate. |
| `random_select_eg.py` | RandomSelectPE example (new TriggerSignal/GateSignal conventions):. |
| `random_step2_eg.py` | Example: Random Step 2 — Musical ratios with parallel stepped LPF and slew. |
| `random_step_eg.py` | Example: Random Step - Poisson sample-and-hold random generator. |
| `random_trigger_eg.py` | Example: RandomTriggerPE - Poisson-process trigger generator. |
| `random_value_eg.py` | Example: RandomValuePE - continuously wandering random voltage generator. |
| `reverb_eg.py` | reverb_eg.py. |
| `ring_modulator_eg.py` | Example: Ring Modulator — Sideband synthesis and vocal morphing. |
| `sum_of_sines_eg.py` | Example: Demo sine summation for popular waveforms (square, triangle, saw, pulse). |
| `super_saw_eg.py` | Example: Super Saw - Rich, detuned unison oscillator. |
| `tralfam_eg.py` | tralfam_eg.py. |
| `wargle_eg.py` | etude #12: a funny sounding instrument. |
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

See `examples/20_alternative_temperaments.py` for a detailed demonstration.

## Running Tests

```bash
# Using uv
uv run pytest
uv run pytest --cov=src --cov-report=html  # With coverage

# Using pipenv
pipenv run pytest
pipenv run pytest --cov=src --cov-report=html  # With coverage
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
