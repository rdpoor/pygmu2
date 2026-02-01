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
from pygmu2 import SinePE, GainPE, AudioRenderer

# Create a 440 Hz sine wave
sine_stream = SinePE(frequency=440.0, amplitude=0.5)

# Play through speakers
with AudioRenderer(sample_rate=44100) as renderer:
    renderer.set_source(sine_stream)
    renderer.start()
    renderer.play_range(0, 44100 * 3)  # Play 3 seconds
```

## Core Concepts

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
crop_stream = CropPE(source_stream, Extent(0, 44100))  # First second only

# Infinite extent (e.g., oscillators)
sine_stream = SinePE(frequency=440.0)  # Extent(None, None) - plays forever
```

### Renderers

A **Renderer** pulls audio from the PE graph:

```python
# Play to speakers
with AudioRenderer(sample_rate=44100) as renderer:
    renderer.set_source(my_pe)
    renderer.start()
    renderer.play_range(0, 44100 * 5)

# Or render silently (for testing/processing)
renderer = NullRenderer(sample_rate=44100)
```

## Available Processing Elements

### Oscillators
| PE | Description |
|----|-------------|
| `SinePE(frequency, amplitude, phase)` | Sine wave (supports modulation) |
| `BlitSawPE(frequency, amplitude, m)` | Band-limited sawtooth (alias-free) |
| `AnalogOscPE(frequency, duty_cycle, waveform)` | Bandlimited PWM rectangle + duty-controlled saw/triangle morph |
| `SuperSawPE(frequency, voices, detune_cents)` | Detuned unison sawtooth |
| `FunctionGenPE(frequency, duty_cycle, waveform)` | Naive DSP-like rectangle + duty-controlled saw/triangle morph (aliased) |
| `WavetablePE(wavetable, indexer)` | Wavetable oscillator |

### Sources
| PE | Description |
|----|-------------|
| `ConstantPE(value, channels)` | Constant value |
| `PiecewisePE(points, transition_type)` | Piecewise curve (replaces RampPE) |
| `DiracPE(channels)` | Unit impulse |
| `IdentityPE(channels)` | Sample index as output |
| `WavReaderPE(path)` | Read from WAV file |

### Transforms
| PE | Description |
|----|-------------|
| `GainPE(source, gain)` | Apply gain (supports automation) |
| `MixPE(*sources)` | Sum multiple sources |
| `DelayPE(source, delay)` | Delay by N samples |
| `CropPE(source, extent)` | Limit to temporal range |
| `LoopPE(source, loop_start=None, loop_end=None, count=None, crossfade_seconds=None, crossfade_samples=None)` | Loop a finite source (optional crossfade in seconds or samples) |
| `SlicePE(source, start, duration, fade_in_samples=None, fade_in_seconds=None, fade_out_samples=None, fade_out_seconds=None)` | Extract a region and shift to time 0 (optional fades) |
| `ConvolvePE(src, filter, fft_size=None)` | FFT-based streaming convolution with a finite FIR filter |
| `TransformPE(source, func)` | Apply custom function |
| `ReversePitchEchoPE(source, block_seconds, pitch_ratio, ...)` | Pitch-shifted reverse echo effect |

### Filters
| PE | Description |
|----|-------------|
| `BiquadPE(source, mode, frequency, q)` | Biquad filter (lowpass, highpass, bandpass, etc.) |
| `LadderPE(source, frequency, resonance, mode)` | Moog-style ladder filter (lp/bp/hp 12/24 dB) |
| `CombPE(source, frequency, feedback)` | Feedback comb filter tuned by frequency |

### Dynamics
| PE | Description |
|----|-------------|
| `CompressorPE(source, threshold, ratio, ...)` | All-in-one compressor |
| `LimiterPE(source, ceiling, ...)` | Brick-wall limiter |
| `GatePE(source, threshold, ...)` | Noise gate |
| `DynamicsPE(source, envelope, ...)` | Flexible dynamics (sidechain support) |
| `EnvelopePE(source, attack, release)` | Envelope follower |

### Control
| PE | Description |
|----|-------------|
| `AdsrPE(gate, attack_samples=None, attack_seconds=None, decay_samples=None, decay_seconds=None, sustain_level=0.7, release_samples=None, release_seconds=None)` | ADSR envelope generator (defaults specified in seconds; resolved at configure time) |

### Analysis
| PE | Description |
|----|-------------|
| `WindowPE(source, window, mode)` | Windowed statistics (max, rms, mean) |

### Output
| PE | Description |
|----|-------------|
| `WavWriterPE(source, path)` | Write to WAV file |

## Examples

The `examples/` directory contains runnable demos:

```bash
# Using uv:
uv run python examples/01_hello_sine.py

# Using pipenv:
pipenv run python examples/01_hello_sine.py
```

| Example | Description |
|---------|-------------|
| `01_hello_sine.py` | Simple sine wave |
| `02_play_wav.py` | Play a WAV file |
| `03_looping.py` | Looping |
| `04_filtering.py` | Filtering |
| `05_flanging.py` | Flanging effect |
| `06_autowah.py` | Auto-wah effect |
| `07_soft_clipping.py` | Soft clipping |
| `08_write_to_file.py` | Write to file |
| `09_super_saw.py` | SuperSaw oscillator |
| `10_compression.py` | Compression/limiting/gating |
| `11_dynamics.py` | Advanced dynamics (sidechain) |
| `12_audio_library.py` | Load audio files from remote Strudel maps |
| `13_random.py` | Musical randomness (RandomPE) |
| `14_trigger.py` | TriggerPE (one-shot and gated modes) |
| `15_reverse_pitch_echo.py` | Reverse pitch echo effect |
| `16_comb_filter.py` | Comb filter resonance |
| `17_ladder_filter.py` | Moog-style ladder filter |
| `18_adsr.py` | ADSR + ResetPE |
| `19_sequence.py` | SequencePE |
| `20_alternative_temperaments.py` | Alternative tuning systems (12-ET, just intonation, Pythagorean chords) |
| `21_sequence_with_durations.py` | SequencePE with explicit durations |
| `20_timewarp.py` | TimeWarpPE variable-speed playback |
| `21_analog_osc.py` | AnalogOscPE (PWM, saw/triangle morph, subtractive patch) |
| `22_function_gen.py` | FunctionGenPE (naive) + A/B vs AnalogOscPE at high pitch |
| `23_convolution.py` | ConvolvePE convolution demo using room impulse responses (requires `short_ir*.wav` / `long_ir*.wav`) |
| `24_slice.py` | SlicePE snippet audition framework (edit start/duration points) |

## Modulation and Automation

Many PE parameters accept either constant values or other PEs for modulation:

```python
# Constant frequency
sine_stream = SinePE(frequency=440.0)

# Vibrato (frequency modulated by LFO)
lfo_stream = SinePE(frequency=5.0, amplitude=10.0)
vibrato_stream = SinePE(frequency=MixPE(ConstantPE(440.0), lfo_stream))

# Tremolo (amplitude modulated)
tremolo_lfo_stream = GainPE(SinePE(frequency=4.0), gain=0.3)
tremolo_stream = GainPE(sine_stream, gain=MixPE(ConstantPE(0.7), tremolo_lfo_stream))
```

## Error Handling

Configure error behavior for debugging vs production:

```python
from pygmu2 import ErrorMode, set_error_mode

# Strict mode (default): errors raise exceptions
set_error_mode(ErrorMode.STRICT)

# Lenient mode: non-fatal errors become warnings
set_error_mode(ErrorMode.LENIENT)
```

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
