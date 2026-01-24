# pygmu2

A Python framework for generating and processing digital audio, with a bias towards generating music.

## Overview

pygmu2 provides a flexible, composable architecture for building audio processing pipelines. Audio is generated on-demand through a directed acyclic graph (DAG) of Processing Elements (PEs), enabling efficient processing of long or infinite streams.

**Key Features:**
- Lazy evaluation: audio generated on-demand
- Composable design: PEs connect to form complex audio graphs
- Rich library of oscillators, filters, effects, and dynamics processors
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
sine = SinePE(frequency=440.0, amplitude=0.5)

# Play through speakers
with AudioRenderer(sample_rate=44100) as renderer:
    renderer.set_source(sine)
    renderer.start()
    renderer.play_range(0, 44100 * 3)  # Play 3 seconds
```

## Core Concepts

### Processing Elements (PEs)

A **Processing Element** is the fundamental building block. Each PE generates or transforms audio:

```python
# Source PE: generates audio
sine = SinePE(frequency=440.0)

# Transform PE: processes audio from another PE
quieter = GainPE(sine, gain=0.5)

# Combine PEs: mix multiple sources
mix = MixPE(sine1, sine2, sine3)
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
crop = CropPE(source, Extent(0, 44100))  # First second only

# Infinite extent (e.g., oscillators)
sine = SinePE(frequency=440.0)  # Extent(None, None) - plays forever
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
| `SuperSawPE(frequency, voices, detune_cents)` | Detuned unison sawtooth |
| `WavetablePE(wavetable, indexer)` | Wavetable oscillator |

### Sources
| PE | Description |
|----|-------------|
| `ConstantPE(value, channels)` | Constant value |
| `RampPE(start, end, duration)` | Linear ramp |
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
| `LoopPE(source)` | Loop a finite source |
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
| `12_strudel_sample_map.py` | Load samples from remote Strudel maps |
| `13_ladder_filter.py` | Moog-style ladder filter |
| `14_comb_filter.py` | Comb filter resonance |
| `15_reverse_pitch_echo.py` | Reverse pitch echo effect |

## Modulation and Automation

Many PE parameters accept either constant values or other PEs for modulation:

```python
# Constant frequency
sine = SinePE(frequency=440.0)

# Vibrato (frequency modulated by LFO)
lfo = SinePE(frequency=5.0, amplitude=10.0)
vibrato = SinePE(frequency=MixPE(ConstantPE(440.0), lfo))

# Tremolo (amplitude modulated)
tremolo_lfo = GainPE(SinePE(frequency=4.0), gain=0.3)
tremolo = GainPE(sine, gain=MixPE(ConstantPE(0.7), tremolo_lfo))
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

If you see `ssl.SSLCertVerificationError` when using `SampleMap.from_url()`, this is a common issue with Python installed from python.org on macOS. Fix it by running:

```bash
# Option 1: Run the certificate installer (in Finder)
# Applications → Python 3.x → "Install Certificates.command"

# Option 2: Install certifi
uv add certifi   # or: pip install certifi
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, architecture details, and how to create new Processing Elements.

## License

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License - see [LICENSE](LICENSE) for details.
