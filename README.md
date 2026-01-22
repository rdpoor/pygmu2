# pygmu2

A Python framework for generating and processing digital audio, with a bias towards generating music.

## Overview

pygmu2 provides a flexible, composable architecture for building audio processing pipelines. In its current form, pygmu2 is text-based and "offline" — you create Python scripts that assemble various processing elements into a directed acyclic graph (DAG), then render the results to audio output or files.

**Key Features:**
- Lazy evaluation: audio is generated on-demand, enabling efficient processing of long or infinite streams
- Composable design: processing elements connect together to form complex audio graphs
- Extensible: easily create custom processing elements for new audio operations
- Cross-platform audio playback via `sounddevice`
- WAV file I/O via `soundfile`

## Core Concepts

### Processing Elements (PEs)

A **Processing Element** is the fundamental building block of pygmu2. Each PE represents an audio operation — generating sound, transforming it, or combining multiple sources.

PEs form a **directed acyclic graph (DAG)** where:
- Source PEs (like `SinePE` or `WavReaderPE`) generate audio from scratch
- Transform PEs (like `GainPE` or `DelayPE`) process audio from upstream PEs
- The graph is evaluated lazily via the `render(start, duration)` method

### Snippets

A **Snippet** is a thin wrapper around a NumPy array containing audio samples. It represents a chunk of audio data with:
- A `start` sample index (where this chunk begins in absolute time)
- A `data` array of shape `(samples, channels)` with `float32` dtype
- Derived properties: `duration`, `end`, `channels`

### Extents

An **Extent** defines the temporal bounds of a PE's output:
- `start`: first valid sample index (inclusive), or `None` for "infinite past"
- `end`: last valid sample index (exclusive), or `None` for "infinite future"
- Utility methods: `intersects()`, `intersection()`, `union()`, `contains()`, `spans()`

### Renderers

A **Renderer** pulls audio from a PE graph and sends it somewhere:
- `AudioRenderer`: plays audio through your system's audio device
- `NullRenderer`: discards audio (useful for testing and benchmarking)
- Custom renderers can write to files, streams, or other destinations

## Quick Start

```python
from pygmu2 import SinePE, GainPE, AudioRenderer

# Create a 440 Hz sine wave
sine = SinePE(frequency=440.0, amplitude=0.3)

# Apply gain (optional)
output = GainPE(sine, gain=0.5)

# Play through speakers
renderer = AudioRenderer(sample_rate=44100)
renderer.set_source(output)

with renderer:
    renderer.play_extent(0, 44100 * 5)  # Play 5 seconds
```

## Primary Classes

### ProcessingElement (Abstract Base Class)

The base class for all audio processing units.

```python
class ProcessingElement(ABC):
    def render(self, start: int, duration: int) -> Snippet
        """Generate audio samples for the given range. Always returns a
        Snippet of the requested size, zero-padded outside the PE's extent."""
    
    def extent(self) -> Extent
        """Return the temporal bounds of this PE's output."""
    
    def inputs(self) -> list[ProcessingElement]
        """Return list of input PEs (empty for source PEs)."""
    
    def is_pure(self) -> bool
        """Return True if this PE is stateless and safe for multiple sinks."""
    
    def channel_count(self) -> int | None
        """Return number of output channels, or None if determined by context."""
    
    def configure(self, sample_rate: int) -> None
        """Called by Renderer to inject sample rate into the graph."""
    
    def on_start(self) -> None
        """Lifecycle hook called when rendering begins (bottom-up)."""
    
    def on_stop(self) -> None
        """Lifecycle hook called when rendering ends (top-down)."""
```

### SourcePE (Abstract Base Class)

A specialization of `ProcessingElement` for PEs with no inputs:

```python
class SourcePE(ProcessingElement):
    def inputs(self) -> list[ProcessingElement]:
        return []  # Source PEs have no inputs
    
    def is_pure(self) -> bool:
        return True  # Source PEs are typically stateless
```

### Snippet

A container for audio sample data:

```python
class Snippet:
    def __init__(self, start: int, data: NDArray[np.floating])
    
    @property
    def start(self) -> int          # Starting sample index
    @property
    def end(self) -> int            # Ending sample index (exclusive)
    @property
    def duration(self) -> int       # Number of samples
    @property
    def channels(self) -> int       # Number of audio channels
    @property
    def data(self) -> NDArray       # Shape: (samples, channels), dtype: float32
    
    @classmethod
    def from_zeros(cls, start: int, duration: int, channels: int) -> Snippet
```

### Extent

Defines temporal bounds for audio data:

```python
class Extent:
    def __init__(self, start: int | None, end: int | None)
    
    @property
    def start(self) -> int | None   # None = infinite past
    @property
    def end(self) -> int | None     # None = infinite future
    @property
    def duration(self) -> int | None  # None if unbounded
    
    def contains(self, sample: int) -> bool
    def spans(self, start: int, end: int) -> bool
    def intersects(self, other: Extent) -> bool
    def intersection(self, other: Extent) -> Extent | None
    def union(self, other: Extent) -> Extent
```

### Renderer (Abstract Base Class)

Manages audio output and graph lifecycle:

```python
class Renderer(ABC):
    def __init__(self, sample_rate: int, channel_count: int = 2)
    
    def set_source(self, source: ProcessingElement) -> None
        """Set the root PE and configure/validate the graph."""
    
    def start(self) -> None
        """Begin rendering (calls on_start() bottom-up through graph)."""
    
    def stop(self) -> None
        """End rendering (calls on_stop() top-down through graph)."""
    
    def render(self, start: int, duration: int) -> Snippet
        """Request audio from the source PE."""
    
    # Context manager support
    def __enter__(self) -> Renderer
    def __exit__(self, ...) -> None  # Calls stop()
```

## Available Processing Elements

### Source PEs (Generate Audio)
- `SinePE(frequency, amplitude, phase)` — Sine wave generator (supports PE inputs for modulation)
- `ConstantPE(value, channels)` — Outputs a constant value
- `RampPE(start_value, end_value, duration, channels)` — Linear ramp
- `WavReaderPE(path)` — Reads audio from a WAV file
- `IdentityPE(channels)` — Outputs the sample index
- `DiracPE(channels)` — Unit impulse (1.0 at sample 0, 0.0 elsewhere)

### Transform PEs (Process Audio)
- `GainPE(source, gain)` — Apply gain (float or PE for automation)
- `DelayPE(source, delay)` — Delay by N samples
- `CropPE(source, extent)` — Limit output to a specified extent
- `MixPE(*sources)` — Sum multiple PE outputs together

### Output PEs (Side Effects)
- `WavWriterPE(source, path, sample_rate)` — Write audio to a WAV file

## Project Structure

```
pygmu2/
├── src/pygmu2/
│   ├── __init__.py           # Package exports
│   ├── processing_element.py # PE and SourcePE base classes
│   ├── snippet.py            # Snippet class
│   ├── extent.py             # Extent class
│   ├── renderer.py           # Renderer base class
│   ├── audio_renderer.py     # Real-time audio playback
│   ├── null_renderer.py      # Null output (for testing)
│   ├── config.py             # Error handling configuration
│   ├── logger.py             # Logging utilities
│   ├── sine_pe.py            # Sine wave generator
│   ├── constant_pe.py        # Constant value source
│   ├── ramp_pe.py            # Linear ramp source
│   ├── mix_pe.py             # Audio mixer
│   ├── gain_pe.py            # Gain control
│   ├── delay_pe.py           # Sample delay
│   ├── crop_pe.py            # Temporal cropping
│   ├── identity_pe.py        # Sample index output
│   ├── dirac_pe.py           # Unit impulse
│   ├── wav_reader_pe.py      # WAV file reader
│   └── wav_writer_pe.py      # WAV file writer
├── tests/                    # Comprehensive test suite
├── examples/                 # Example scripts
├── Pipfile                   # pipenv dependencies
├── pyproject.toml            # Project configuration
└── LICENSE                   # MIT License
```

## Installation

### Prerequisites

Install pipenv:
```bash
pip install pipenv
```

### Install Dependencies

```bash
pipenv install --dev
```

### Activate Environment

```bash
pipenv shell
```

Or run commands directly:
```bash
pipenv run python -m pytest
```

## Running Tests

```bash
# Run all tests
pipenv run pytest

# Run with coverage
pipenv run pytest --cov=src --cov-report=html
```

## Development

### Code Formatting
```bash
pipenv run black src tests
```

### Type Checking
```bash
pipenv run mypy src
```

### Linting
```bash
pipenv run flake8 src tests
```

## Error Handling

pygmu2 provides configurable error handling via `ErrorMode`:

```python
from pygmu2 import ErrorMode, set_error_mode

# Strict mode (default): errors raise exceptions
set_error_mode(ErrorMode.STRICT)

# Lenient mode: non-fatal errors become warnings
set_error_mode(ErrorMode.LENIENT)
```

## License

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
