# Contributing to pygmu2

This guide covers the architecture, conventions, and processes for developing pygmu2.

## Development Setup

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/yourname/pygmu2.git
cd pygmu2

# Install dependencies (creates .venv automatically)
uv sync

# Run tests to verify setup
uv run pytest
```

### Using pipenv

```bash
# Install pipenv
pip install pipenv

# Clone and setup
git clone https://github.com/yourname/pygmu2.git
cd pygmu2

# Install dependencies
pipenv install --dev

# Activate environment
pipenv shell

# Run tests to verify setup
pytest
```

## Project Structure

```
pygmu2/
├── src/pygmu2/
│   ├── __init__.py              # Package exports (update when adding PEs)
│   ├── processing_element.py    # PE and SourcePE base classes
│   ├── snippet.py               # Audio data container
│   ├── extent.py                # Temporal bounds
│   ├── renderer.py              # Renderer base + profiling
│   ├── audio_renderer.py        # Real-time playback
│   ├── null_renderer.py         # Silent rendering (testing)
│   ├── config.py                # Error handling configuration
│   ├── conversions.py           # dB/pitch/time conversions
│   ├── logger.py                # Logging utilities
│   │
│   │   # Processing Elements
│   ├── sine_pe.py               # Sine oscillator
│   ├── blit_saw_pe.py           # Band-limited sawtooth
│   ├── super_saw_pe.py          # Detuned unison saw
│   ├── adsr_pe.py               # ADSR envelope generator
│   ├── constant_pe.py           # Constant value
│   ├── ramp_pe.py               # Linear ramp
│   ├── gain_pe.py               # Gain control
│   ├── mix_pe.py                # Audio mixer
│   ├── delay_pe.py              # Sample delay
│   ├── crop_pe.py               # Temporal cropping
│   ├── loop_pe.py               # Looping
│   ├── slice_pe.py              # Slice/cut region + optional fades
│   ├── biquad_pe.py             # Biquad filter
│   ├── envelope_pe.py           # Envelope follower
│   ├── dynamics_pe.py           # Flexible dynamics processor
│   ├── compressor_pe.py         # Compressor/limiter/gate
│   ├── random_pe.py             # Random value generator
│   ├── window_pe.py             # Windowed statistics
│   ├── wav_reader_pe.py         # WAV file input
│   └── wav_writer_pe.py         # WAV file output
│
├── tests/                       # Unit tests (pytest)
├── examples/                    # Runnable example scripts
├── benchmarks/                  # Performance benchmarks
│   └── benchmark_pes.py         # Auto-discovering PE benchmark suite
├── pyproject.toml               # Project config (uv, pytest, tools)
├── uv.lock                      # uv lockfile (auto-generated)
├── Pipfile                      # pipenv dependencies (alternative)
├── Pipfile.lock                 # pipenv lockfile
└── LICENSE
```

## Architecture

### Processing Element Lifecycle

```
0. set_sample_rate()  Set the global sample rate before any PE construction
1. Construction     PE created with parameters
2. set_source()     Renderer validates graph (purity, channels)
3. on_start()       Called bottom-up (inputs before outputs)
4. render()         Called repeatedly to generate audio
5. on_stop()        Called top-down (outputs before inputs)
```

### Graph Evaluation

Audio flows lazily through the graph:

```
render(start, duration) called on root PE
    └─> Root PE calls render() on its inputs
            └─> Inputs call render() on their inputs
                    └─> ... recursively to source PEs
```

Each `render()` call returns a `Snippet` containing exactly `duration` samples.

## Immutability Contract

Processing elements must treat input `Snippet` buffers as immutable.
Do not modify `snippet.data` from any input PE in-place. Always write into
a new buffer (or a copy) when producing output. This prevents accidental
buffer aliasing across the graph and keeps PE behavior consistent.

## Extent Stability

An element's extent is fixed at construction time and does not change afterward.
The extent may be finite or indefinite (infinite), but it should never vary over
the lifetime of the instance.

Because graphs are built from leaves toward the root, input extents are known
when composing higher-level PEs. As a result, the root extent is determined
at construction time and remains stable for the life of the graph.

## PE Construction Guidelines

Prefer to compute as much as possible in `__init__`. This keeps PEs predictable
and easier to reason about now that sample rate is globally available.

**Do in `__init__` whenever possible:**
- validate parameters and invariants
- build internal graphs
- resolve constants
- infer and set `sample_rate` if it can be computed immediately
- compute extents and purity if independent of `sample_rate`

**Do at construction time whenever possible:**
- values that depend on sample rate (now globally available)
- seconds → samples conversions
- extents that depend on sample rate or other known inputs
- allocating buffers/state tied to the configured rate

### Purity and State

**Pure PEs** produce the same output for the same input and can be shared:
- Most source PEs (SinePE, ConstantPE)
- Stateless transforms (GainPE, MixPE, CropPE)

**Stateful PEs** maintain internal state and cannot be shared:
- EnvelopePE (tracks envelope value)
- BiquadPE (filter state)
- BlitSawPE (phase accumulator)

The Renderer validates that stateful PEs have only one downstream consumer.

## Creating a New Processing Element

### Template

```python
"""
MyNewPE - Description of what it does.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from typing import Optional, Union
import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class MyNewPE(ProcessingElement):
    """
    One-line summary.
    
    Detailed description of the PE's behavior, including:
    - What it does
    - Key parameters and their effects
    - Any important notes about state or purity
    
    Args:
        source: Input audio PE
        param1: Description (default: X)
        param2: Description, can be float or PE for modulation
    
    Example:
        result_stream = MyNewPE(source_stream, param1=0.5)
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        param1: float = 1.0,
        param2: Union[float, ProcessingElement] = 0.0,
    ):
        self._source = source
        self._param1 = param1
        self._param2 = param2
        
        # State (if any)
        self._state: Optional[float] = None
    
    def inputs(self) -> list[ProcessingElement]:
        """Return list of input PEs."""
        result = [self._source]
        if isinstance(self._param2, ProcessingElement):
            result.append(self._param2)
        return result
    
    def is_pure(self) -> bool:
        """Return True if stateless, False if has internal state."""
        return self._state is None  # Or just: return False
    
    def channel_count(self) -> Optional[int]:
        """Return output channel count, or None to inherit from input."""
        return self._source.channel_count()
    
    def _compute_extent(self) -> Extent:
        """Compute and return this PE's temporal bounds."""
        return self._source.extent()
    
    def on_start(self) -> None:
        """Initialize state when rendering begins."""
        self._state = 0.0
    
    def on_stop(self) -> None:
        """Clean up when rendering ends."""
        self._state = None
    
    def _render(self, start: int, duration: int) -> Snippet:
        """Generate output samples."""
        # Get input
        source_snippet = self._source.render(start, duration)
        audio = source_snippet.data.astype(np.float64)
        
        # Get parameter (constant or from PE)
        if isinstance(self._param2, ProcessingElement):
            param2_snippet = self._param2.render(start, duration)
            param2 = param2_snippet.data[:, 0]
        else:
            param2 = self._param2
        
        # Process
        output = audio * self._param1 + param2
        
        return Snippet(start, output.astype(np.float32))
    
    def __repr__(self) -> str:
        return f"MyNewPE(param1={self._param1})"
```

### Checklist for New PEs

1. **Create the PE file** in `src/pygmu2/`
2. **Add to `__init__.py`**:
   - Import the class
   - Add to `__all__`
3. **Write unit tests** in `tests/test_myname_pe.py`
4. **Add benchmark config** (optional) in `benchmarks/benchmark_pes.py`
5. **Run tests**: `pipenv run pytest tests/test_myname_pe.py -v`
6. **Run full suite**: `pipenv run pytest`
7. **Add an example** (optional) in `examples/`

### Conventions

- **Naming**: `<Name>PE` for classes, `<name>_pe.py` for files
- **Parameters**: Use dB for levels, seconds for time, Hz for frequency
- **PE inputs**: Accept `Union[float, ProcessingElement]` for modulatable params
- **Docstrings**: Include Args, Example, and behavioral notes
- **Errors**: Use `config.handle_error()` instead of raising directly
- **Time parameters**:
  - Prefer defaults expressed in **seconds** (no implicit sample-rate assumptions).
  - Convert seconds→samples in `__init__` using `ProcessingElement._time_to_samples(...)`.
  - If you support both `*_samples` and `*_seconds`, accept them as `Optional[...]` and let `_time_to_samples` enforce mutual exclusion.

## Testing

### Running Tests

Using **uv** (recommended):

```bash
# All tests
uv run pytest

# Specific file
uv run pytest tests/test_sine_pe.py -v

# With coverage
uv run pytest --cov=src --cov-report=html

# Skip slow tests (if any)
uv run pytest -m "not slow"
```

Using **pipenv**:

```bash
# All tests
pipenv run pytest

# Specific file
pipenv run pytest tests/test_sine_pe.py -v

# With coverage
pipenv run pytest --cov=src --cov-report=html

# Skip slow tests (if any)
pipenv run pytest -m "not slow"
```

### Test Structure

```python
"""Tests for MyNewPE."""

import numpy as np
import pytest
from pygmu2 import MyNewPE, SinePE, NullRenderer


class TestMyNewPEBasics:
    """Test creation and properties."""
    
    def test_create_default(self):
        pe_stream = MyNewPE(SinePE(frequency=440.0))
        assert pe_stream.param1 == 1.0
    
    def test_inputs(self):
        source_stream = SinePE(frequency=440.0)
        pe_stream = MyNewPE(source_stream)
        assert source_stream in pe_stream.inputs()


class TestMyNewPERender:
    """Test rendering behavior."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_render_returns_snippet(self, renderer):
        pe_stream = MyNewPE(SinePE(frequency=440.0))
        renderer.set_source(pe_stream)
        renderer.start()
        
        snippet = pe_stream.render(0, 1000)
        
        assert snippet.data.shape == (1000, 1)
```

## Profiling and Benchmarks

### Built-in Profiling

The Renderer has optional profiling:

```python
renderer = NullRenderer(sample_rate=44100)
renderer.enable_profiling()
renderer.set_source(my_pe)
renderer.start()

# Render some audio
for i in range(100):
    renderer.render(i * 1024, 1024)

renderer.stop()
renderer.print_profile_report()
```

### Benchmark Suite

Using **uv**:

```bash
# Run all benchmarks
uv run python benchmarks/benchmark_pes.py

# Quick mode (fewer iterations)
uv run python benchmarks/benchmark_pes.py --quick

# List discovered PEs
uv run python benchmarks/benchmark_pes.py --list

# Buffer size scaling test
uv run python benchmarks/benchmark_pes.py --scaling
```

Using **pipenv**:

```bash
# Run all benchmarks
pipenv run python benchmarks/benchmark_pes.py

# Quick mode (fewer iterations)
pipenv run python benchmarks/benchmark_pes.py --quick
```

### Adding Benchmark Configs

In `benchmarks/benchmark_pes.py`, add to `setup_fallback_configs()`:

```python
register_fallback("MyNewPE", [
    BenchmarkConfig(
        "MyNewPE (config1)",
        lambda: MyNewPE(SinePE(frequency=440.0), param1=1.0),
        "category"
    ),
])
```

### Performance Tips

- Use numpy vectorized operations, avoid Python loops
- For IIR filters, use `scipy.signal.lfilter`
- For sliding window operations, use `scipy.ndimage` filters
- Profile before optimizing: `pipenv run python -m cProfile -s cumtime script.py`

## Error Handling

Use `config.handle_error()` instead of raising exceptions directly:

```python
from pygmu2.config import handle_error

# Non-fatal error (warns in lenient mode, raises in strict mode)
if something_wrong:
    if handle_error("Something went wrong"):
        return default_value  # Lenient mode fallback

# Fatal error (always raises)
if critical_error:
    handle_error("Critical failure", fatal=True)

# With custom exception type
handle_error("Invalid value", fatal=True, exception_class=ValueError)
```

## Code Style

### Formatting

```bash
uv run black src tests
# or: pipenv run black src tests
```

### Type Checking

```bash
uv run mypy src
# or: pipenv run mypy src
```

### Linting

```bash
uv run flake8 src tests
# or: pipenv run flake8 src tests
```

### Import Order

```python
# Standard library
from typing import Optional, Union
import numpy as np

# Third-party (if any)
from scipy.signal import lfilter

# Local
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet
```

## Git Workflow

1. Create a branch for your feature
2. Write tests first (TDD encouraged)
3. Implement the feature
4. Run the full test suite
5. Update documentation if needed
6. Create a pull request

### Commit Messages

```
Add MyNewPE for doing something useful

- Describe what it does
- Note any important implementation details
- Reference any related issues
```

## Questions?

Open an issue on GitHub or check existing issues for similar questions.
