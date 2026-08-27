# Contributing to pygmu2

This guide covers the architecture, conventions, and processes for developing
pygmu2. The governing design document is [`DESIGN_PHILOSOPHY.md`](DESIGN_PHILOSOPHY.md);
where this guide states a rule, that document states the reason and the CI gate
that enforces it.

## Development Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/rdpoor/pygmu2.git
cd pygmu2
uv sync --all-extras   # extras cover the numba/midi/audio PEs; CI uses this too

uv run pytest -q       # verify: the whole suite should be green
```

## Project Structure

```
pygmu2/
├── src/pygmu2/
│   ├── __init__.py              # Public surface: eager + lazy exports, __all__
│   ├── processing_element.py    # PE base class: render chokepoint, stateful contract
│   ├── snippet.py               # Audio buffer (float32, read-only) + broadcast_channels
│   ├── extent.py                # Temporal bounds (None = unbounded)
│   ├── semantic_signal.py       # Shared base for GateSignal / TriggerSignal
│   ├── renderer.py              # Renderer base: lifecycle walks, no validation
│   ├── audio_renderer.py        # Real-time playback (sounddevice)
│   ├── null_renderer.py         # Silent rendering (tests, offline)
│   ├── diagnostics.py           # THE profiler: with diagnostics.profile() as report
│   ├── config.py                # Global sample rate
│   ├── conversions.py           # dB/pitch/time conversions
│   ├── temperament.py           # Tuning systems
│   ├── utils.py                 # play / play_offline / browse / render_to_file
│   ├── *_pe.py                  # The PE catalog (one file per PE)
│   └── meltysynth/              # Vendored SoundFont synth — do not reformat/edit
├── tests/
│   ├── pe_factories.py          # One construction recipe per exported PE
│   ├── test_contract.py         # Universal contract suite (auto-discovers __all__)
│   ├── test_examples.py         # Every example must import cleanly
│   ├── test_boundaries.py       # Exports, README tables, import hygiene, benchmarks
│   ├── probes.py                # Test-only PEs (IdentityPE timestamp probe)
│   └── test_*.py                # Per-PE behavioural tests
├── examples/                    # *_eg.py — runnable demos (smoke-tested in CI)
├── benchmarks/                  # benchmark_pes.py (shares tests/pe_factories.py)
├── scripts/                     # jogshuttle player, MIDI demos, gen_readme_tables
└── pyproject.toml               # THE dependency manifest (uv.lock pins it)
```

## Architecture

### Lifecycle

```
0. pg.set_sample_rate(rate)   before constructing any PE (enforced)
1. Construction               parameters validated; internal graphs built
2. renderer.set_source(pe)    attaches the graph — no validation walk
3. renderer.start()           on_start() bottom-up over the graph
4. render(start, duration)    pull-based, lazy, exact-length Snippets
5. renderer.stop()            on_stop() top-down
```

There is deliberately **no up-front graph validation**: correctness is proven
exhaustively by the contract suite in CI, and anything wrong at runtime fails
loudly where the fact surfaces (philosophy PD-2). A channel mismatch raises
numpy's own broadcast error at render time; that is by design, not neglect.

### The `stateful` contract (contiguity)

Every PE declares whether it holds render state:

```python
class MyFilterPE(ProcessingElement):
    stateful = True   # filter memory, phase accumulator, delay line, ...
```

- `stateful = False` (the default): `render(start, duration)` is a pure
  function of its arguments — any order, any position, multiple sinks.
- `stateful = True`: renders must be **contiguous** (each `start` equals the
  previous request's end). The base class enforces this: a gap or seek raises
  `RuntimeError: non-contiguous render`. The sanctioned way to seek is an
  explicit `reset_state()`, which clears the expectation and the PE's state.

When statefulness depends on constructor arguments (e.g. an oscillator that is
closed-form with constant parameters but accumulates phase when a parameter is
a PE), declare it as a property:

```python
@property
def stateful(self) -> bool:
    return bool(self.inputs())
```

The declaration is **falsified by CI**: the contract suite renders every PE in
shuffled order and requires it to either match the contiguous reference
(stateless) or raise (stateful). A lie in either direction fails the build.

### Snippets are immutable

`Snippet` buffers are `float32`, shape `(samples, channels)`, and **read-only**
(`data.flags.writeable = False`). Buffers are shared across sinks; an in-place
write would corrupt siblings silently, so it raises at the write instead.
Always produce output into a new array. To broadcast a mono control across
channels, use `snippet.broadcast_channels(data, channels)` — it returns a view
and raises on a genuine multichannel mismatch.

### Extents

`extent()` is fixed at construction and never changes (the contract suite
checks this). `None` bounds mean unbounded. Note that `render()` does **not**
clamp to the extent — samples outside it are zero-filled by the PE itself, and
`extend_mode` hold regions (CropPE) live *outside* the advertised extent, so
they are only observable when a wider consumer pulls past it.

### Composite PEs (internal graphs)

A PE that builds an internal graph must **expose it through `inputs()`** so
the Renderer's lifecycle walk reaches every internal node:

```python
class MyCompositePE(ProcessingElement):
    def __init__(self, source, ...):
        cached = CachePE(source)          # if source feeds >1 internal sink
        self._out = OtherPE(SomePE(cached), cached)

    def inputs(self):
        return [self._out]                # renderer walks the whole graph

    def _render(self, start, duration):
        return self._out.render(start, duration)

    def channel_count(self):
        return self._out.channel_count()

    def _compute_extent(self):
        return self._out.extent()
```

Do **not** hide internal PEs behind `inputs() == []` and hand-forward
lifecycle calls — that bypasses the contiguity bookkeeping (`reset_state()`
via the private `_reset_state()` does not clear the expectation) and has
caused real bugs twice.

**CachePE and fan-out:** wrap a source in `CachePE` when it feeds multiple
internal sinks. CachePE is order-safe by design (identical repeated requests
are served from cache); if a sink diverges, the request falls through to the
source, whose own contiguity check raises naming the actual state owner. The
composite itself usually holds no state — leave it stateless and let the
internal nodes declare theirs.

### Gates and triggers

`GateSignal` = sustained level, values exactly {0, 1}, duration meaningful.
`TriggerSignal` = isolated one-sample {0, 1} events. Gates are the primitive:
derive triggers with `GateToTriggerPE(gate)`. Note the rate convention when
deriving from a toggling gate: `RandomGatePE(rate=r)` yields r/2 triggers/sec.

## Creating a New Processing Element

The checklist is five steps. Everything else (contiguity, framing, dtype,
zero-fill, extent stability, reset semantics) is enforced by the base class
and verified automatically by the contract suite — if a proposed convention
would add a sixth step here, it belongs in the base class instead
(philosophy R6).

1. **Write `src/pygmu2/<name>_pe.py`** — implement `_render()`, `inputs()`,
   `channel_count()`, `_compute_extent()`; set `stateful = True` if the PE
   holds render state (and implement `_reset_state()` to clear it).
2. **Export it**: add to `__init__.py` (the lazy registry if it imports
   scipy/numba/mido/miniaudio — `import pygmu2` must stay heavy-dep-free)
   and to `__all__`.
3. **Add a factory** to `tests/pe_factories.py` — one canonical, seeded
   construction. This buys full contract-suite coverage and a benchmark
   config for free.
4. **Write behavioural tests** in `tests/test_<name>_pe.py` for the DSP
   itself (the contract suite already covers the framework contract).
5. **Run `uv run pytest -q`** — this includes the export-completeness walk,
   the README-table check (run `uv run python scripts/gen_readme_tables.py`
   if the table drifted), and the examples smoke test.

Conventions: `<Name>PE` / `<name>_pe.py`; dB for levels, seconds for time, Hz
for frequency; modulatable parameters accept `float | ProcessingElement` (use
`self._scalar_or_pe_values(...)` in `_render`); seconds→samples in `__init__`.
Errors are plain raises where the fact surfaces — there is no error-mode
machinery, and no speculative validation (philosophy PD-2/R3). The class
docstring's first line becomes the README table entry — make it count.

## Testing

```bash
uv run pytest -q                       # everything (fast: no coverage)
uv run pytest tests/test_contract.py   # the universal contract suite
uv run pytest --cov=src                # coverage, when you want it
```

Test tiers (philosophy §6): **contract** (auto-discovered, every PE),
**behavioural** (per-PE DSP), **boundary** (examples import, benchmarks run,
README matches, `import pygmu2` loads no heavy deps), **numerical**
(analytical DSP properties — see `tests/test_analytical_pe.py`).

Rules of thumb: go through a `NullRenderer` for anything lifecycle-related;
seed all randomness; `tests/probes.py:IdentityPE` outputs the sample index,
which makes time-manipulating PEs trivially verifiable.

## Profiling

`pygmu2.diagnostics` is the profiler (the render chokepoint has zero-cost
hooks). New PE code is plain numpy until a profile names it hot — numba or
other acceleration needs a measurement attached (philosophy §4):

```python
from pygmu2 import diagnostics

with diagnostics.profile() as report:
    renderer.render(0, 44100)
print(report.summary(sample_rate=44100))   # per-class ms, samples/s, x realtime
```

For examples, `PYGMU_RENDER_MODE=profile uv run python examples/foo_eg.py 1`
renders silently and prints the profile; `=offline` renders to a file first.

## The change protocol

A change to `src/` is not complete until its consumers are proven
(philosophy §7). CI enforces: the full suite (contract + boundary + examples
smoke) on every push, plus `black --check`. Renaming or removing a public
name means migrating `examples/`, `benchmarks/`, `scripts/`, `tests/`, and
`*.md` in the same commit — no aliases, no deprecation shims (pre-1.0), and
**never** parking a broken file as `.py-disabled` (CI rejects new ones).

## Code Style

```bash
uv run black src tests examples benchmarks scripts   # format (CI checks)
uv run flake8 src tests                              # lint
uv run mypy src/pygmu2/processing_element.py ...     # types (core ratchet)
```

Vendored `src/pygmu2/meltysynth/` is excluded from all of the above.

## Commit Messages

```
One-line summary of what changed

Body: what and why — including what the change deletes or falsifies.
If the change claims an invariant, name the test that fails when it is
violated. If it adds a debugging aid or optimization, name the measured
error or profile that motivated it.
```
