# pygmu2 Backlog

Items are listed roughly in priority order within each section. Completed
items are removed (git history remembers). The overhaul's own tracking
lives in IMPLEMENTATION_PLAN.md / PE_CONSOLIDATION.md.

## Features

- **TralfamPE**: Add zero padding and automatic gain normalization.
  (CropPE(..., clip=False) now provides the padding primitive.)

- **Wavetable example**: WavetablePE/ArrayPE have no examples. The
  docstring recipes (naive-saw index ramp, RandomValuePE indices) would
  make a good `wavetable_eg.py`.

## Cleanup / Refactoring

- **`spatial_eg.py`**: Review and improve `demo_hrtf_spatialization`;
  finish the two TODO stubs in spatial_pe.py (SpatialConstantPower
  validation/implementation).

- **Examples**: 56 demo functions still open with a `print()` that
  partially duplicates the run_demos banner; prune the ones that carry
  no parameter info.

## Second-pass ledger (philosophy §4 — needs an attached measurement)

- User-assignable `name=` on ProcessingElement
- Shape checks naming both PEs in multi-input PEs
- Cycle guard with path trace (today: RecursionError)
- Graceful error capture in AudioRenderer's callback thread
- PatchCable (reified graph edges)
- Configurable error handling (post-ErrorMode)
- Streaming-to-WAV segment capture (today: in-memory Recording, written on
  completion — a cost only for takes long enough to pressure RAM)
- AudioInPE: live input as a graph source (the duplex substrate exists;
  needs a real-time monitoring latency budget to justify)
- Persistent duplex stream shared by calibrate() and recording (today each
  opens its own sd.Stream; per-open buffer alignment adds ±~20 samples —
  measured end-to-end overdub error 0.29 ms on H4 aggregate, 2026-08-27)

## Tests

- Grow Tier 4 numerical characterisation (test_analytical_pe.py):
  oscillator THD, filter -3 dB points, convolution IR length, dynamics
  gain accuracy.

## Documentation

- docs/temperament-implementation.md is a completed design note kept for
  reference.
