# Optimizing pygmu2 Code

This document describes how to profile and optimize processing elements (PEs) and related code, using the PiecewisePE SIGMOID/CONSTANT_POWER optimization as a worked example.

## Workflow

1. **Identify** — Identify suspected hot paths (e.g. PEs called every block, loops over samples).
2. **Measure** — Profile before changing code so you have a baseline and can confirm improvements.
3. **Optimize** — Apply targeted changes (vectorization, fewer allocations, better algorithms).
4. **Re-measure** — Profile again to verify speedup and that behavior is unchanged (tests).

## Profiling Tools

### 1. Renderer built-in profiling

The `Renderer` (and `NullRenderer`) can record per-PE render time when profiling is enabled:

```python
from pygmu2 import NullRenderer, CropPE, Extent

renderer = NullRenderer(sample_rate=44100)
renderer.enable_profiling()
renderer.set_source(your_pe)
renderer.start()
# ... render blocks via renderer.render(start, duration) ...
renderer.print_profile_report()   # or report = renderer.get_profile_report()
```

The report shows total time, call count, and average time per PE class, sorted by total time. Use this to see which PEs dominate CPU.

See `benchmarks/profile_biquad_vs_svfilter.py` for a full example comparing two filter graphs.

### 2. `time.perf_counter_ns()` for narrow sections

For timing a single function or block:

```python
import time
t0 = time.perf_counter_ns()
# ... code ...
elapsed_ns = time.perf_counter_ns() - t0
```

Prefer nanoseconds when correlating with the renderer’s per-call timings.

### 3. `cProfile` for call-tree hotspots

To see which functions consume time across the whole process:

```bash
python -m cProfile -s cumulative your_script.py
```

Use this to find unexpected hotspots (e.g. in NumPy, or in a helper called from `_render`).

## Worked Example: PiecewisePE SIGMOID / CONSTANT_POWER

### Problem

`PiecewisePE._render()` was implemented as a **per-sample loop**: for each output sample `i`, it found the segment, computed a normalized parameter `t`, and called `_segment_curve(np.array([t]), v0, v1, mode)`. For SIGMOID this meant one `np.exp(-np.clip(...))` per sample; for CONSTANT_POWER, one `np.sin` or `np.cos` per sample. That caused:

- Many small NumPy operations instead of one vectorized call per segment.
- Python loop overhead for every sample in the block.

So SIGMOID and CONSTANT_POWER ramps were unnecessarily expensive.

### Approach: Vectorize by segment

The curve is **piecewise**: each segment `[seg_start, seg_end)` has the same formula and endpoints. So we can:

1. Loop over **segments** (not samples).
2. For each segment, find the **range of output indices** that fall in that segment.
3. Compute **all** `t` values for that range in one array.
4. Call `_segment_curve(t, v0, v1, mode)` **once** for the whole segment.
5. Write the result into `data[indices, :]` in one go.

That replaces `duration` calls to `_segment_curve` with one call per segment that overlaps the requested range. For a single long segment, that’s one vectorized call instead of thousands of scalar calls.

### Implementation sketch

**Before (per-sample):**

```python
for i in range(duration):
    s = start + i
    # ... find segment j, seg_start, seg_end, v0, v1 ...
    t = (s - seg_start) / (seg_end - seg_start)
    t_arr = np.array([t], dtype=np.float64)
    val = _segment_curve(t_arr, v0, v1, self._transition_type)[0]
    data[i, :] = val
```

**After (per-segment):**

```python
for j in range(self._n - 1):
    seg_start = int(self._times[j])
    seg_end = int(self._times[j + 1])
    if seg_end <= seg_start:
        continue
    overlap_start = max(seg_start, start)
    overlap_end = min(seg_end, start + duration)
    if overlap_end <= overlap_start:
        continue
    i_lo = overlap_start - start
    i_hi = overlap_end - start
    i_indices = np.arange(i_lo, i_hi, dtype=np.intp)
    s = start + i_indices
    t = (s.astype(np.float64) - seg_start) / (seg_end - seg_start)
    vals = _segment_curve(t, v0, v1, self._transition_type)
    data[i_indices, :] = vals[:, np.newaxis]
```

`_segment_curve` already used vectorized NumPy (`np.exp`, `np.sin`, `np.clip`, etc.), so it accepts a 1D `t` array and returns a 1D array. No change to that function was required; only the caller was changed to pass arrays.

### What to verify

- **Correctness** — Existing unit tests (e.g. `tests/test_piecewise_pe.py`) should pass; they check exact values at specific indices.
- **Performance** — Re-run the same workload with `enable_profiling()` and compare `PiecewisePE` total/average time before and after. For blocks that hit one or a few segments, you should see a clear drop in time for SIGMOID and CONSTANT_POWER.

## General Principles

1. **Prefer vectorized NumPy over Python loops** — One call `f(arr)` is much cheaper than a loop that calls `f(arr[i])` per element. Design `_render` so that inner helpers accept arrays (start/duration or index arrays) and return arrays.
2. **Batch by natural boundaries** — When the math is the same over a contiguous range (e.g. one segment, one block), compute that range in one go instead of one sample at a time.
3. **Avoid per-sample allocations** — Creating `np.array([t])` inside a loop allocates every time. Build one array of all `t` values for the segment (or block) and pass that.
4. **Profile before and after** — Use the renderer’s profiling or `cProfile` to confirm the hot path and to validate that an optimization actually improves total time.
5. **Keep tests** — Optimizations can change edge behavior (e.g. integer bounds). Run the full test suite and any relevant benchmarks after each change.

## Where to look next

- **Other PEs** — If a PE’s `_render` loops over samples and calls NumPy per element, consider segment- or block-wise vectorization in the same spirit as PiecewisePE.
- **Benchmarks** — Add or extend benchmarks (e.g. in `benchmarks/`) for PEs you care about, and run them before/after optimizations.
- **Diagnostics** — The `diagnostics` module (pull counts, timing) can help see how often a PE is pulled and how long each pull takes in a real graph.
