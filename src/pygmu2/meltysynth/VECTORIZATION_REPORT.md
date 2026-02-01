# Vectorizing meltysynth inner loops with NumPy

**Scope:** Investigate using NumPy (and related) data structures to vectorize hot inner loops in the meltysynth package. Report only; no code changes.

---

## 1. Current state

### 1.1 Dependencies

- **pygmu2** already depends on **numpy>=1.20.0** (and scipy, numba) in `pyproject.toml`. Meltysynth is a subpackage; no new dependency would be added by using NumPy inside meltysynth.

### 1.2 Data structures

- **Buffers:** `create_buffer(length)` returns `array.array("d", ...)` (float64). Used for:
  - Per-voice block (`voice._block`)
  - Synthesizer mix buffers (`_block_left`, `_block_right`)
  - Sequencer output buffers
  - Example scripts (left/right for render)
- **Wave data:** SoundFont samples are loaded as `array.array("f", ...)` via `read_int16_array_as_float_array`.
- **API:** `Synthesizer.render(left, right)` and `MidiFileSequencer.render(left, right)` take `MutableSequence[float]` (e.g. `array.array` or list). Callers currently use `create_buffer()` → `array.array`.

### 1.3 Block sizes

- Default block size: **64** (`SynthesizerSettings._DEFAULT_BLOCK_SIZE`).
- Allowed range: 8–1024.
- So inner loops are over **8–1024** samples per call, with 64 as the common case.

---

## 2. Hot inner loops (candidates for vectorization)

### 2.1 ArrayMath (math_utils.py) — **high value, straightforward**

- **multiply_add:** `for i in range(len(destination)): destination[i] += a * x[i]`
- **multiply_add_slope:** same loop with `a += step` each sample (linear gain ramp).

**Called from:** `Synthesizer._write_block()` for **every active voice** per block (e.g. 1–64 voices × 2 channels). So this loop runs 2× (left/right) × active_voice_count per block.

**Vectorization:** With NumPy arrays, `multiply_add` → `destination += a * x`; `multiply_add_slope` → `destination += (a + step * np.arange(n)) * x` (or in-place ramp). Both are simple and numerically equivalent.

**Caveat:** Block length is small (64 typical). NumPy dispatch and possible type handling have fixed cost; for very short lengths the Python loop can be competitive. Needs benchmarking (e.g. block_size 64 vs 256 vs 1024).

### 2.2 Synthesizer render copy loop (synthesizer.py) — **easy win, no NumPy required**

- **Current:** `for t in range(rem): left[offset + wrote + t] = self._block_left[self._block_read + t]` (and same for right).

**Vectorization:** Replace with slice assignment:  
`left[offset + wrote : offset + wrote + rem] = self._block_left[self._block_read : self._block_read + rem]`  
`array.array` supports slice assignment from another array, so this works with current types and avoids a Python loop. No NumPy needed.

### 2.3 _render_block zeroing (synthesizer.py) — **minor**

- **Current:** `for t in range(self._block_size): self._block_left[t] = 0` (and right).

**Vectorization:** With `array.array`, could do `self._block_left[:] = array("d", [0]*self._block_size)` (allocates) or keep the loop. With NumPy, `self._block_left.fill(0)`. Only 64–1024 iterations; impact is small.

### 2.4 BiQuadFilter.process (filter_.py) — **medium value, more invasive**

- **Current:** IIR recurrence, sample-by-sample:  
  `output = a0*input + a1*x1 + a2*x2 - a3*y1 - a4*y2`, then state update.

**Vectorization:** Use `scipy.signal.lfilter(b, a, block)` for the transfer function, then set filter state from the end of the block so the next call continues correctly. Requires:
- Converting block to NumPy (or accepting ndarray).
- Managing state (zi) across calls via `lfilter_zi` or manual state carry.

Fully compatible with current coefficient setup; mainly an implementation swap and state handling.

### 2.5 Oscillator.fill_block_* (oscillator.py) — **medium value, more invasive**

- **Current:** For each sample: advance `position`, compute integer indices and frac, linear interpolation `block[t] = x1 + a*(x2-x1)`.

**Vectorization:** Position advances linearly: `position += pitch_ratio` per sample. So for a block we can compute:
- `positions = start_position + np.arange(len(block)) * pitch_ratio`
- Integer indices and fractional parts for interpolation.
- Vectorized read: e.g. `block[:] = (1 - frac) * data[idx_lo] + frac * data[idx_hi]` (with wrap for loop mode).

Requires:
- Handling loop/no-loop and wrap (e.g. modulo or clamp of indices).
- Wave data (`self._data`) as NumPy array for indexing; currently `array.array("f", ...)` from SoundFont load. So either convert wave data to NumPy at load time or keep a branch for array indexing.

### 2.6 Other loops

- **Voice/voice_collection:** Loops over active voices (small N, object work); not per-sample. Vectorization not applicable.
- **Model/parsing:** One-time file/stream parsing; not hot. No need to vectorize.
- **binary_reader.read_int16_array_as_float_array:** Already uses a generator; AUDIT notes a “larger win” would need something like NumPy (e.g. `np.frombuffer(..., dtype=np.int16) / 32768.0`). That’s a load-time path; optional and independent of real-time synth loops.

---

## 3. Design choices if we vectorize

### 3.1 Buffer type

- **Option A — Keep `array.array` everywhere:** Only vectorize where we can do so without changing types (e.g. slice assignment for the render copy loop). ArrayMath and filter/oscillator stay as Python loops.
- **Option B — Internal NumPy:** `create_buffer()` returns `np.zeros(length, dtype=np.float64)`. Voice blocks, synth mix buffers, and sequencer buffers become NumPy. ArrayMath, filter, oscillator can then use vectorized NumPy. **API:** `render(left, right)` still takes `MutableSequence[float]`. If the user passes `array.array`, we either:
  - Copy in/out to NumPy (extra cost for the common case where they use `create_buffer()`), or
  - Branch in the hot path: `if isinstance(destination, np.ndarray): ... else: current loop`.
- **Option C — NumPy at API boundary:** Require `left`/`right` to be NumPy arrays. Breaking change for any caller using `array.array` or lists.

**Recommendation:** Option B with a **branch in the hot path**: when `destination`/`source` (and in render, `left`/`right`) are NumPy arrays, use vectorized ops; otherwise keep the current loop. That preserves the existing API and allows gradual adoption of NumPy buffers (e.g. a `create_buffer_numpy()` helper) without forcing it.

### 3.2 Wave data (SoundFont samples)

- Currently stored as `array.array("f", ...)`. For vectorized oscillator we need indexed access; NumPy indexing is a good fit. So either:
  - Convert to `np.ndarray` when loading the SoundFont (one-time cost, one place: sample_data or soundfont), or
  - In the oscillator, branch: if `isinstance(self._data, np.ndarray)` use vectorized path, else use current per-sample loop.

### 3.3 Block size and overhead

- For block_size 8–64, Python loop overhead can be comparable to NumPy’s. For 256–1024, NumPy usually wins. A practical approach is to **benchmark** render time for:
  - Default block_size=64 and block_size=1024,
  - Current implementation vs internal NumPy buffers + vectorized ArrayMath (and optionally filter/oscillator).
- Optionally, only enable the NumPy path when `len(destination) > 128` or when type is `ndarray`, to avoid regressions on tiny blocks.

---

## 4. Summary table

| Location              | Loop / op              | Vectorize?        | NumPy required? | Effort | Note                          |
|-----------------------|------------------------|-------------------|-----------------|--------|-------------------------------|
| ArrayMath             | multiply_add, slope    | Yes               | Yes (or branch) | Low    | Benchmark for small blocks    |
| Synthesizer.render   | copy block → left/right| Slice assignment  | No              | Low    | Works with array.array        |
| _render_block         | zero buffers           | Slice/fill        | Optional        | Low    | Small gain                    |
| BiQuadFilter.process | IIR per sample         | scipy.signal.lfilter | Yes (block)   | Medium | State carry between calls     |
| Oscillator.fill_block| position + interpolate | Yes               | Yes (data + block) | Medium | Linear advance, wrap handling |
| binary_reader         | int16→float            | np.frombuffer     | Yes             | Low    | Load-time only                |

---

## 5. Recommendation

1. **Do first (low risk, no dependency change):**  
   Replace the per-sample copy in `Synthesizer.render()` with slice assignment. Works with current `array.array` buffers and removes a Python loop in the output path.

2. **Then (if benchmarks justify):**  
   Introduce an optional NumPy path: when buffers are `np.ndarray`, use vectorized `ArrayMath` (and optionally zero with `.fill(0)`). Keep the existing loop for `array.array` so API and current callers stay unchanged. Benchmark with block_size 64 and 1024 before/after.

3. **Larger refactors (only if profiling shows they matter):**  
   Vectorize BiQuadFilter (e.g. scipy.signal.lfilter + state) and Oscillator (linear position advance + vectorized interpolation), with NumPy blocks and, for the oscillator, NumPy or branch-on-type wave data. These need careful state and loop-boundary handling.

4. **Do not change:**  
   Require NumPy at the public API (`render(left, right)`) without a fallback; that would be a breaking change. Prefer “branch on type” so both `array.array` and `np.ndarray` are supported.

5. **Benchmark:**  
   Before changing hot paths, measure render time (e.g. example_render_wav “simple_chord” or “flourish”) for current code vs each vectorization step, at default and large block sizes, to confirm benefit.

---

## 6. AUDIT reference

AUDIT.md §3.2 says that for `read_int16_array_as_float_array`, “for a larger win you’d need something like numpy, which may be out of scope.” That refers to that specific helper (load-time), not to the real-time synth path. Using NumPy inside meltysynth for vectorization does not add a new project dependency and is consistent with the rest of pygmu2.
