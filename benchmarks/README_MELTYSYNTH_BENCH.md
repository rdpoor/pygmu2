# Meltysynth benchmark workflow

Run all scripts with `uv run ...` (from project root).

To compare **non-vectorized** vs **vectorized** meltysynth render time:

1. **Save the vectorization changes (then revert in working tree)**

   ```bash
   git stash push -m "meltysynth-vectorization" -- \
     src/pygmu2/meltysynth/__init__.py \
     src/pygmu2/meltysynth/math_utils.py \
     src/pygmu2/meltysynth/model/soundfont.py \
     src/pygmu2/meltysynth/synth/oscillator.py \
     src/pygmu2/meltysynth/synth/synthesizer.py \
     src/pygmu2/meltysynth/synth/voice.py
   ```

   After this, the working tree has the **non-vectorized** (last-committed) version of those files.

2. **Run the benchmark (baseline)**

   ```bash
   uv run python benchmarks/benchmark_meltysynth.py
   ```

   Optionally pass a soundfont path and `--runs 5 --warmup 2`. Record the output (e.g. `block_size=64 mean=... ms realtime_ratio=...`).

3. **Reintroduce the vectorization changes**

   ```bash
   git stash pop
   ```

4. **Run the benchmark again (vectorized)**

   ```bash
   uv run python benchmarks/benchmark_meltysynth.py
   ```

5. **Compare** the two runs (same block sizes, same soundfont). `realtime_ratio > 1` means faster than realtime.

---

## Results (example run)

**Before vectorization (block_size=1024):**  
  mean=313.41 ms  realtime_ratio=9.57x

**After vectorization (block_size=1024):**  
  mean=242.47 ms  realtime_ratio=12.37x

(~23% faster; realtime ratio improved from 9.57x to 12.37x.)

---

**Alternative: use a patch file instead of stash**

- Save vectorization as a patch:  
  `git diff HEAD -- src/pygmu2/meltysynth/ > /tmp/meltysynth-vectorization.patch`
- Revert:  
  `git checkout HEAD -- src/pygmu2/meltysynth/`
- Run benchmark (baseline):  
  `uv run python benchmarks/benchmark_meltysynth.py`
- Re-apply:  
  `git apply /tmp/meltysynth-vectorization.patch`
- Run benchmark again (vectorized):  
  `uv run python benchmarks/benchmark_meltysynth.py`

This keeps the benchmark script and any other uncommitted changes intact while only toggling the vectorization diff.
