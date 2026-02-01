# Meltysynth MIDI demo: identifying input-to-output lag

The meltysynth_midi_demo can feel less responsive than toy_midi_sampler. Below are **likely sources of lag** and **ways to identify** them.

---

## 1. Block alignment (MIDI → callback)

**What:** The key is pressed at time T. The MIDI driver puts a message in the queue. We only drain the queue when we **pull** MidiInPE, which happens once per render block. If the key press occurs just after we’ve finished draining for this block, we don’t see the message until the **next** block.

**Worst case:** One block of delay before the callback runs.  
At 512 samples/block @ 44.1 kHz → **~11.6 ms**.

**How to check:** Log `time.perf_counter()` in the callback and compare to the previous `render()` start. If the gap is often ~block_duration, you’re seeing this.  
**Mitigation:** Use a smaller block size (e.g. 256) so “next block” is sooner. Trade-off: more CPU per second and possibly more glitches if render time spikes.

---

## 2. Pull order in MixPE

**What:** MixPE pulls inputs in list order: `[midi_2ch, synth_pe]`. So we pull MIDI first (callback runs, note_on sent), then we pull the synth. That’s correct: the note is in the **same** block we’re rendering. If the order were reversed, we’d have one extra block of delay.

**Status:** Order is already correct; no change needed.

---

## 3. Audio output buffer (blocking stream)

**What:** In the demo we use `AudioRenderer` with a blocking render loop: `render(start, 512)` → `_output(snippet)` → `OutputStream.write(snippet.data)`. The **blocking** stream in `_output()` is created **without** `blocksize` or `latency`. So sounddevice/PortAudio uses its default buffer size, which is often **1024–4096+ samples** (e.g. 23–93 ms at 44.1 kHz). That adds directly to input-to-speaker latency.

**Why toy_midi_sampler can feel better:** Same render loop and same stream creation – so if the toy feels snappier, the main difference is likely **sound design** (see below), not this. But fixing this still reduces latency for both.

**How to check:** Create the blocking stream with explicit `blocksize=512` and `latency='low'`, then compare perceived lag. If it drops, output buffer was a big contributor.

**Mitigation:** In `AudioRenderer._output()`, when creating `_blocking_stream`, pass `blocksize=self._blocksize` and `latency=self._latency` so the blocking loop uses the same low-latency settings as the streaming API.

---

## 4. Meltysynth voice attack (sound design)

**What:** On note_on, the synth starts a voice with an **envelope**. Many presets (e.g. piano) have a **nonzero attack** (a few ms to tens of ms) before level is audible. That’s not “system” latency but **perceived** latency: the key is processed quickly, but the sound ramps up.

**Why toy_midi_sampler can feel more responsive:** The toy plays **pre-recorded slices** (e.g. choir). The slice may have **instant onset** (or a short attack baked in). So you hear sound almost as soon as the trigger fires. With a piano SoundFont, the attack is in the preset and adds to the “key to sound” time.

**How to check:** Switch to a preset with very short attack (e.g. some plucks, organ) and compare feel. If lag is much lower, attack was a big part.  
**Mitigation:** Choose or design presets with short attack for “playable” use; leave longer attack for pads/strings.

---

## 5. Render time and buffer buildup

**What:** If **render** (graph + meltysynth) sometimes takes **longer** than one block period (e.g. >11.6 ms for 512 samples), the loop can’t keep up. Then either (a) we block on `write()` and fall behind, or (b) the output buffer grows and we’re always playing “old” audio – so latency grows.

**How to check:** In the render loop, measure `render()` wall time per block. Log when it exceeds `block_duration` (e.g. 11.6 ms). If you see frequent overruns, CPU is a bottleneck.  
**Mitigation:** Reduce block size (fewer samples per meltysynth call), reduce polyphony, or use a lighter soundfont/preset.

---

## Summary: what to try first

| Priority | What to do | What it tells you |
|----------|------------|--------------------|
| 1 | Pass `blocksize` and `latency` into the **blocking** OutputStream in `AudioRenderer._output()` | Reduces output buffer; if lag drops a lot, DAC buffer was a major source. |
| 2 | Log `render()` wall time per block; alert when > block_duration | Confirms whether we’re missing realtime and building up delay. |
| 3 | Try smaller block size (e.g. 256) in the demo | Reduces worst-case “one block” delay for MIDI; may increase CPU load. |
| 4 | Try a short-attack preset (e.g. organ, pluck) | Separates “system” latency from “envelope” latency. |

The **blocking stream** in `AudioRenderer` is the one clear difference from the streaming path (which already uses `blocksize` and `latency`). Fixing that gives both demos lower latency; the rest is tuning (block size, preset, and CPU headroom).
