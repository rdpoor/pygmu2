# toy_midi_sampler — Concepts, Architecture, and Pygmu Idioms

High-level documentation for the toy MIDI sampler script: the ideas behind it, how it is structured, and the pygmu patterns it uses.

---

## Concepts

**What it does**  
A simple keyboard sampler: notes 48–72 (middle C to C above) each play the same short spoken word at a pitch that follows the key. Middle C (60) = original pitch; each semitone up multiplies playback rate by 2^(1/12). Note-on starts playback (with retrigger); note-off stops it. Velocity scales amplitude (0–127 → 0.0–1.0 gain).

**Key-per-voice**  
There is one independent “voice” per MIDI note in range: 25 keys → 25 voice chains. No voice stealing; each key has its own slice playback state. Pressing the same key again retriggers from the start.

**Trigger/gate**  
Each voice is gated: it only produces sound while the key is held. The “trigger” is a control signal (1.0 = key down, 0.0 = key up). Pygmu models this with **TriggerPE**: it plays a source from time 0 when the trigger goes high, and stops (and optionally retriggers on the next high) when the trigger goes low.

**Pitch from MIDI note**  
Pitch is implemented as playback rate: `rate = 2^((midi_note - 60) / 12)`. **TimeWarpPE** resamples the slice at that rate, so the same slice sounds higher or lower. The slice is a fixed segment of the WAV; only the rate changes per key.

**Velocity to gain**  
MIDI velocity (0–127) is converted to a gain (0.0–1.0) and applied with **GainPE**. The gain is a “control” value that is constant over a block but can change when the user hits a new note (see KeyTracker below).

---

## Architecture

**Graph shape**  
The audio graph is a single **MixPE** with 26 inputs:

1. **One “silence” branch**: `GainPE(MidiInPE(callback=...), 0.0)`  
   - Contributes no audio (gain 0) but ensures **MidiInPE** is rendered every block so the MIDI queue is drained and the callback runs.

2. **25 voice branches**: one per MIDI note in [48, 73).  
   Each branch is:  
   `TriggerPE(TimeWarpPE(GainPE(SLICE_MONO, gain_stream), rate), trigger_stream)`  
   - **SLICE_MONO**: one shared mono slice (same word for all keys).  
   - **gain_stream** / **trigger_stream**: per-key control signals provided by **KeyTracker** (see below).

**KeyTracker**  
A small object per MIDI note that:

- Holds two **ConstantPE**s: one for trigger (0.0 or 1.0), one for gain (0.0 or velocity/127).
- Exposes `note_on(velocity)` and `note_off()` that **mutate** those ConstantPEs’ `_value` (see Idioms).
- Exposes `get_pitched_slice()` which builds and returns the voice chain for that key (GainPE → TimeWarpPE → TriggerPE) using the shared **SLICE_MONO** and that key’s rate and control streams.

**Data flow**

- **MIDI thread** (mido): receives note_on/note_off, pushes messages into a queue.
- **Render thread** (main): each block, `renderer.render(start, duration)` runs. That pulls the root **MixPE**, which pulls all 26 inputs. Pulling the silence branch pulls **MidiInPE**, which drains the queue and calls the user **callback** for each message.
- **Callback**: for each message, finds the **KeyTracker** for that note and calls `note_on(velocity)` or `note_off()`, updating the trigger and gain ConstantPEs.
- The same block then pulls the 25 voice branches; each **TriggerPE** reads its trigger ConstantPE (now 0 or 1), and the **GainPE** reads its gain ConstantPE. So “this block” reflects the state set by the callback earlier in the same block.

**Lifecycle**  
The script calls `renderer.start()` before the loop and `renderer.stop()` after. That causes the renderer to call **on_start** / **on_stop** on all PEs in the graph (e.g. MidiInPE opens/closes the MIDI port; TriggerPE and TimeWarpPE reset internal state). The render loop only calls `renderer.render(sample_index, BLOCK_SIZE)` and advances `sample_index`; it does not open/close the audio device every block (the **AudioRenderer** keeps a single long-lived output stream for the whole run).

---

## Pygmu-specific idioms

**1. Impure PE that must run every block (MidiInPE)**  
MidiInPE is stateful and does not produce meaningful audio; it exists to drain the MIDI queue and run the callback. In pygmu, the only way to run a PE every block is to have it in the graph and pull it. So the script puts it in the mix with **gain 0**: `GainPE(midi_in_pe, 0.0)`. That branch is pulled once per block (because MixPE pulls all inputs), so MidiInPE runs every block and the callback sees all queued messages. Impure PEs with a single “logical” role (e.g. one MidiInPE) should have a single sink in the graph; here the single sink is that GainPE.

**2. Control streams via ConstantPE and mutation**  
KeyTracker uses **ConstantPE(0)** for trigger and for gain. The published API of ConstantPE is “output a constant value”; the value is normally fixed at construction. Here, the script **mutates** `ConstantPE._value` inside `note_on` / `note_off`. The next time that ConstantPE is rendered, it outputs the new value. So the same PE acts as a “control stream” that can change when the user hits or releases a key. This is an intentional abuse of the internal `_value` slot: it works because the render loop is single-threaded and the same KeyTracker (and its ConstantPEs) are the only writers. A future, more idiomatic option would be a PE designed for updatable control values (e.g. a setter or a dedicated “control” PE).

**3. TriggerPE and “play from start on gate high”**  
TriggerPE in **RETRIGGER** mode: when the trigger goes high (e.g. key down), it enters ACTIVE and plays the source from time 0; when the trigger goes low (key up), it stops; when the trigger goes high again, it **retriggers** (plays from start again). Each time TriggerPE enters ACTIVE it calls **source.reset_state()**, so the source (here TimeWarpPE) resets its read position and the slice always starts from the beginning on each note-on. That gives the expected “new note = new start” behavior.

**4. TimeWarpPE for pitch**  
TimeWarpPE resamples its source at a given rate (source samples per output sample). Rate &gt; 1 ⇒ faster playback ⇒ higher pitch; rate = 2^((note−60)/12) gives equal-temperament pitch per key. TimeWarpPE is stateful (tape head); reset_state() (called by TriggerPE on trigger) zeros the head so each note starts from the beginning of the slice.

**5. Mono slice for uniform channel count**  
MixPE requires all inputs to have the same channel count. The WAV slice may be stereo; the “silence” branch (MidiInPE) is mono. So the script converts the slice to mono with **SpatialPE(SLICE_SOURCE, method=SpatialAdapter(channels=1))**. SpatialAdapter(channels=1) downmixes to mono (e.g. average of L/R for stereo). All 26 mix inputs are then mono.

**6. Render loop and sample index**  
The script advances time by sample index: each iteration it calls `renderer.render(sample_index, BLOCK_SIZE)` then `sample_index += BLOCK_SIZE`. There is no sleep; the loop runs as fast as it can. The **AudioRenderer** uses a single long-lived output stream and **blocking** write, so each `render()` blocks until the device has consumed the block (or buffer space is available). That keeps playback in sync with the render loop.

**7. start() / stop() and PE lifecycle**  
Before the first render, the script calls `renderer.start()`, which walks the graph and calls **on_start()** on each PE (e.g. MidiInPE opens the MIDI port). After the loop, it calls `renderer.stop()`, which calls **on_stop()** (e.g. MidiInPE closes the port; TriggerPE/TimeWarpPE reset state). So “session” lifecycle (open/close hardware, reset state) is explicit and tied to the renderer, not to individual render() calls.

---

## Stumbling blocks / questions for maintainers

- **ConstantPE._value mutation**: Documented above as an intentional abuse. If a first-class “updatable constant” or “control PE” is added later, the script could be refactored to use it and the comment about “abuse” could be removed.
- **Callback `sample_index`**: The MIDI callback receives `(sample_index, message)` where `sample_index` is the start of the current render block. The script only uses it for logging. If you ever want to schedule events to block boundaries or quantize, this is the place; otherwise it can be ignored.
- **25 voices, no voice stealing**: The design is “one voice per key.” If you need a fixed pool of voices with stealing (e.g. 8 voices, new note steals oldest), that would require a different architecture (e.g. a pool of TriggerPEs and logic that assigns keys to voices and updates trigger/gain per voice).
