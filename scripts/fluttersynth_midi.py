#!/usr/bin/env python3
"""
FlutterSynth — polyphonic MIDI saw synth with velocity-coupled filter flutter.

Each simultaneous note occupies one polyphony slot; within each slot,
NUM_SYNTHS_PER_NOTE independent voices share the same pitch and amplitude but
have different random seeds, so their filter steps diverge immediately.

At each note_on both the filter step-rate and the accessible harmonic range
jump to velocity-scaled peaks, then decay back to a dark resting state over
SETTLE_TIME seconds.

Voice stealing: when all slots are occupied the oldest note is stolen.

Requires: mido, sounddevice.  Run from project root:
  uv run python scripts/fluttersynth_midi.py [--resonance R] [--settle SECS]
"""

import argparse
import random
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "src")

from pygmu2 import (
    AudioRenderer,
    BlitSawPE,
    CachePE,
    ControlPE,
    DelayPE,
    GainPE,
    LadderPE,
    LadderMode,
    MidiInPE,
    MixPE,
    RandomStepPE,
    RingModulatorPE,
    SlewLimiterPE,
    SlewMode,
    SpatialConstantPower,
    SpatialPE,
    SpatialAdapter,
    TransformPE,
    get_logger,
    pitch_to_freq,
    set_sample_rate,
    setup_logging,
)

# ── Audio engine ───────────────────────────────────────────────────────────────

SAMPLE_RATE = 22050
BLOCK_SIZE  = 512

set_sample_rate(SAMPLE_RATE)

logger = get_logger("fluttersynth_midi")

# ── Polyphony ──────────────────────────────────────────────────────────────────

MAX_POLYPHONY       = 8   # simultaneous notes (slots); oldest stolen when full
NUM_SYNTHS_PER_NOTE = 3   # independent filter voices per slot (different seeds)

# ── Filter harmonic table ──────────────────────────────────────────────────────

# Multipliers applied to the played note to generate cutoff candidates.
# Every entry is a harmonic, so all targets are spectrally related to the pitch.
FILTER_RATIOS = np.array([0.75,1,2, 4, 6, 8, 10, 12, 16, 20, 24, 32], dtype=float)

LPF_MIN_HZ =    90.0   # absolute floor clamped on computed cutoff values
LPF_MAX_HZ = 9_000.0  # absolute ceiling clamped on computed cutoff values

# ── Randomness — filter step rate ─────────────────────────────────────────────

MAX_RANDOMNESS = 84.0   # RandomStepPE rate (steps/s) at velocity=127, onset
MIN_RANDOMNESS =  2.0   # rate at settle end (0 = filter frozen)

# ── Filter harmonic range ──────────────────────────────────────────────────────

MAX_RANGE = 1.00   # fraction of table at velocity=127 onset (all harmonics)
MIN_RANGE = 0.25   # fraction of table at settle end (≈ 3 lowest harmonics)

# ── Envelope ───────────────────────────────────────────────────────────────────

SETTLE_TIME              = 0.45    # seconds for filter to decay to resting values
ENVELOPE_UPDATE_INTERVAL = 0.015  # seconds between envelope thread steps (~10 ms)

# ── Signal chain ───────────────────────────────────────────────────────────────

PITCH_SLEW_RATE  = 12_000.0   # Hz/s     — portamento glide between notes
FILTER_SLEW_RATE =      3.0   # octaves/s — cutoff glide between discrete harmonics
                               #   e.g. 3 oct/s: 200→400 Hz in ~0.33 s, 400→800 Hz in ~0.33 s
AMP_SLEW_RATE    =    200.0   # amp units/s — ~5 ms attack/release ramp (de-click)

RESONANCE   = 0.85   # Moog ladder resonance (0..1; >0.8 approaches self-oscillation)
OUTPUT_GAIN = 0.90   # target total loudness; divided across all voices and slots

# ── Stereo panning ─────────────────────────────────────────────────────────────

# Voices within each note are spread evenly across ±PAN_WIDTH degrees.
# With NUM_SYNTHS_PER_NOTE=2: voice 0 → -PAN_WIDTH, voice 1 → +PAN_WIDTH.
PAN_WIDTH = 75.0   # degrees (0 = centre, 90 = hard left/right)

# ── Delay / echo ───────────────────────────────────────────────────────────────

DELAY_TIME_SECS = 0.375   # echo delay in seconds (3/8 s — musical at most tempos)
DELAY_FEEDBACK  = 0.40    # echo return level (0 = no echo, 1 = infinite repeat)

# Initial values for shared control signals
INITIAL_FREQ_HZ = 440.0
INITIAL_AMP     = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Flutter envelope
# ──────────────────────────────────────────────────────────────────────────────

class FlutterEnvelope:
    """
    Velocity-scaled linear decay for one voice's filter_rate_ctrl and range_ctrl.

    On trigger both controls jump to velocity-scaled peaks, then a daemon thread
    steps them back to MIN_RANDOMNESS / MIN_RANGE over SETTLE_TIME seconds.
    A new trigger cancels any in-progress decay and restarts from the new peak.
    """

    def __init__(
        self,
        filter_rate_ctrl: ControlPE,
        range_ctrl: ControlPE,
        settle_time: float = SETTLE_TIME,
    ):
        self._filter_rate_ctrl = filter_rate_ctrl
        self._range_ctrl       = range_ctrl
        self._settle_time      = settle_time
        self._cancel           = threading.Event()

    def trigger(self, velocity_norm: float) -> None:
        """Fire from the given normalised velocity (0..1).

        Settle time is scaled by velocity:
          ff (vel=1) → 3× SETTLE_TIME  (quick, decisive)
          pp (vel=0) → 1× SETTLE_TIME  (slow, languid)
        """
        self._cancel.set()

        peak_rate  = MIN_RANDOMNESS + (MAX_RANDOMNESS - MIN_RANDOMNESS) * velocity_norm
        peak_range = MIN_RANGE      + (MAX_RANGE      - MIN_RANGE)      * velocity_norm

        # Settle time: long for soft notes, short for loud ones
        actual_settle = self._settle_time * (1.0 + 2.0 * (velocity_norm))

        # Set peak immediately before the decay thread runs
        self._filter_rate_ctrl.set_value(peak_rate)
        self._range_ctrl.set_value(peak_range)

        cancel = threading.Event()
        self._cancel = cancel
        threading.Thread(
            target=self._decay,
            args=(peak_rate, peak_range, actual_settle, cancel),
            daemon=True,
        ).start()

    def _decay(self, peak_rate, peak_range, settle_time, cancel: threading.Event) -> None:
        steps = max(1, int(settle_time / ENVELOPE_UPDATE_INTERVAL))
        for i in range(1, steps + 1):
            if cancel.is_set():
                return
            time.sleep(ENVELOPE_UPDATE_INTERVAL)
            t = i / steps
            self._filter_rate_ctrl.set_value(
                MIN_RANDOMNESS + (peak_rate  - MIN_RANDOMNESS) * (1.0 - t)
            )
            self._range_ctrl.set_value(
                MIN_RANGE      + (peak_range - MIN_RANGE)      * (1.0 - t)
            )


# ──────────────────────────────────────────────────────────────────────────────
# Voice builder
# ──────────────────────────────────────────────────────────────────────────────

def make_voice(
    freq_ctrl: ControlPE,
    freq_cached,
    amp_cached,
    resonance: float,
    settle_time: float,
    seed: int | None,
    voice_gain: float,
    pan_azimuth: float = 0.0,
) -> tuple:
    """
    Build one independent filter voice for a polyphony slot.

    Args:
        freq_ctrl:   Raw ControlPE (read by filter_freq_func closure for .value).
        freq_cached: CachePE(SlewLimiterPE(freq_ctrl)) — shared slewed pitch.
        amp_cached:  CachePE(amp_ctrl) — shared note gate.
        resonance:   Moog ladder resonance.
        settle_time: Filter decay time.
        seed:        RNG seed for this voice's RandomStepPE.
        voice_gain:  Per-voice output gain.
        pan_azimuth: Stereo position in degrees (-90=L, 0=centre, +90=R).

    Returns:
        (stereo_pe, FlutterEnvelope)
    """
    # Per-voice filter controls owned exclusively by this voice
    filter_rate_ctrl = ControlPE(initial_value=MIN_RANDOMNESS)
    range_ctrl       = ControlPE(initial_value=MIN_RANGE)
    envelope = FlutterEnvelope(filter_rate_ctrl, range_ctrl, settle_time=settle_time)

    # Oscillator: driven by the slot-shared slewed frequency
    osc = BlitSawPE(frequency=freq_cached)

    # Filter CV: step × range_ctrl → quantise to harmonics of the played note
    filter_step = RandomStepPE(rate=filter_rate_ctrl, seed=seed)
    scaled_cv   = RingModulatorPE(filter_step, range_ctrl, bias=0.0, mix=1.0)

    def filter_freq_func(r: np.ndarray) -> np.ndarray:
        # freq_ctrl.value gives the slot's target pitch (one block lag, inaudible)
        note_hz = freq_ctrl.value
        freqs   = np.clip(note_hz * FILTER_RATIOS, LPF_MIN_HZ, LPF_MAX_HZ)
        n   = len(freqs)
        idx = np.clip(np.floor(r * n).astype(int), 0, n - 1)
        return freqs[idx].astype(np.float32)

    cutoff_stepped = TransformPE(scaled_cv, func=filter_freq_func)
    cutoff = SlewLimiterPE(cutoff_stepped, rate=FILTER_SLEW_RATE, mode=SlewMode.LOGARITHMIC)

    filtered = LadderPE(osc, frequency=cutoff, resonance=resonance,
                        mode=LadderMode.LP24)

    # amp_cached gates on/off; voice_gain normalises for full polyphony
    gained = GainPE(GainPE(filtered, amp_cached), voice_gain)

    # Pan this voice to its assigned stereo position
    stereo = SpatialPE(gained, method=SpatialConstantPower(azimuth=pan_azimuth))

    return stereo, envelope


# ──────────────────────────────────────────────────────────────────────────────
# Polyphony slot
# ──────────────────────────────────────────────────────────────────────────────

class NoteSlot:
    """
    One polyphony slot: a freq_ctrl, amp_ctrl, shared slewed pitch, and
    NUM_SYNTHS_PER_NOTE independent filter voices.

    The voices share pitch/amplitude but have independent filter steps.
    All ControlPEs are wrapped in CachePE before fanning out to voices,
    satisfying the renderer's rule that stateful PEs may only have one sink.
    """

    def __init__(
        self,
        slot_index: int,
        base_seed: int,
        resonance: float,
        settle_time: float,
        voice_gain: float,
    ):
        self.note: int | None = None   # MIDI note number currently held, or None

        self.freq_ctrl = ControlPE(initial_value=INITIAL_FREQ_HZ)
        self.amp_ctrl  = ControlPE(initial_value=INITIAL_AMP)

        # One portamento stage shared by all voices; CachePE makes it pure so
        # multiple voice oscillators can consume it without triggering the
        # "multiple sinks" validator error.
        freq_cached = CachePE(
            SlewLimiterPE(self.freq_ctrl, rate=PITCH_SLEW_RATE, mode=SlewMode.LINEAR)
        )
        # Short linear ramp on amplitude eliminates note-on/off clicks.
        # SlewLimiterPE is stateful so it also needs a CachePE wrapper.
        amp_cached = CachePE(
            SlewLimiterPE(self.amp_ctrl, rate=AMP_SLEW_RATE, mode=SlewMode.LINEAR)
        )

        self._voices: list = []
        self._envelopes: list[FlutterEnvelope] = []

        # Spread voices evenly from -PAN_WIDTH to +PAN_WIDTH degrees.
        # Single voice → centre; two voices → hard left / hard right; etc.
        n = NUM_SYNTHS_PER_NOTE
        pan_angles = (
            [0.0] if n == 1
            else [-PAN_WIDTH + 2 * PAN_WIDTH * i / (n - 1) for i in range(n)]
        )

        for i in range(n):
            seed = base_seed + slot_index * n + i
            stereo, env = make_voice(
                self.freq_ctrl,
                freq_cached,
                amp_cached,
                resonance,
                settle_time,
                seed,
                voice_gain,
                pan_azimuth=pan_angles[i],
            )
            self._voices.append(stereo)
            self._envelopes.append(env)

        # Mix all (stereo) voices in this slot
        self.stereo_out = MixPE(*self._voices)

    def note_on(self, note: int, velocity: int) -> None:
        self.note = note
        self.freq_ctrl.set_value(float(pitch_to_freq(note)))
        self.amp_ctrl.set_value(1.0)
        vel_norm = velocity / 127.0
        for env in self._envelopes:
            env.trigger(vel_norm)

    def note_off(self, note: int) -> None:
        if self.note == note:
            self.note = None
            self.amp_ctrl.set_value(INITIAL_AMP)


# ──────────────────────────────────────────────────────────────────────────────
# Voice pool (polyphony manager)
# ──────────────────────────────────────────────────────────────────────────────

class VoicePool:
    """
    Manages MAX_POLYPHONY note slots with oldest-note voice stealing.

    note_on: re-triggers an existing slot for the same note, otherwise
             claims a free slot, otherwise steals the oldest one.
    note_off: silences the slot holding the matching note, or defers it
              when the sustain pedal is held.
    sustain:  CC 64 pedal — when released, silences all deferred note_offs.
    """

    def __init__(
        self,
        base_seed: int,
        resonance: float,
        settle_time: float,
    ):
        # Per-voice gain: normalise so playing all slots at once hits OUTPUT_GAIN
        voice_gain = OUTPUT_GAIN / (MAX_POLYPHONY * NUM_SYNTHS_PER_NOTE)

        self._slots = [
            NoteSlot(i, base_seed, resonance, settle_time, voice_gain)
            for i in range(MAX_POLYPHONY)
        ]
        self._ages  = [0] * MAX_POLYPHONY   # lower = older = first to steal
        self._clock = 0

        self._sustain_held:    bool      = False
        self._sustained_notes: set[int]  = set()   # notes released while pedal held

    def note_on(self, note: int, velocity: int) -> None:
        # If the note was being sustained, it is being re-struck — remove it
        # from the deferred set so a later note_off will act immediately.
        self._sustained_notes.discard(note)

        # Re-trigger existing slot for same note
        for slot in self._slots:
            if slot.note == note:
                slot.note_on(note, velocity)
                return

        # Claim a free slot
        for i, slot in enumerate(self._slots):
            if slot.note is None:
                slot.note_on(note, velocity)
                self._ages[i] = self._clock
                self._clock += 1
                return

        # Steal the oldest active slot
        oldest = min(range(MAX_POLYPHONY), key=lambda i: self._ages[i])
        self._slots[oldest].note_on(note, velocity)
        self._ages[oldest] = self._clock
        self._clock += 1

    def note_off(self, note: int) -> None:
        if self._sustain_held:
            self._sustained_notes.add(note)
        else:
            for slot in self._slots:
                slot.note_off(note)

    def sustain(self, value: int) -> None:
        """Handle MIDI CC 64 (sustain pedal). value >= 64 = pedal down."""
        pedal_down = value >= 64
        if pedal_down and not self._sustain_held:
            self._sustain_held = True
        elif not pedal_down and self._sustain_held:
            self._sustain_held = False
            # Release all notes that received note_off while the pedal was held
            for note in self._sustained_notes:
                for slot in self._slots:
                    slot.note_off(note)
            self._sustained_notes.clear()

    @property
    def stereo_outputs(self) -> list:
        return [slot.stereo_out for slot in self._slots]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"Polyphonic ({MAX_POLYPHONY} notes × {NUM_SYNTHS_PER_NOTE} voices) "
            "MIDI saw synth with velocity-coupled filter flutter."
        )
    )
    parser.add_argument(
        "--resonance",
        type=float,
        default=RESONANCE,
        metavar="R",
        help=f"Ladder resonance 0.0-1.0 (default: {RESONANCE}; >0.8 self-oscillates)",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=SETTLE_TIME,
        metavar="SECS",
        help=f"Seconds for filter to settle after note_on (default: {SETTLE_TIME})",
    )
    args = parser.parse_args()

    if not 0.0 <= args.resonance <= 1.0:
        print(f"--resonance must be 0.0-1.0, got {args.resonance}", file=sys.stderr)
        return 1
    if args.settle <= 0:
        print(f"--settle must be > 0, got {args.settle}", file=sys.stderr)
        return 1

    setup_logging(level="INFO")

    base_seed = random.randrange(2**31)
    logger.info(
        "polyphony=%d voices/note=%d resonance=%.2f settle=%.2fs seed=%d",
        MAX_POLYPHONY, NUM_SYNTHS_PER_NOTE, args.resonance, args.settle, base_seed,
    )

    print(
        f"FlutterSynth — {MAX_POLYPHONY}-note × {NUM_SYNTHS_PER_NOTE}-voice "
        "polyphonic saw + Moog flutter filter."
    )
    print(f"  resonance={args.resonance:.2f}  settle={args.settle:.2f}s  "
          f"base_seed={base_seed}")
    print("  Play notes — harder velocity = wider, faster filter flutter at onset.")
    print("  Ctrl+C to quit.")
    print()

    pool = VoicePool(
        base_seed=base_seed,
        resonance=args.resonance,
        settle_time=args.settle,
    )

    def _callback(sample_index: int, msg) -> None:
        note     = getattr(msg, "note",     None)
        velocity = getattr(msg, "velocity", 0)
        logger.debug("midi type=%s note=%s vel=%s", msg.type, note, velocity)

        if msg.type == "note_on" and velocity > 0:
            pool.note_on(note, velocity)
            logger.info("note_on  note=%d  hz=%.1f  vel=%d",
                        note, float(pitch_to_freq(note)), velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and velocity == 0):
            pool.note_off(note)
            logger.info("note_off note=%d", note)
        elif msg.type == "control_change" and msg.control == 64:
            pool.sustain(msg.value)
            logger.info("sustain  %s  (CC64 value=%d)",
                        "pedal_down" if msg.value >= 64 else "pedal_up", msg.value)

    # Mix all slot stereo outputs (voices already panned inside each slot)
    mixed = MixPE(*pool.stereo_outputs)


    # MidiInPE outputs zeros; mixed at gain 0 so the renderer pulls it
    # (firing callbacks) without adding audio.
    midi_pe = MidiInPE(callback=_callback)
    mix = MixPE(
        mixed,
        GainPE(SpatialPE(midi_pe, method=SpatialAdapter(channels=2)), 0.0),
    )

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE, blocksize=BLOCK_SIZE)
    renderer.set_source(mix)
    renderer.start()

    sample_index = 0
    try:
        while True:
            renderer.render(sample_index, BLOCK_SIZE)
            sample_index += BLOCK_SIZE
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        renderer.stop()
        logger.info("stopped at sample_index=%d", sample_index)

    return 0


if __name__ == "__main__":
    sys.exit(main())
