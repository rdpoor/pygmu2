#!/usr/bin/env python3
"""
Toy script: play a pitch-shifted spoken word on each MIDI keypress.

Notes 48–72 (inclusive): middle C (60) = no shift, 72 = one octave up.
Uses SlicePE(word from spoken_voice44.wav) + TimeWarpPE(rate) + TriggerPE(RETRIGGER).
25 separate PE chains, one per key; MixPE sums them. Note-off stops that key's playback.

Requires: mido, sounddevice. Run from project root:
  uv run python scripts/toy_midi_sampler.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from pygmu2 import (
    AudioRenderer,
    ConstantPE,
    GainPE,
    MidiInPE,
    MixPE,
    SlicePE,
    SpatialPE,
    SpatialAdapter,
    TimeWarpPE,
    TransformPE,
    TriggerPE,
    TriggerMode,
    WavReaderPE,
    get_logger,
    setup_logging,
)
logger = get_logger("toy_midi_sampler")

AUDIO_DIR = Path(__file__).parent.parent / "examples" / "audio"
VOICE_WAV = AUDIO_DIR / "spoken_voice44.wav"
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
NOTE_LO, NOTE_HI = 48, 73  # 25 keys: 48..72 inclusive

# Real-time check: we must render each block in less than this (seconds)
BLOCK_DURATION_SEC = BLOCK_SIZE / SAMPLE_RATE


def ss(t: float) -> int:
    """Seconds to samples at 44.1 kHz."""
    return int(round(t * SAMPLE_RATE))

WORDS_STREAM = WavReaderPE(str(VOICE_WAV))
SLICE_SOURCE = SlicePE(WORDS_STREAM, start=ss(1.407), duration=ss(0.483))
# Mono slice so MixPE gets uniform channel count (midi_silence is 1 ch)
SLICE_MONO = SpatialPE(SLICE_SOURCE, method=SpatialAdapter(channels=1))

class KeyTracker:
    def __init__(self, midi_pitch: int):
        self._trigger_stream = ConstantPE(0)
        self._gain_stream = ConstantPE(0)
        self._midi_pitch = midi_pitch
        # middle C => no change in pitch, octave up => 2x rate, etc.
        self._rate = 2.0 ** ((midi_pitch - 60) / 12.0)

    def note_on(self, velocity: int):
        # Changing the _value slot of ConstantPE is an abuse of its published 
        # API.  It works, but we should document why.
        self._trigger_stream._value = 1.0
        self._gain_stream._value = velocity / 127.0

    def note_off(self):
        self._trigger_stream._value = 0.0
        self._gain_stream._value = 0.0

    def get_pitched_slice(self):
        attenuated_slice = GainPE(SLICE_MONO, self._gain_stream)
        pitched_slice = TimeWarpPE(attenuated_slice, self._rate)
        triggered_slice = TriggerPE(
            pitched_slice,
            self._trigger_stream,
            trigger_mode=TriggerMode.RETRIGGER,
        )
        return triggered_slice

def make_toy_midi_sampler():
    # Build one KeyTracker per MIDI note in range
    key_dict = {}
    for midi_pitch in range(NOTE_LO, NOTE_HI):
        key_dict[midi_pitch] = KeyTracker(midi_pitch)

    # Callback invoked when MIDI messages are drained in MidiInPE._render
    def _callback(sample_index, midi_message):
        logger.info(
            "midi sample_index=%s type=%s note=%s velocity=%s",
            sample_index,
            midi_message.type,
            getattr(midi_message, "note", None),
            getattr(midi_message, "velocity", None),
        )
        if midi_message.type == "note_on":
            kt = key_dict.get(midi_message.note, None)
            if kt is not None:
                kt.note_on(midi_message.velocity)
                logger.debug("note_on note=%s velocity=%s", midi_message.note, midi_message.velocity)
            else:
                logger.debug("note_on note=%s out of range (ignored)", midi_message.note)
        elif midi_message.type == "note_off":
            kt = key_dict.get(midi_message.note, None)
            if kt is not None:
                kt.note_off()
                logger.debug("note_off note=%s", midi_message.note)
            else:
                logger.debug("note_off note=%s out of range (ignored)", midi_message.note)

    midi_in_pe = MidiInPE(callback=_callback)
    # MidiInPE must be pulled each block to drain the queue; mix it with gain 0
    midi_silence = GainPE(midi_in_pe, 0.0)
    pes = [midi_silence] + [key_dict[p].get_pitched_slice() for p in range(NOTE_LO, NOTE_HI)]
    return MixPE(*pes)

def main():
    setup_logging(level="INFO")
    logger.info(
        "starting: sample_rate=%s blocksize=%s notes=%s..%s",
        SAMPLE_RATE,
        BLOCK_SIZE,
        NOTE_LO,
        NOTE_HI - 1,
    )
    print("MIDI voice keys: notes 48–72 play pitch-shifted word (middle C = no shift)")
    print("  Press keys; release to stop. Ctrl+C to quit.")
    print()

    mix = make_toy_midi_sampler()
    logger.debug("built mix with %s inputs (1 silence + %s voices)", NOTE_HI - NOTE_LO + 1, NOTE_HI - NOTE_LO)

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE, blocksize=BLOCK_SIZE)
    renderer.set_source(mix)
    renderer.start()
    logger.debug("renderer started")

    sample_index = 0
    # wall_start = time.perf_counter()
    # last_status_at = wall_start
    # behind_count = 0
    try:
        while True:
            # t_before = time.perf_counter()
            renderer.render(sample_index, BLOCK_SIZE)
            # t_after = time.perf_counter()
            # render_sec = t_after - t_before
            # if render_sec > BLOCK_DURATION_SEC:
            #     behind_count += 1
            #     logger.warning(
            #         "render not keeping up: block took %.1f ms (need < %.1f ms)",
            #         render_sec * 1000,
            #         BLOCK_DURATION_SEC * 1000,
            #     )
            sample_index += BLOCK_SIZE
            # # Every ~1 s log timing: lead = seconds of audio we've rendered ahead of wall clock
            # now = time.perf_counter()
            # if now - last_status_at >= 1.0:
            #     wall_elapsed = now - wall_start
            #     lead_sec = (sample_index / SAMPLE_RATE) - wall_elapsed
            #     logger.info(
            #         "realtime: render_ms=%.1f block_ms=%.1f lead_sec=%.3f behind_count=%s",
            #         render_sec * 1000,
            #         BLOCK_DURATION_SEC * 1000,
            #         lead_sec,
            #         behind_count,
            #     )
            #     last_status_at = now
    except KeyboardInterrupt:
        logger.debug("keyboard interrupt")
        print("\nStopped.")
    finally:
        renderer.stop()
        logger.info("stopped at sample_index=%s", sample_index)


if __name__ == "__main__":
    main()
