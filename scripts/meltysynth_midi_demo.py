#!/usr/bin/env python3
"""
Minimal demo: MIDI keyboard drives meltysynth SoundFont via pygmu2.

Uses MidiInPE to receive mido-style MIDI; callback forwards note_on/note_off
to MeltysynthPE.synthesizer. Audio is rendered as Snippets from the PE graph.

Requires: mido, sounddevice. Run from project root:
  uv run python scripts/meltysynth_midi_demo.py [path/to/soundfont.sf2] [--program 0-127]

Default soundfont: examples/audio/TimGM6mb.sf2 (if present).
Default program: 0 (Acoustic Grand Piano). Use --program N for GM program 0-127.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from pygmu2 import (
    AudioRenderer,
    MeltysynthPE,
    MidiInPE,
    MixPE,
    SpatialPE,
    SpatialAdapter,
    get_logger,
    setup_logging,
    set_sample_rate,
)
logger = get_logger("meltysynth_midi_demo")

SAMPLE_RATE = 44100
set_sample_rate(SAMPLE_RATE)
BLOCK_SIZE = 256

AUDIO_DIR = Path(__file__).parent.parent / "examples" / "audio"
DEFAULT_SOUNDFONT = AUDIO_DIR / "TimGM6mb.sf2"


def _default_soundfont_path() -> str:
    if DEFAULT_SOUNDFONT.exists():
        return str(DEFAULT_SOUNDFONT)
    return "TimGM6mb.sf2"


def make_meltysynth_midi_demo(soundfont_path: str, program: int | None = None):
    synth_pe = MeltysynthPE(soundfont_path, block_size=64, program=program)

    def _callback(sample_index, midi_message):
        logger.info(
            "midi sample_index=%s type=%s note=%s velocity=%s program=%s pitch=%s control=%s value=%s",
            sample_index,
            midi_message.type,
            getattr(midi_message, "note", None),
            getattr(midi_message, "velocity", None),
            getattr(midi_message, "program", None),
            getattr(midi_message, "pitch", None),
            getattr(midi_message, "control", None),
            getattr(midi_message, "value", None),
        )
        synth = synth_pe.synthesizer
        if synth is None:
            return
        if midi_message.type == "note_on" and midi_message.velocity > 0:
            # synth.note_on(midi_message.channel, midi_message.note, midi_message.velocity)
            # play it loud
            synth.note_on(midi_message.channel, midi_message.note, 100)
        elif midi_message.type == "note_off" or (
            midi_message.type == "note_on" and midi_message.velocity == 0
        ):
            synth.note_off(midi_message.channel, midi_message.note)
        elif midi_message.type == "program_change":
            synth.process_midi_message(
                midi_message.channel, 0xC0, midi_message.program, 0
            )
        elif midi_message.type == "pitchwheel":
            # Mido pitch is -8192..8191; MIDI raw is 14-bit 0..16383 (LSB, MSB)
            raw = midi_message.pitch + 8192
            data1 = raw & 0x7F
            data2 = (raw >> 7) & 0x7F
            synth.process_midi_message(midi_message.channel, 0xE0, data1, data2)
        elif midi_message.type == "control_change":
            # Hack: use knob K8 to select program
            if midi_message.control == 77:
                print_preset_name(synth, midi_message.value)
                synth.process_midi_message(0, 0xC0, midi_message.value, 0)
            else:
                synth.process_midi_message(
                    midi_message.channel, 0xB0, midi_message.control, midi_message.value
                )

    midi_in_pe = MidiInPE(callback=_callback)
    midi_2ch = SpatialPE(midi_in_pe, method=SpatialAdapter(channels=2))
    return MixPE(midi_2ch, synth_pe)

def print_preset_name(synth, patch):
    # channel 0 in your example
    ch = 0
    bank = synth._channels[ch].bank_number

    preset_id = (bank << 16) | patch
    preset = synth._preset_lookup.get(preset_id)

    # fallback logic (same as Synthesizer.note_on)
    if preset is None:
        gm_preset_id = patch if bank < 128 else (128 << 16)
        preset = synth._preset_lookup.get(gm_preset_id, synth._default_preset)

    print(f"Program change -> bank={bank} patch={patch}: {preset.name}")


def main():
    parser = argparse.ArgumentParser(
        description="MIDI keyboard drives meltysynth SoundFont (GM program 0-127)."
    )
    parser.add_argument(
        "soundfont",
        nargs="?",
        default=None,
        help="Path to .sf2 (default: examples/audio/TimGM6mb.sf2 if present)",
    )
    parser.add_argument(
        "--program",
        type=str,
        default=None,
        metavar="N",
        help="GM program 0-127 for channel 0 (default: 0 = Acoustic Grand Piano)",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    soundfont = args.soundfont if args.soundfont is not None else _default_soundfont_path()
    if not Path(soundfont).exists():
        print(f"SoundFont not found: {soundfont}", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1

    program = None
    if args.program is not None:
        try:
            program = int(args.program)
        except ValueError:
            print(f"Invalid --program: {args.program!r}; must be an integer 0-127", file=sys.stderr)
            return 1
        if not 0 <= program <= 127:
            print(f"Program must be 0-127, got {program}", file=sys.stderr)
            return 1

    logger.info(
        "soundfont=%s program=%s sample_rate=%s blocksize=%s",
        soundfont, program, SAMPLE_RATE, BLOCK_SIZE,
    )
    print("MIDI → Meltysynth SoundFont. Press keys; Ctrl+C to quit.")
    if program is not None:
        print(f"  Program: {program} (GM)")
    print()

    mix = make_meltysynth_midi_demo(soundfont, program=program)
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
        logger.info("stopped at sample_index=%s", sample_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
