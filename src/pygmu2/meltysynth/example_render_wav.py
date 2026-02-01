"""
Meltysynth example: render SoundFont synthesis to WAV files.

Replicates the examples from py-meltysynth main.py:
  - simple_chord: play middle C, E, G and write a 3-second WAV
  - flourish: play a MIDI file and write a WAV

Usage (from project root, with deps installed e.g. uv run):
  # Simple chord (requires a .sf2 file, e.g. TimGM6mb.sf2):
  uv run python -m pygmu2.meltysynth.example_render_wav simple_chord [path/to/soundfont.sf2]

  # MIDI file (requires .sf2 and .mid):
  uv run python -m pygmu2.meltysynth.example_render_wav flourish [path/to/soundfont.sf2] [path/to/file.mid]

  # Custom output path:
  uv run python -m pygmu2.meltysynth.example_render_wav simple_chord TimGM6mb.sf2 -o my_chord.wav

Default paths if omitted: TimGM6mb.sf2 (or examples/audio/TimGM6mb.sf2 when run from repo root), flourish.mid (current directory).
"""

import argparse
import sys
import time
import wave
from array import array
from collections.abc import Sequence
from pathlib import Path

from pygmu2.meltysynth import (
    MidiFile,
    MidiFileSequencer,
    SoundFont,
    Synthesizer,
    SynthesizerSettings,
    create_buffer,
)


def _default_soundfont_path() -> str:
    """Default .sf2 path: use project examples/audio/TimGM6mb.sf2 if present, else cwd."""
    # Repo root from this file: src/pygmu2/meltysynth/example_render_wav.py -> 4 levels up
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    in_repo = repo_root / "examples" / "audio" / "TimGM6mb.sf2"
    if in_repo.exists():
        return str(in_repo)
    return "TimGM6mb.sf2"


def write_wav_file(
    sample_rate: int,
    left: Sequence[float],
    right: Sequence[float],
    path: str,
) -> None:
    max_value = 0.0

    for t in range(len(left)):
        if abs(left[t]) > max_value:
            max_value = abs(left[t])
        if abs(right[t]) > max_value:
            max_value = abs(right[t])

    if max_value <= 0:
        max_value = 1.0
    a = 0.99 / max_value

    data = array("h")
    for t in range(len(left)):
        sample_left = int(32768 * a * left[t])
        sample_right = int(32768 * a * right[t])
        data.append(sample_left)
        data.append(sample_right)

    wav = wave.open(path, "wb")
    wav.setframerate(sample_rate)
    wav.setnchannels(2)
    wav.setsampwidth(2)
    wav.writeframesraw(data.tobytes())
    wav.close()


def simple_chord(sound_font_path: str, out_path: str = "simple_chord.wav") -> None:
    """Play middle C, E, G for 3 seconds and write a WAV file."""
    sound_font = SoundFont.from_file(sound_font_path)
    settings = SynthesizerSettings(44100)
    synthesizer = Synthesizer(sound_font, settings)

    synthesizer.note_on(0, 60, 100)  # middle C
    synthesizer.note_on(0, 64, 100)  # E
    synthesizer.note_on(0, 67, 100)  # G

    length_samples = 3 * settings.sample_rate
    left = create_buffer(length_samples)
    right = create_buffer(length_samples)

    start = time.perf_counter()
    synthesizer.render(left, right)
    elapsed = time.perf_counter() - start

    print(f"Rendered {length_samples} samples in {elapsed:.3f}s")
    write_wav_file(settings.sample_rate, left, right, out_path)
    print(f"Wrote {out_path}")


def flourish(
    sound_font_path: str,
    midi_path: str,
    out_path: str = "flourish.wav",
) -> None:
    """Play a MIDI file and write a WAV file."""
    sound_font = SoundFont.from_file(sound_font_path)
    settings = SynthesizerSettings(44100)
    synthesizer = Synthesizer(sound_font, settings)

    midi_file = MidiFile.from_file(midi_path)
    sequencer = MidiFileSequencer(synthesizer)
    sequencer.play(midi_file, loop=False)

    length_samples = int(settings.sample_rate * midi_file.length)
    left = create_buffer(length_samples)
    right = create_buffer(length_samples)

    start = time.perf_counter()
    sequencer.render(left, right)
    elapsed = time.perf_counter() - start

    print(f"Rendered {length_samples} samples ({midi_file.length:.2f}s) in {elapsed:.3f}s")
    write_wav_file(settings.sample_rate, left, right, out_path)
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Meltysynth: render SoundFont/MIDI to WAV (see module docstring)."
    )
    parser.add_argument(
        "example",
        choices=["simple_chord", "flourish"],
        help="Which example to run",
    )
    parser.add_argument(
        "soundfont",
        nargs="?",
        default=None,
        help="Path to .sf2 SoundFont (default: examples/audio/TimGM6mb.sf2 in repo, else TimGM6mb.sf2)",
    )
    parser.add_argument(
        "midi",
        nargs="?",
        default="flourish.mid",
        help="Path to .mid file (only for flourish; default: flourish.mid)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output WAV path (default: simple_chord.wav or flourish.wav)",
    )
    args = parser.parse_args()

    soundfont = args.soundfont if args.soundfont is not None else _default_soundfont_path()

    if args.example == "simple_chord":
        if not Path(soundfont).exists():
            print(f"SoundFont not found: {soundfont}", file=sys.stderr)
            print("Download a .sf2 (e.g. TimGM6mb.sf2) or pass its path.", file=sys.stderr)
            return 1
        simple_chord(soundfont, args.output or "simple_chord.wav")
    else:
        if not Path(soundfont).exists():
            print(f"SoundFont not found: {soundfont}", file=sys.stderr)
            return 1
        if not Path(args.midi).exists():
            print(f"MIDI file not found: {args.midi}", file=sys.stderr)
            return 1
        flourish(
            soundfont,
            args.midi,
            args.output or "flourish.wav",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
