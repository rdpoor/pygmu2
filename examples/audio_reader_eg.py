"""
audio_reader_eg.py

Demonstrates AudioReaderPE, which uses miniaudio to decode compressed audio
files (MP3, FLAC, OGG Vorbis, WAV) into memory and serve samples on demand.

Requires:
    pip install miniaudio

Usage:
  uv run python examples/audio_reader_eg.py
  uv run python examples/audio_reader_eg.py 1
  uv run python examples/audio_reader_eg.py 2
  uv run python examples/audio_reader_eg.py a
"""

from pathlib import Path

import pygmu2 as pg
pg.set_sample_rate(44100)

AUDIO_DIR = Path(__file__).parent / "audio"

# ------------------------------------------------------------------------------
# Demos
# ------------------------------------------------------------------------------

def demo_wav():
    """Play a WAV file via AudioReaderPE."""
    print("Demo: WAV via AudioReaderPE")
    print("---------------------------")
    audio_file = str(AUDIO_DIR / "djembe_hit.wav")
    source = pg.AudioReaderPE(audio_file, max_level_db = -3.0)
    print(f"  file: {audio_file}")
    print(f"  native rate : {source.file_sample_rate} Hz")
    print(f"  channels    : {source.channel_count()}")
    print(f"  duration    : {source.extent().end / 44100:.2f} s")
    pg.play(source)


def demo_mp3():
    """Play an MP3 file via AudioReaderPE (resampled to 44100 Hz if needed)."""
    print("Demo: MP3 via AudioReaderPE")
    print("---------------------------")
    audio_file = str(AUDIO_DIR / "clown_horn.mp3")
    source = pg.AudioReaderPE(audio_file, max_level_db = -3.0)
    print(f"  file: {audio_file}")
    print(f"  native rate : {source.file_sample_rate} Hz")
    print(f"  channels    : {source.channel_count()}")
    print(f"  duration    : {source.extent().end / 44100:.2f} s")
    pg.play(source)


DEMOS = {
    "WAV file": demo_wav,
    "MP3 file": demo_mp3,
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    def resolve_choice(choice: str):
        item_list = list(DEMOS.items())
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(item_list):
                return item_list[idx - 1]
            return None, None
        if choice in DEMOS:
            return choice, DEMOS[choice]
        return None, None

    def print_menu():
        print("Available demos:")
        for i, name in enumerate(DEMOS.keys(), start=1):
            print(f"  {i}: {name}")
        print("  ?: show list")
        print("  a: run all")
        print("  q: quit")

    def choose_and_play():
        while True:
            choice = input("Select demo (name or number): ").strip()
            if choice.lower() == "q":
                break
            if choice.lower() == "a":
                for fn in DEMOS.values():
                    fn()
                continue
            if choice == "?":
                print_menu()
                continue
            _name, fn = resolve_choice(choice)
            if fn is not None:
                fn()
            else:
                print(f"unrecognized choice {choice!r}, '?' to see choices")

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
        if choice == "a":
            for fn in DEMOS.values():
                fn()
            raise SystemExit(0)
        _name, fn = resolve_choice(choice)
        if fn is not None:
            fn()
        else:
            print(f"Invalid choice {choice!r}")
    else:
        print_menu()
        choose_and_play()
