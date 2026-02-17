"""
tralfam_eg.py

TralfamPE example: dry uke vs spectrum-randomized (Tralfam) uke.

Uses examples/audio/uke_54.wav. Demo 1 plays the file dry; demo 2 plays
the same source through TralfamPE (magnitudes kept, phases randomized).

Usage:
  uv run python examples/tralfam_eg.py
  uv run python examples/tralfam_eg.py 1
  uv run python examples/tralfam_eg.py 2
  uv run python examples/tralfam_eg.py a
"""

from pathlib import Path

import pygmu2 as pg

SAMPLE_RATE = 44100
pg.set_sample_rate(SAMPLE_RATE)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Demos: dry uke, then TralfamPE(uke)

AUDIO_DIR = Path(__file__).parent / "audio"
UKE_WAV = pg.WavReaderPE(str(AUDIO_DIR / "uke_54.wav"))
VOX_WAV = pg.WavReaderPE(str(AUDIO_DIR / "spoken_voice44.wav"))
# Pad very short slice with silence.  This makes the "tralfam" segment longer,
# so you don't hear the repetetive nature
MAN_WAV = pg.SequencePE(
    (pg.SlicePE(VOX_WAV, 62604, 16964), None), # "man!"
    (pg.CropPE(pg.ConstantPE(0, channels=2), 0, SAMPLE_RATE*2), None),
)

def demo_uke_dry():
    print("Demo 1: dry uke")
    print("--------")
    source = UKE_WAV
    # pg.play_offline(source=source, sample_rate=SAMPLE_RATE, path='uke_dry.wav')
    pg.play(source, SAMPLE_RATE)


def demo_uke_tralfam():
    print("Demo 2: TralfamPE(uke)")
    print("--------")
    source = UKE_WAV
    tralfam = pg.TralfamPE(source, seed=42)
    # pg.play_offline(source=tralfam, sample_rate=SAMPLE_RATE, path='tralfam.wav')
    pg.play(tralfam, SAMPLE_RATE)


def demo_uke_looped_tralfam():
    print("Demo 3: Looped TralfamPE(uke)")
    print("--------")
    source = UKE_WAV
    tralfam = pg.TralfamPE(source, seed=42)
    looped_tralfam = pg.LoopPE(tralfam, count=4)
    # pg.play_offline(source=looped_tralfam, sample_rate=SAMPLE_RATE, path='looped_tralfam.wav')
    pg.play(looped_tralfam, SAMPLE_RATE)


def demo_man_dry():
    print("Demo 1: dry man")
    print("--------")
    source = MAN_WAV
    # pg.play_offline(source=source, sample_rate=SAMPLE_RATE, path='man_dry.wav')
    pg.play(source, SAMPLE_RATE)


def demo_man_tralfam():
    print("Demo 2: TralfamPE(man)")
    print("--------")
    source = MAN_WAV
    tralfam = pg.TralfamPE(source, seed=42)
    # pg.play_offline(source=tralfam, sample_rate=SAMPLE_RATE, path='tralfam.wav')
    pg.play(tralfam, SAMPLE_RATE)


def demo_man_looped_tralfam():
    print("Demo 3: Looped TralfamPE(man)")
    print("--------")
    source = MAN_WAV
    tralfam = pg.TralfamPE(source, seed=42)
    looped_tralfam = pg.LoopPE(tralfam, count=8)
    # pg.play_offline(source=looped_tralfam, sample_rate=SAMPLE_RATE, path='looped_tralfam.wav')
    pg.play(looped_tralfam, SAMPLE_RATE)


DEMOS = {
    "Dry uke": demo_uke_dry,
    "TralfamPE(uke)": demo_uke_tralfam,
    "Looped TralfamPE(uke)": demo_uke_looped_tralfam,
    "Dry man": demo_man_dry,
    "TralfamPE(man)": demo_man_tralfam,
    "Looped TralfamPE(man)": demo_man_looped_tralfam,
}

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    import sys

    def resolve_choice(choice: str):
        """
        Return (name, fn) on valid choice, (None, None) otherwise.
        """
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
        names = list(DEMOS.keys())
        print("Available demos:")
        for i, name in enumerate(names, start=1):
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
                print(f"unrecognized choice {choice}, '?' to see choices")

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
            print(f"Invalid choice '{choice}'")
    else:
        print_menu()
        choose_and_play()
