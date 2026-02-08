"""
00_template_eg.py

Template example with a _play() helper and command-line demo selection.

Usage:
  python examples/00_template_eg.py
  python examples/00_template_eg.py 1
  python examples/00_template_eg.py 2
  python examples/00_template_eg.py a
"""

from pathlib import Path

import pygmu2 as pg
pg.set_sample_rate(44100)



def _play(source, sample_rate):
    """
    Render a source to its full extent using AudioRenderer.
    """
    renderer = pg.AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(source)
    with renderer:
        renderer.start()
        renderer.play_extent()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# write your demos here, list demo names and demo functions in DEMOS

def demo_one():
    print("Demo one")
    print("--------")
    audio_dir = Path(__file__).parent / "audio"
    wav_path = audio_dir / "choir.wav"
    source = pg.WavReaderPE(str(wav_path))
    sample_rate = source.file_sample_rate or 44100
    _play(source, sample_rate)


def demo_two():
    print("Demo two")
    print("--------")
    sample_rate = 44100
    source = pg.SinePE(frequency=440.0, amplitude=0.3)
    source = pg.CropPE(source, 0, int(2.0 * sample_rate))
    _play(source, sample_rate)

DEMOS = {
    "Demo one": demo_one,
    "Demo two": demo_two,
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
            # Numerical choice (one-based)
            idx = int(choice)
            if 1 <= idx <= len(item_list):
                return item_list[idx - 1]
            else:
                return None, None

        if choice in DEMOS:
            # String match
            return choice, DEMOS[choice]
        else:
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
        """
        DEMOS: dict[demo_name, demo_function]
        Present list of demo names, call user's choice.  Loop until 'q'
        """
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
        # Command line choice: Run single demo (or 'a' for all') and quit
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
        # Enter interactive loop
        print_menu()
        choose_and_play()
