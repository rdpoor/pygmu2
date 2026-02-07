"""
reverb_eg.py

Example: ReverbPE with fixed and time-varying wet/dry mix.

Usage:
  python examples/reverb_eg.py
  python examples/reverb_eg.py 1
  python examples/reverb_eg.py 2
  python examples/reverb_eg.py 3
  python examples/reverb_eg.py a
"""

from pathlib import Path

import pygmu2 as pg
pg.set_sample_rate(44100)



def _play(source, sample_rate):
    """Render a source to its full extent using AudioRenderer."""
    renderer = pg.AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(source)
    with renderer:
        renderer.start()
        renderer.play_extent()


def _load_sources():
    audio_dir = Path(__file__).parent / "audio"
    wav_path = audio_dir / "djembe44.wav"
    ir_path = audio_dir / "plate_ir44.wav"

    source = pg.WavReaderPE(str(wav_path))
    ir = pg.WavReaderPE(str(ir_path))
    sample_rate = source.file_sample_rate or 44100

    # Extend source by looping three times
    source = pg.LoopPE(source, count=3)

    return source, ir, sample_rate


# ------------------------------------------------------------------------------
# Demos


def demo_dry_only():
    print("Demo: dry only")
    print("--------------")
    source, _ir, sample_rate = _load_sources()
    _play(source, sample_rate)


def demo_fixed_mix():
    print("Demo: fixed mix (40% wet)")
    print("-------------------------")
    source, ir, sample_rate = _load_sources()

    reverb = pg.ReverbPE(source, ir, mix=0.4, normalize_ir=True)
    _play(reverb, sample_rate)


def demo_ramp_mix():
    print("Demo: ramp mix (wet -> dry)")
    print("----------------------------")
    source, ir, sample_rate = _load_sources()

    extent = source.extent()
    if extent.start is None or extent.end is None:
        raise ValueError("Expected finite source extent for ramp mix")
    duration = extent.end - extent.start

    mix_ramp = pg.PiecewisePE(
        [(0, 1.0), (duration, 0.0)],
        transition_type=pg.TransitionType.LINEAR,
        extend_mode=pg.ExtendMode.HOLD_LAST,
    )
    reverb = pg.ReverbPE(source, ir, mix=mix_ramp, normalize_ir=True)
    _play(reverb, sample_rate)


DEMOS = {
    "Dry only": demo_dry_only,
    "Fixed mix": demo_fixed_mix,
    "Ramp mix": demo_ramp_mix,
}


# ------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    import sys

    def resolve_choice(choice: str):
        """Return (name, fn) on valid choice, (None, None) otherwise."""
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
        """Present list of demo names, call user's choice. Loop until 'q'."""
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
