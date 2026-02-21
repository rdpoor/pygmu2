"""
Example: Random Gate - Poisson-process toggle gate

Outputs a gate signal (values 0 or 1) that toggles at each randomly timed
jump event.  Jump timing follows a Poisson process: at each sample there is an
independent probability p = rate / sr of toggling, giving exponentially
distributed hold durations with mean sr/rate samples (1/rate seconds).

At rate r, the gate is high for an expected 1/(2r) seconds then low for
1/(2r) seconds, alternating at random intervals.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path
import pygmu2 as pg

SRATE = 44100
pg.set_sample_rate(SRATE)

AUDIO_DIR = Path(__file__).parent / "audio"
CHOIR_FILE = AUDIO_DIR / "choir.wav"
SEED = 123

# Gap between clips in multi-clip demos
SILENCE_DUR = 0.5  # seconds

# Probe the choir file once at import time to get its length
CHOIR_SAMPLES = pg.WavReaderPE(str(CHOIR_FILE)).extent().end


def s(secs):
    """Convert seconds to samples."""
    return pg.seconds_to_samples(secs, SRATE)

def make_voice():
    """Return a fresh WavReaderPE for the voice file."""
    return pg.WavReaderPE(str(CHOIR_FILE))

def play_clip(pe):
    """Play a finite PE, appending a short silence so clips are clearly separated."""
    n = pe.extent().end
    padded = pg.SetExtentPE(pe, 0, n + s(SILENCE_DUR))
    pg.play(padded, SRATE)

# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────

def demo_fixed_rates():
    """Play gated choir with varying values for rate"""

    print("Demo: Random Gate with different fixed rate values")
    for rate in [1, 3, 10, 30, 100]:
        gate = pg.RandomGatePE(rate=rate, seed=SEED)
        gated_choir = pg.GainPE(make_voice(), gate)
        print(f"  rate = {rate}")
        play_clip(gated_choir)

def demo_ramped_rate():
    """Play gated choir with rate ramping from 1 to 100"""

    print("Demo: Random Gate with rate ramping exponentially from 1 to 100")
    looped_choir = pg.LoopPE(make_voice(), count=4) # extend choir
    rate_ramp = pg.PiecewisePE(
        [(0, 1.0), (looped_choir.extent().end, 100.0)],
        transition_type=pg.TransitionType.EXPONENTIAL
    )

    gate = pg.RandomGatePE(rate=rate_ramp, seed=SEED)
    gated_choir = pg.GainPE(looped_choir, gate)
    play_clip(gated_choir)


DEMOS = {
    "Random Gate with different fixed rate values": demo_fixed_rates,
    "Random Gate with ramped rate": demo_ramped_rate,
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

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
        for i, name in enumerate(DEMOS, start=1):
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
                print(f"Unrecognized choice '{choice}', '?' to see list")

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

