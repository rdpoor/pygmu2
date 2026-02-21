"""
Example: RandomTriggerPE - Poisson-process trigger generator.

Emits a +1 trigger impulse at each randomly timed jump event.  Jump timing
follows a Poisson process: at each sample there is an independent probability
p = rate / sr of firing, giving exponentially distributed inter-event intervals
with mean sr/rate samples (1/rate seconds).

This is the trigger-output analogue of RandomStepPE (which also samples a new
random value at each jump).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path
import pygmu2 as pg
import numpy as np

SRATE = 44100
pg.set_sample_rate(SRATE)

# Gap between clips in multi-clip demos
SILENCE_DUR = 0.5  # seconds

def s(secs):
    """Convert seconds to samples."""
    return pg.seconds_to_samples(secs, SRATE)

def pad_clip(pe):
    """Pad a finite PE with a short silence so clips are clearly separated."""
    n = pe.extent().end
    padded = pg.SetExtentPE(pe, 0, n + s(SILENCE_DUR))
    return padded

def ping():
    return pg.KarplusStrongPE(frequency=440.0)

# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────

def demo_fixed_rates():
    """Use RandomTriggerPE selected values for rate"""

    print("Demo: Random Trigger with different fixed rate values")
    for rate in [1, 3, 10, 30, 100]:
        trigger_signal = pg.RandomTriggerPE(rate=rate)
        ding = pg.TriggerRestartPE(trigger = trigger_signal, src = ping())
        print(f"  rate = {rate}")
        cropped_ding = pg.CropPE(ding, 0, s(4))
        pg.play(pad_clip(cropped_ding))
        
def demo_ramped_rate():
    """Play notes with rate ramping from 1 to 100"""

    print("Demo: Random Step with rate ramping exponentially from 1 to 100")
    duration_samples = s(8)
    rate_ramp = pg.PiecewisePE(
        [(0, 1.0), (duration_samples, 100.0)],
        transition_type=pg.TransitionType.EXPONENTIAL
    )
    trigger_signal = pg.RandomTriggerPE(rate=rate_ramp)
    ding = pg.TriggerRestartPE(trigger=trigger_signal, src=ping())
    cropped_ding = pg.CropPE(ding, 0, duration_samples)
    pg.play(pad_clip(cropped_ding))

DEMOS = {
    "Random Trigger with different fixed rate values": demo_fixed_rates,
    "Random Trigger with ramped rate": demo_ramped_rate,
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

