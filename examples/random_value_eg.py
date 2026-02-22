"""
Example: RandomValuePE - random value generator inspired by the Buchla 266
"Source of Uncertainty" (Fluctuating Random Voltages section).

Implements an Ornstein-Uhlenbeck process:

    y[n] = y[n-1] + α * (noise[n] - y[n-1])

where α = rate / sr.  This is identical to SlewLimiterPE(EXPONENTIAL), so the
implementation is a pure composition:

    NoisePE(0..1, seed) ──► SlewLimiterPE(rate, EXPONENTIAL) ──► output

Output is bounded in [0, 1] because each y[n] is a convex combination of
y[n-1] ∈ [0,1] and noise[n] ∈ [0,1].

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path
import pygmu2 as pg
import numpy as np

SRATE = 44100
pg.set_sample_rate(SRATE)

AUDIO_DIR = Path(__file__).parent / "audio"

# Gap between clips in multi-clip demos
SILENCE_DUR = 0.5  # seconds

def s(secs):
    """Convert seconds to samples."""
    return pg.seconds_to_samples(secs, SRATE)

def pad_clip(pe):
    """Play a finite PE, appending a short silence so clips are clearly separated."""
    n = pe.extent().end
    padded = pg.SetExtentPE(pe, 0, n + s(SILENCE_DUR))
    return padded

def random_to_frequency(r):
    """map 0..1 to a frequency quantized to an equal-temperment pitch"""
    return pg.pitch_to_freq(48 + (r * 50))

# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────

def demo_fixed_rates():
    """Use RandomValuePE to control pitch with selected values for rate"""

    print("Demo: Random Value with different fixed rate values")
    for rate in [1, 3, 10, 30, 100]:
        random_value = pg.RandomValuePE(rate=rate)
        f0 = pg.TransformPE(random_value, func=random_to_frequency)
        note = pg.SuperSawPE(frequency=110)
        filtered_note = pg.BiquadPE(
            source = note,
            frequency = f0,
            q = 7.0,
            mode = pg.BiquadMode.LOWPASS)
        cropped_note = pg.CropPE(filtered_note, 0, s(4)) # crop to 4 seconds
        print(f"  rate = {rate}")
        attenuated_note = pg.GainPE(cropped_note, 0.1)
        pg.play(source=pad_clip(attenuated_note))
        
def demo_ramped_rate():
    """Play notes with rate ramping from 100 to 1"""

    print("Demo: Random Value with rate ramping exponentially from 100 to 1")
    duration_samples = s(10)
    rate_ramp = pg.PiecewisePE(
        [(0, 100.0), (duration_samples, 1.0)],
        transition_type=pg.TransitionType.EXPONENTIAL
    )
    random_value = pg.RandomValuePE(rate=rate_ramp)
    f0 = pg.TransformPE(random_value, func=random_to_frequency)
    note = pg.SuperSawPE(frequency=110)
    filtered_note = pg.BiquadPE(
        source = note,
        frequency = f0,
        q = 7.0,
        mode = pg.BiquadMode.LOWPASS)
    cropped_note = pg.CropPE(filtered_note, 0, duration_samples)
    attenuated_note = pg.GainPE(cropped_note, 0.1)
    pg.play(source=pad_clip(attenuated_note))

def demo_random_mumbling():
    """3-resonator filter vocal-tract model"""
    def map(x0, x1, y0, y1):
        """return a function f(x) that maps [x0, x1] => [y0, y1]"""
        def f(x):
            y = y0 + (y1-y0)*(x-x0)/(x1-x0)
            return y
        return f

    print("Demo: Drunken robot")
    duration_samples = s(20)
    # Compute three randomly wandering formant frequencies
    fo1 = pg.TransformPE(pg.RandomValuePE(rate=8), map(0, 1, 250, 900))
    fo2 = pg.TransformPE(pg.RandomValuePE(rate=10), map(0, 1, 700, 2500))
    fo3 = pg.TransformPE(pg.RandomValuePE(rate=12), map(0, 1, 1700, 3500))
    q1 = 5
    q2 = 8
    q3 = 12
    src = pg.CachePE(pg.SuperSawPE(frequency=pg.pitch_to_freq(30)))
    raw_mix = pg.MixPE(
        pg.BiquadPE(src, frequency=fo1, q = q1, mode=pg.BiquadMode.BANDPASS),
        pg.BiquadPE(src, frequency=fo2, q = q2, mode=pg.BiquadMode.BANDPASS),
        pg.BiquadPE(src, frequency=fo3, q = q3, mode=pg.BiquadMode.BANDPASS),
    )
    mix = pg.GainPE(pg.CropPE(raw_mix, 0, duration_samples), 0.25)
    pg.play_offline(mix)


DEMOS = {
    "Random Value with different fixed rate values": demo_fixed_rates,
    "Random Value with ramped rate": demo_ramped_rate,
    "Random mumbling": demo_random_mumbling,
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

