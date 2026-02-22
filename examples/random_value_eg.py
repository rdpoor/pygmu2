"""
Example: RandomValuePE - continuously wandering random voltage generator
inspired by the Buchla 266 "Source of Uncertainty" (Fluctuating Random
Voltages section).

At each sample a Bernoulli trial with probability p = rate/sr decides whether
to draw a new target value from Uniform[0, 1]; the output then exponentially
chases the current target with the same coefficient.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import pygmu2 as pg
from examples_helper import s, pad_clip, run_demos

pg.set_sample_rate(44100)


def random_to_frequency(r):
    """Map 0..1 to a frequency quantized to an equal-temperament pitch."""
    return pg.pitch_to_freq(48 + (r * 50))


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────

def demo_fixed_rates():
    """Use RandomValuePE to control filter cutoff with selected values for rate."""
    print("Demo: Random Value with different fixed rate values")
    for rate in [1, 3, 10, 30, 100]:
        random_value = pg.RandomValuePE(rate=rate)
        f0 = pg.TransformPE(random_value, func=random_to_frequency)
        note = pg.SuperSawPE(frequency=110)
        filtered_note = pg.BiquadPE(
            source=note,
            frequency=f0,
            q=7.0,
            mode=pg.BiquadMode.LOWPASS)
        cropped_note = pg.CropPE(filtered_note, 0, s(4))
        print(f"  rate = {rate}")
        pg.play(source=pad_clip(pg.GainPE(cropped_note, 0.1)))

def demo_ramped_rate():
    """Play notes with rate ramping exponentially from 100 to 1."""
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
        source=note,
        frequency=f0,
        q=7.0,
        mode=pg.BiquadMode.LOWPASS)
    cropped_note = pg.CropPE(filtered_note, 0, duration_samples)
    pg.play(source=pad_clip(pg.GainPE(cropped_note, 0.1)))

def demo_random_mumbling():
    """3-resonator filter vocal-tract model with wandering formants."""
    def map(x0, x1, y0, y1):
        """Return a function f(x) that maps [x0, x1] → [y0, y1]."""
        def f(x):
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return f

    print("Demo: Drunken robot")
    duration_samples = s(20)
    fo1 = pg.TransformPE(pg.RandomValuePE(rate=8),  map(0, 1, 250, 900))
    fo2 = pg.TransformPE(pg.RandomValuePE(rate=10), map(0, 1, 700, 2500))
    fo3 = pg.TransformPE(pg.RandomValuePE(rate=12), map(0, 1, 1700, 3500))
    src = pg.CachePE(pg.SuperSawPE(frequency=pg.pitch_to_freq(30)))
    raw_mix = pg.MixPE(
        pg.BiquadPE(src, frequency=fo1, q=5,  mode=pg.BiquadMode.BANDPASS),
        pg.BiquadPE(src, frequency=fo2, q=8,  mode=pg.BiquadMode.BANDPASS),
        pg.BiquadPE(src, frequency=fo3, q=12, mode=pg.BiquadMode.BANDPASS),
    )
    mix = pg.GainPE(pg.CropPE(raw_mix, 0, duration_samples), 0.25)
    pg.play_offline(mix)


DEMOS = [
    ("Random Value with different fixed rate values", demo_fixed_rates),
    ("Random Value with ramped rate", demo_ramped_rate),
    ("Random mumbling", demo_random_mumbling),
]

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demos(DEMOS)
