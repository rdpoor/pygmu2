"""
Example: Random Step - Poisson sample-and-hold random generator.

Outputs a piecewise-constant (stepped) random signal in [0, 1].  At each
sample the output jumps to a new uniform random value with probability
p = rate / sr, giving exponentially distributed hold times with mean sr/rate
samples (1/rate seconds).

This is the "stepped random voltages" complement to RandomValuePE (continuous
Ornstein-Uhlenbeck wandering) — analogous to the Buchla 266 Source of
Uncertainty stepped vs. fluctuating outputs.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pygmu2 as pg
from examples_helper import s, pad_clip, run_demos

pg.set_sample_rate(44100)


def random_to_frequency(r):
    """Map 0..1 to a frequency quantized to an equal-temperament pitch."""
    return pg.pitch_to_freq(np.round(48 + (r * 40)))


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────

def demo_fixed_rates():
    """Use RandomStepPE to control pitch with selected values for rate."""
    print("Demo: Random Step with different fixed rate values")
    for rate in [1, 3, 10, 30, 100]:
        step_value = pg.RandomStepPE(rate=rate)
        freq = pg.TransformPE(step_value, func=random_to_frequency)
        note = pg.BlitSawPE(frequency=freq)
        cropped_note = pg.CropPE(note, 0, s(4))
        print(f"  rate = {rate}")
        pg.play_offline(source=pad_clip(pg.GainPE(cropped_note, 0.1)))

def demo_ramped_rate():
    """Play notes with rate ramping exponentially from 1 to 100."""
    print("Demo: Random Step with rate ramping exponentially from 1 to 100")
    duration_samples = s(8)
    rate_ramp = pg.PiecewisePE(
        [(0, 1.0), (duration_samples, 100.0)],
        transition_type=pg.TransitionType.EXPONENTIAL
    )
    step_value = pg.RandomStepPE(rate=rate_ramp)
    freq = pg.TransformPE(step_value, func=random_to_frequency)
    note = pg.BlitSawPE(frequency=freq)
    cropped_note = pg.CropPE(note, 0, duration_samples)
    pg.play_offline(source=pad_clip(pg.GainPE(cropped_note, 0.1)))


DEMOS = [
    ("Random Step with different fixed rate values", demo_fixed_rates),
    ("Random Step with ramped rate", demo_ramped_rate),
]

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demos(DEMOS)
