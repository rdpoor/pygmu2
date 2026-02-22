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

import pygmu2 as pg
import numpy as np
from examples_helper import s, pad_clip, run_demos

pg.set_sample_rate(44100)


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

DEMOS = [
    ("Random Trigger with different fixed rate values", demo_fixed_rates),
    ("Random Trigger with ramped rate", demo_ramped_rate),
]

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demos(DEMOS)

