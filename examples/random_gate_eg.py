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
from examples_helper import s, pad_clip, run_demos

pg.set_sample_rate(44100)

AUDIO_DIR = Path(__file__).parent / "audio"
CHOIR_FILE = AUDIO_DIR / "choir.wav"
SEED = 123

# Probe the choir file once at import time to get its length
CHOIR_SAMPLES = pg.WavReaderPE(str(CHOIR_FILE)).extent().end


def make_voice():
    """Return a fresh WavReaderPE for the voice file."""
    return pg.WavReaderPE(str(CHOIR_FILE))

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
        pg.play(pad_clip(gated_choir))

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


DEMOS = [
    ("Random Gate with different fixed rate values", demo_fixed_rates),
    ("Random Gate with ramped rate", demo_ramped_rate),
]

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demos(DEMOS)

