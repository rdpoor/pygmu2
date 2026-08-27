"""
Example: random triggers, derived from a random gate.

Triggers are derived signals: GateToTriggerPE(RandomGatePE(rate)) emits one
+1 impulse at each rising edge of the random gate.

RATE CONVENTION (important): RandomGatePE(rate=r) TOGGLES at r events/sec,
so only half its events are rising edges — the derived trigger fires at
r/2 triggers/sec.  For t triggers/sec, construct RandomGatePE(rate=2*t).
(The former RandomTriggerPE(rate=t) fired at t/sec directly.)

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import pygmu2 as pg
from examples_helper import s, pad_clip, run_demos

pg.set_sample_rate(44100)


def random_trigger(rate: float, seed: int | None = None) -> pg.GateToTriggerPE:
    """One random trigger per 1/rate seconds on average (note the 2x:
    the underlying gate toggles, so only half its events are rising)."""
    return pg.GateToTriggerPE(pg.RandomGatePE(rate=2.0 * rate, seed=seed))


def ping():
    return pg.KarplusStrongPE(frequency=440.0)


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────


def demo_fixed_rates():
    """Random triggers at selected rates (derived from RandomGatePE)."""

    for rate in [1, 3, 10, 30, 100]:
        trigger_signal = random_trigger(rate=rate)
        ding = pg.TriggerRestartPE(trigger=trigger_signal, src=ping())
        print(f"  rate = {rate}", flush=True)
        cropped_ding = pg.CropPE(ding, 0, s(4))
        pg.play(pg.GainPE(pad_clip(cropped_ding), gain=1.58))


def demo_ramped_rate():
    """Play notes with rate ramping from 1 to 100"""

    duration_samples = s(8)
    rate_ramp = pg.PiecewisePE(
        [(0, 1.0), (duration_samples, 100.0)],
        transition_type=pg.TransitionType.EXPONENTIAL,
    )
    trigger_signal = random_trigger(rate=rate_ramp)
    ding = pg.TriggerRestartPE(trigger=trigger_signal, src=ping())
    cropped_ding = pg.CropPE(ding, 0, duration_samples)
    pg.play(pg.GainPE(pad_clip(cropped_ding), gain=1.74))


DEMOS = [
    ("Random Trigger with different fixed rate values", demo_fixed_rates),
    ("Random Trigger with rate ramping exponentially from 1 to 100", demo_ramped_rate),
]

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

README = """\
Random triggers — derived from RandomGatePE via GateToTriggerPE.

Emits +1 trigger impulses at random intervals following a Poisson process.
At each sample there is an independent probability p = rate/sr of firing,
giving exponentially distributed inter-event intervals.

Demo 1: fixed rates (1, 3, 10, 30, 100 Hz) triggering Karplus-Strong pings.
Demo 2: rate ramps exponentially from 1 to 100 Hz over 8 seconds.
"""

if __name__ == "__main__":
    run_demos(DEMOS, readme=README)
