"""
Example: Slew Rate Limiting on a stringed instrument

Although SlewRatePE was originally conceived for limiting gesture-level signals
(e.g. limit change in pitch for a portamento effect), this is a test to hear its
effect on a tonal waveform.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path

import pygmu2 as pg
from examples_helper import pad_clip, run_demos

pg.set_sample_rate(44100)

AUDIO_DIR = Path(__file__).parent / "audio"
PLUCK_FILE = AUDIO_DIR / "uke_54.wav"

# Probe the voice file once at import time to get its length
PLUCK_SAMPLES = pg.WavReaderPE(str(PLUCK_FILE)).extent().end

def make_pluck():
    """Return a fresh WavReaderPE for thepluck file."""
    return pg.WavReaderPE(str(PLUCK_FILE))


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────

def demo_dry_pluck():
    """Play the unprocessed pluckfor reference."""
    pg.play(pg.GainPE(make_pluck(), gain=2.14))


def demo_slew_rate_limit():
    """Slew rate limit to varying degrees.
    """
    print("Demo: Slew Rate Limit")
    for rate in [10, 30, 100, 300, 1000]:
        pluck = make_pluck()
        srl = pg.SetExtentPE(pg.SlewLimiterPE(pluck, rate), 0, PLUCK_SAMPLES)
        print(f"  slew rate limit = {rate} units/second", flush=True)
        pg.play(pg.GainPE(pad_clip(srl), gain=2.15))
    print("Done!")


DEMOS = [
    ("Dry pluck (reference)", demo_dry_pluck),
    ("Stepped slew rate limit", demo_slew_rate_limit),
]

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demos(DEMOS)
