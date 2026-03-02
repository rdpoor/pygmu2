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
from examples_helper import run_demos
pg.set_sample_rate(44100)


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
    source, _ir, sample_rate = _load_sources()
    pg.play(pg.GainPE(source, gain=0.81), sample_rate)


def demo_fixed_mix():
    source, ir, sample_rate = _load_sources()

    reverb = pg.ReverbPE(source, ir, mix=0.4, normalize_ir=True)
    pg.play(pg.GainPE(reverb, gain=1.31), sample_rate)


def demo_ramp_mix():
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
    pg.play(pg.GainPE(reverb, gain=1.08), sample_rate)


DEMOS = [
    ("Dry only", demo_dry_only),
    ("Fixed mix (40% wet)", demo_fixed_mix),
    ("Ramped mix (wet -> dry)", demo_ramp_mix),
]

# ------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    run_demos(DEMOS)
