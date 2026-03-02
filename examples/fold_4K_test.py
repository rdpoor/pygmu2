"""
what does it sound like when you ring modulate a sound with high frequency
content with 4KHz?
"""

from pathlib import Path

import pygmu2 as pg
from examples_helper import run_demos
pg.set_sample_rate(44100)

AUDIO_DIR = Path(__file__).parent / "audio"

# ------------------------------------------------------------------------------
# Demos
# ------------------------------------------------------------------------------

def demo_original():
    """Play original sound"""
    audio_file = str(AUDIO_DIR / "strawberry-shaker__long_forte_shaken.mp3")
    carrier = pg.AudioReaderPE(audio_file, max_level_db = -3.0)
    pg.play_offline(pg.GainPE(carrier, gain=0.71), path="shaker_original.wav")


def demo_fold_4k():
    """Ring modulate at 4K with sine wave"""
    audio_file = str(AUDIO_DIR / "strawberry-shaker__long_forte_shaken.mp3")
    carrier = pg.AudioReaderPE(audio_file, max_level_db = -3.0)
    modulator = pg.SinePE(frequency=4000.0)
    ring_mod = pg.RingModulatorPE(carrier=carrier, modulator=modulator)
    pg.play_offline(pg.GainPE(ring_mod, gain=0.71), path="shaker_folded_4khz.wav")

DEMOS = [
    ("Original sound", demo_original),
    ("Ring modulated at 4KHz", demo_fold_4k),
]

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    run_demos(DEMOS)
