"""
Starting template for new pygmu2 examples.

Template example with pg.play() and command-line demo selection.

Usage:
  python examples/00_template_eg.py
  python examples/00_template_eg.py 1
  python examples/00_template_eg.py 2
  python examples/00_template_eg.py a
"""

from pathlib import Path

import pygmu2 as pg
from examples_helper import run_demos

pg.set_sample_rate(44100)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# write your demos here, list demo names and demo functions in DEMOS


def demo_one():
    print("Demo one")
    print("--------")
    audio_dir = Path(__file__).parent / "audio"
    wav_path = audio_dir / "choir.wav"
    source = pg.WavReaderPE(str(wav_path))
    sample_rate = source.file_sample_rate or 44100
    pg.set_sample_rate(sample_rate)
    pg.play(pg.GainPE(source, gain=4.95))


def demo_two():
    print("Demo two")
    print("--------")
    sample_rate = 44100
    source = pg.SinePE(frequency=440.0, amplitude=0.3)
    source = pg.CropPE(source, 0, int(2.0 * sample_rate))
    pg.play(pg.GainPE(source, gain=1.67))


DEMOS = [
    ("Demo one", demo_one),
    ("Demo two", demo_two),
]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main

README = """\
Template example — starter file for new pygmu2 demos.

Shows the minimal boilerplate for a demo script: import pygmu2, define
one or more demo functions, register them in DEMOS, and call run_demos().

Demo 1: plays a WAV file through GainPE.
Demo 2: plays a 440 Hz sine wave for 2 seconds.
"""

if __name__ == "__main__":
    run_demos(DEMOS, readme=README)
