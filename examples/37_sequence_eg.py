"""
37_sequence_eg.py

SequencePE example showing OVERLAP and NON_OVERLAP modes with audio material.
"""

from pathlib import Path

import pygmu2 as pg
from examples_helper import run_demos
pg.set_sample_rate(44100)



def _build_sources():
    audio_dir = Path(__file__).parent / "audio"
    choir_path = audio_dir / "choir.wav"
    source = pg.WavReaderPE(str(choir_path))
    sample_rate = source.file_sample_rate or 44100

    # Original choir
    choir = source

    # Pitch-shift down three semitones
    rate = pg.semitones_to_ratio(-3)
    choir_down = pg.TimeWarpPE(source, rate=rate)

    # Pitch-shift up four semitones
    rate = pg.semitones_to_ratio(4)
    choir_up = pg.TimeWarpPE(source, rate=rate)

    return choir, choir_down, choir_up, sample_rate


def demo_overlap():
    print("SequencePE with mode=OVERLAP")
    print("----------------------------")
    choir, choir_down, choir_up, sample_rate = _build_sources()

    seq = pg.SequencePE(
        (choir, 0),
        (choir_down, int(1.0 * sample_rate)),
        (choir_up, int(2.0 * sample_rate)),
        mode=pg.SequenceMode.OVERLAP,
    )
    pg.play(pg.GainPE(pg.CropPE(seq, 0, int(3.5 * sample_rate)), gain=2.54), sample_rate)


def demo_non_overlap():
    print("SequencePE with mode=NON_OVERLAP")
    print("--------------------------------")
    choir, choir_down, choir_up, sample_rate = _build_sources()

    seq = pg.SequencePE(
        (choir, 0),
        (choir_down, int(1.0 * sample_rate)),
        (choir_up, int(2.0 * sample_rate)),
        mode=pg.SequenceMode.NON_OVERLAP,
    )
    pg.play(pg.GainPE(pg.CropPE(seq, 0, int(3.5 * sample_rate)), gain=4.07), sample_rate)


DEMOS = [
    ("Demo overlap", demo_overlap),
    ("Demo non-overlap", demo_non_overlap),
]

if __name__ == "__main__":
    run_demos(DEMOS)
