"""
tralfam_eg.py

TralfamPE example: dry uke vs spectrum-randomized (Tralfam) uke.

Uses examples/audio/uke_54.wav. Demo 1 plays the file dry; demo 2 plays
the same source through TralfamPE (magnitudes kept, phases randomized).

Usage:
  uv run python examples/tralfam_eg.py
  uv run python examples/tralfam_eg.py 1
  uv run python examples/tralfam_eg.py 2
  uv run python examples/tralfam_eg.py a
"""

from pathlib import Path

import pygmu2 as pg
from examples_helper import run_demos

SAMPLE_RATE = 44100
pg.set_sample_rate(SAMPLE_RATE)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Demos: dry uke, then TralfamPE(uke)

AUDIO_DIR = Path(__file__).parent / "audio"
UKE_WAV = pg.WavReaderPE(str(AUDIO_DIR / "uke_54.wav"))
VOX_WAV = pg.WavReaderPE(str(AUDIO_DIR / "spoken_voice44.wav"))
SHORT_MAN = pg.SlicePE(VOX_WAV, 62604, 16964)  # "man!"


def demo_uke_dry():
    source = UKE_WAV
    # pg.play_offline(source=source, path='uke_dry.wav')
    pg.play(pg.GainPE(source, gain=0.71))


def demo_uke_tralfam():
    source = UKE_WAV
    tralfam = pg.TralfamPE(source, seed=42, normalize_peak=0.33)
    # pg.play_offline(source=tralfam, path='tralfam.wav')
    pg.play(pg.GainPE(tralfam, gain=1.52))


def demo_uke_looped_tralfam():
    source = UKE_WAV
    tralfam = pg.TralfamPE(source, seed=42, normalize_peak=0.33)
    looped_tralfam = pg.LoopPE(tralfam, count=4)
    # pg.play_offline(source=looped_tralfam, path='looped_tralfam.wav')
    pg.play(pg.GainPE(looped_tralfam, gain=1.52))


def demo_short_man_dry():
    source = SHORT_MAN
    # pg.play_offline(source=source, path='man_dry.wav')
    pg.play(pg.GainPE(source, gain=2.83))


def demo_short_man_tralfam():
    source = SHORT_MAN
    tralfam = pg.TralfamPE(source, seed=42, normalize_peak=0.33)
    # pg.play_offline(source=tralfam, path='short_man.wav')
    pg.play(pg.GainPE(tralfam, gain=1.52))


def demo_short_man_looped_tralfam():
    source = SHORT_MAN
    tralfam = pg.TralfamPE(source, seed=42, normalize_peak=0.33)
    looped_tralfam = pg.LoopPE(tralfam, count=8)
    # pg.play_offline(source=looped_tralfam, path='looped_short_man.wav')
    pg.play(pg.GainPE(looped_tralfam, gain=1.52))


def demo_padded_man_tralfam():
    # Pad with two seconds of silence to the short snippet before processing
    tralfam = pg.TralfamPE(
        SHORT_MAN,
        seed=42,
        padded_length=SHORT_MAN.extent().end + (SAMPLE_RATE * 2),
        normalize_peak=0.33,
    )
    # pg.play_offline(source=tralfam, path='padded_man.wav')
    pg.play(pg.GainPE(tralfam, gain=1.52))


def demo_padded_man_looped_tralfam():
    # Pad with two seconds of silence to the short snippet before processing
    tralfam = pg.TralfamPE(
        SHORT_MAN,
        seed=42,
        padded_length=SHORT_MAN.extent().end + (SAMPLE_RATE * 2),
        normalize_peak=0.33,
    )
    looped_tralfam = pg.LoopPE(tralfam, count=5)
    # pg.play_offline(source=looped_tralfam, path='looped_padded_man.wav')
    pg.play(pg.GainPE(looped_tralfam, gain=1.52))


DEMOS = [
    ("Dry uke", demo_uke_dry),
    ("TralfamPE(uke)", demo_uke_tralfam),
    ("Looped TralfamPE(uke)", demo_uke_looped_tralfam),
    ("Short, Dry man", demo_short_man_dry),
    ("TralfamPE(man)", demo_short_man_tralfam),
    (
        "Looped TralfamPE(man) - you can hear the loop points",
        demo_short_man_looped_tralfam,
    ),
    ("TralfamPE(man) padded with 2 seconds of 0s", demo_padded_man_tralfam),
    ("Looped TralfamPE(padded man)", demo_padded_man_looped_tralfam),
]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    run_demos(DEMOS)
