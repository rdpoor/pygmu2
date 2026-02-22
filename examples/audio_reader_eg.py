"""
audio_reader_eg.py

Demonstrates AudioReaderPE, which uses miniaudio to decode compressed audio
files (MP3, FLAC, OGG Vorbis, WAV) into memory and serve samples on demand.

Requires:
    pip install miniaudio

Usage:
  uv run python examples/audio_reader_eg.py
  uv run python examples/audio_reader_eg.py 1
  uv run python examples/audio_reader_eg.py 2
  uv run python examples/audio_reader_eg.py a
"""

from pathlib import Path

import pygmu2 as pg
from examples_helper import run_demos
pg.set_sample_rate(44100)

AUDIO_DIR = Path(__file__).parent / "audio"

# ------------------------------------------------------------------------------
# Demos
# ------------------------------------------------------------------------------

def demo_wav():
    """Play a WAV file via AudioReaderPE."""
    print("Demo: WAV via AudioReaderPE")
    print("---------------------------")
    audio_file = str(AUDIO_DIR / "djembe_hit.wav")
    source = pg.AudioReaderPE(audio_file, max_level_db = -3.0)
    print(f"  file: {audio_file}")
    print(f"  native rate : {source.file_sample_rate} Hz")
    print(f"  channels    : {source.channel_count()}")
    print(f"  duration    : {source.extent().end / 44100:.2f} s")
    pg.play(pg.GainPE(source, gain=0.71))


def demo_mp3():
    """Play an MP3 file via AudioReaderPE (resampled to 44100 Hz if needed)."""
    print("Demo: MP3 via AudioReaderPE")
    print("---------------------------")
    audio_file = str(AUDIO_DIR / "clown_horn.mp3")
    source = pg.AudioReaderPE(audio_file, max_level_db = -3.0)
    print(f"  file: {audio_file}")
    print(f"  native rate : {source.file_sample_rate} Hz")
    print(f"  channels    : {source.channel_count()}")
    print(f"  duration    : {source.extent().end / 44100:.2f} s")
    pg.play(pg.GainPE(source, gain=0.71))


DEMOS = [
    ("WAV file", demo_wav),
    ("MP3 file", demo_mp3),
]

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    run_demos(DEMOS)
