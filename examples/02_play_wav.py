"""
Example 02: Play WAV - Loading and playing audio files

Loads a WAV file and plays it through the audio output.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path

from pygmu2 import WavReaderPE
import pygmu2 as pg
from examples_helper import run_demos
pg.set_sample_rate(44100)


# Path to audio file (relative to this script)
AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "faun.wav"

def play_wav():
    print("=== pygmu2 Example 02: Play WAV ===", flush=True)
    print(f"Loading: {WAV_FILE}", flush=True)

    # Load the WAV file
    source_stream = WavReaderPE(str(WAV_FILE))

    # Get file info (file_sample_rate is available before configuration)
    file_sr = source_stream.file_sample_rate
    print(f"  Channels: {source_stream.channel_count()}", flush=True)
    print(f"  Sample rate: {file_sr} Hz", flush=True)

    # Now we can get extent (sample rate is set globally before construction)
    extent = source_stream.extent()
    duration_samples = extent.end - extent.start
    duration_seconds = duration_samples / file_sr
    print(f"  Duration: {duration_seconds:.2f} seconds ({duration_samples} samples)", flush=True)

    print(f"Playing...", flush=True)
    pg.play(pg.GainPE(source_stream, gain=1.67), sample_rate=file_sr)

    print("Done!", flush=True)

DEMOS = [
    ("Play .wav file", play_wav),
]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    run_demos(DEMOS)
