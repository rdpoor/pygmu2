"""
Example 24: SlicePE - quick snippet audition framework

Loads two WAV files and lets you define a small "playlist" of SlicePE snippets
by hand-editing start/duration values.

Edit the SNIPS list below (start/duration are in samples).

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pygmu2 import (
    AudioRenderer,
    GainPE,
    LoopPE,
    SequencePE,
    SlicePE,
    WavReaderPE,
    seconds_to_samples,
)


# Path to audio files (relative to this script)
AUDIO_DIR = Path(__file__).parent / "audio"
DRUMS_WAV = AUDIO_DIR / "acoustic_drums.wav" # 44100
VOICE_WAV = AUDIO_DIR / "spoken_voice44.wav" # 44100

SAMPLE_RATE = 44100

def ss(seconds):
    return int(round(seconds * SAMPLE_RATE))

def extract_slices() -> dict:
    """
    Generate a dict of named slices.
    """
    words_stream = WavReaderPE(str(VOICE_WAV))
    sample_rate = words_stream.file_sample_rate
    drums_stream = WavReaderPE(str(DRUMS_WAV))

    slices = {
        'more': SlicePE(words_stream, start=ss(0.0), duration=ss(0.266)), 
        'man': SlicePE(words_stream, start=ss(1.407), duration=ss(0.483)), 
        'so': SlicePE(words_stream, start=ss(2.888), duration=ss(0.440)),
        'out': SlicePE(words_stream, start=ss(3.353), duration=ss(0.607)),
        'cowbell': GainPE(SlicePE(drums_stream, start=ss(1.813), duration=ss(0.131)), 2.0),
        'kick': SlicePE(drums_stream, start=ss(2.590), duration=ss(0.302)),
    }
    return slices

def demo_slice_repertoire():
    print("Play extracted slices")

    slices = extract_slices()
    mix_stream = SequencePE([
        (slices['more'], ss(0)),
        (slices['man'], ss(0.5)),
        (slices['so'], ss(1.0)),
        (slices['out'], ss(1.5)),
        (slices['cowbell'], ss(2.0)),
        (slices['kick'], ss(2.5)),
        ])
    with AudioRenderer(sample_rate=SAMPLE_RATE) as renderer:
        renderer.set_source(mix_stream)
        renderer.start()
        renderer.play_extent()

def demo_slice_polka():
    print("Do a little dance")

    BPM = 120
    SECONDS_PER_BEAT = 60.0 / BPM
    slices = extract_slices()
    measure_stream = SequencePE([
        (slices['kick'], ss(0.0 * SECONDS_PER_BEAT)),
        (slices['cowbell'], ss(0.5 * SECONDS_PER_BEAT)),
        (slices['out'], ss(0.5 * SECONDS_PER_BEAT)),
        (slices['kick'], ss(1.0 * SECONDS_PER_BEAT)),
        (slices['cowbell'], ss(1.5 * SECONDS_PER_BEAT)),
        (slices['so'], ss(1.5 * SECONDS_PER_BEAT)),
        (slices['kick'], ss(2.0 * SECONDS_PER_BEAT)),
        (slices['cowbell'], ss(2.5 * SECONDS_PER_BEAT)),
        (slices['more'], ss(2.5 * SECONDS_PER_BEAT)),
        (slices['kick'], ss(3.0 * SECONDS_PER_BEAT)),
        (slices['cowbell'], ss(3.5 * SECONDS_PER_BEAT)),
        (slices['man'], ss(3.5 * SECONDS_PER_BEAT)),
        ], overlap=True)

    # Play the measure 4 times
    mix_stream = LoopPE(
        measure_stream,
        loop_start=ss(0), 
        loop_end=ss(4.0 * SECONDS_PER_BEAT), 
        count=4)

    with AudioRenderer(sample_rate=SAMPLE_RATE) as renderer:
        renderer.set_source(mix_stream)
        renderer.start()
        renderer.play_extent()

if __name__ == "__main__":
    import sys

    print("pygmu2 Slice Examples")
    print("=" * 50)
    print()
    
    demos = [
        ("1", "Play slices", demo_slice_repertoire),
        ("2", "Play polka", demo_slice_polka),
        ("a", "All demos", None),
    ]
    
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("pygmu2 SlicePE Examples")
        print("----------------------------")
        for key, name, _ in demos:
            print(f"{key}: {name}")
        print()
        choice = input(f"Choose a demo (1-{len(demos)-1} or 'a' for all): ").strip().lower()
        print()

    if choice == "a":
        for _, _, fn in demos:
            if fn is not None:
                fn()
    else:
        for key, _name, fn in demos:
            if key == choice and fn is not None:
                fn()
                break
        else:
            print("Invalid choice.")

