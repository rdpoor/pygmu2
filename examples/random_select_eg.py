# examples/random_select.py

"""
RandomSelectPE example (new TriggerSignal/GateSignal conventions):
- A TriggerSignal is an event stream with samples in {..., -1, 0, +1, ...}
  where + means a rising-edge event and - means a falling-edge event.
- RandomSelectPE chooses one of N inputs on each +event and (via TriggerRestartPE)
  restarts that chosen source from local time 0.

This example uses a simple PeriodicSignal to generate periodic trigger events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pygmu2 as pg
from examples_helper import run_demos


def demo_weighted_pitch():
    SAMPLE_RATE = 44100
    pg.set_sample_rate(SAMPLE_RATE)
    pg.set_global_logging(level="INFO")

    weighted_boops = [
        (pg.SinePE(frequency=pg.pitch_to_freq(55), amplitude=0.3), 0.1),
        (pg.SinePE(frequency=pg.pitch_to_freq(57), amplitude=0.3), 0.4),
        (pg.SinePE(frequency=pg.pitch_to_freq(62), amplitude=0.3), 0.2),
        (pg.SinePE(frequency=pg.pitch_to_freq(64), amplitude=0.3), 0.3),
        (pg.SinePE(frequency=pg.pitch_to_freq(69), amplitude=0.3), 0.4),
        (pg.SinePE(frequency=pg.pitch_to_freq(71), amplitude=0.3), 0.4),
    ]
    inputs = [pe for (pe, _w) in weighted_boops]
    weights = [w for (_pe, w) in weighted_boops]

    # TriggerSignal: +1 impulses at 8 Hz
    trigger = pg.PeriodicTrigger(hz=8.0)

    chooser = pg.RandomSelectPE(
        trigger=trigger,
        inputs=inputs,
        weights=weights,
        seed=1234,
    )

    duration_seconds = 10
    duration_samples = int(duration_seconds * SAMPLE_RATE)

    pg.play(
        pg.GainPE(pg.CropPE(chooser, 0, duration_samples), gain=1.67),
        sample_rate=SAMPLE_RATE,
    )
    print("Done!\n", flush=True)


def demo_weighted_pitch_one_osc():
    SAMPLE_RATE = 44100
    pg.set_sample_rate(SAMPLE_RATE)
    pg.set_global_logging(level="INFO")

    # Similar to demo_weighted_pitch(), but selects from weighted *frequencies*
    # and feeds them into one oscillator.
    # Benefit: avoids clicks from starting/stopping the oscillator.
    weighted_freqs = [
        (pg.ConstantPE(pg.pitch_to_freq(55)), 0.1),
        (pg.ConstantPE(pg.pitch_to_freq(57)), 0.4),
        (pg.ConstantPE(pg.pitch_to_freq(62)), 0.2),
        (pg.ConstantPE(pg.pitch_to_freq(64)), 0.3),
        (pg.ConstantPE(pg.pitch_to_freq(69)), 0.4),
        (pg.ConstantPE(pg.pitch_to_freq(71)), 0.1),
    ]
    freq_inputs = [pe for (pe, _w) in weighted_freqs]
    freq_weights = [w for (_pe, w) in weighted_freqs]

    trigger = pg.PeriodicTrigger(hz=8.0)

    chooser = pg.RandomSelectPE(
        trigger=trigger,
        inputs=freq_inputs,
        weights=freq_weights,
        seed=1234,
    )

    osc = pg.SinePE(frequency=chooser, amplitude=0.3)

    duration_seconds = 10
    duration_samples = int(duration_seconds * SAMPLE_RATE)

    pg.play(
        pg.GainPE(pg.CropPE(osc, 0, duration_samples), gain=1.67),
        sample_rate=SAMPLE_RATE,
    )
    print("Done!\n", flush=True)


def demo_djembe_fury():
    SAMPLE_RATE = 44100
    pg.set_sample_rate(SAMPLE_RATE)
    pg.set_global_logging(level="INFO")

    AUDIO_DIR = Path(__file__).parent / "audio"
    WAV_FILE = AUDIO_DIR / "djembe44.wav"

    source_stream = pg.WavReaderPE(str(WAV_FILE))
    sample_rate = source_stream.file_sample_rate or 44100
    pg.set_sample_rate(sample_rate)

    def start_dur(start: int, end: int) -> tuple[int, int]:
        return (start, end - start)

    # Ten slices from the file (as in the original example)
    slices = [
        pg.SlicePE(source_stream, *start_dur(0, 13811)),          # 0
        pg.SlicePE(source_stream, *start_dur(13811, 20882)),      # 1
        pg.SlicePE(source_stream, *start_dur(20882, 35331)),      # 2
        pg.SlicePE(source_stream, *start_dur(35331, 42732)),      # 3
        pg.SlicePE(source_stream, *start_dur(42732, 57006)),      # 4
        pg.SlicePE(source_stream, *start_dur(57006, 71456)),      # 5
        pg.SlicePE(source_stream, *start_dur(71456, 78857)),      # 6
        pg.SlicePE(source_stream, *start_dur(78857, 93130)),      # 7
        pg.SlicePE(source_stream, *start_dur(93130, 100355)),     # 8
        pg.SlicePE(source_stream, *start_dur(100355, 114541)),    # 9
    ]

    # 10 triggers/sec
    trigger = pg.PeriodicTrigger(hz=10.0)

    chooser = pg.RandomSelectPE(
        trigger=trigger,
        inputs=slices,
        seed=1234,
    )

    duration_seconds = 20
    duration_samples = int(duration_seconds * sample_rate)

    pg.play(
        pg.GainPE(pg.CropPE(chooser, 0, duration_samples), gain=0.81),
        sample_rate=sample_rate,
    )
    print("Done!\n", flush=True)


DEMOS = [
    ("Demo weighted pitches", demo_weighted_pitch),
    ("Demo one oscillator (freq select)", demo_weighted_pitch_one_osc),
    ("Demo djembe fury", demo_djembe_fury),
]

README = """\
RandomSelectPE — weighted random selection from multiple inputs.

Uses TriggerSignal conventions (+1 = rising-edge event) to choose one of N
inputs on each trigger.  TriggerRestartPE restarts the chosen source from
local time 0 so each hit sounds fresh.

Demo 1: weighted pitch selection from six sine oscillators.
Demo 2: same idea but selects *frequencies* fed to a single oscillator
        (avoids clicks from starting/stopping separate oscillators).
Demo 3: randomly selects slices from a djembe recording at 10 Hz.
"""

if __name__ == "__main__":
    run_demos(DEMOS, readme=README)
