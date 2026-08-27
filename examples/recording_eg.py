#!/usr/bin/env python3
"""
Recording with DuplexRenderer: calibrate, record, punch in/out, mix back.

The recorded input shares the render clock (one full-duplex stream), so a
take is sample-accurate relative to playback up to a constant offset.
calibrate() measures that offset exactly — speaker to mic, room included —
so takes land sample-exact on the timeline they were performed against.

Monitor your instrument externally (amp, acoustically); pygmu2 only plays
the backing track and captures the mic.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import atexit
import os

from pygmu2 import (
    DecayingSinePE,
    DuplexRenderer,
    Extent,
    GainPE,
    MixPE,
    Segment,
)
import pygmu2 as pg
from examples_helper import run_demos

# Recording is hardware-facing: the sample rate must match your audio
# interface end-to-end (a mismatched link slips or resamples — see the
# README "Recording" section), and input/output must share one clock —
# a single duplex device, e.g. a macOS Aggregate Device. Set DEVICE to
# its name or index, or None for the system default.
SAMPLE_RATE = 48000
DEVICE = None

pg.set_sample_rate(SAMPLE_RATE)


def s2s(seconds):
    return int(round(seconds * SAMPLE_RATE))


BPM = 90
LEAD_IN = 4  # count-in beats before recording punches in


def tick(frequency):
    """A short metronome tick: a rapidly decaying sine."""
    return DecayingSinePE(frequency=frequency, amplitude=0.5, tau=0.02)


def click_track(n_beats, bpm=BPM, lead_in=0):
    """One tick per beat; lead-in ticks ring an octave higher so the
    count-in is unmistakable."""
    period = s2s(60.0 / bpm)
    beats = [
        pg.DelayPE(tick(2217 if i < lead_in else 1109), i * period)
        for i in range(lead_in + n_beats)
    ]
    return MixPE(*beats)


# Take files are temporary demo output — remove them on exit.
_takes = []


def _cleanup_takes():
    for path in _takes:
        try:
            os.remove(path)
        except OSError:
            pass


atexit.register(_cleanup_takes)


def make_renderer():
    renderer = DuplexRenderer(sample_rate=SAMPLE_RATE, device=DEVICE)
    print("Measuring round-trip offset (a short sweep will play)...")
    offset = renderer.calibrate()
    print(f"Calibration: {offset} samples " f"({1000.0 * offset / SAMPLE_RATE:.1f} ms)")
    return renderer


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_record_and_mix():
    print("=== Record 8 beats against a click, then play the mix ===")
    renderer = make_renderer()
    period = s2s(60.0 / BPM)
    backing = click_track(8, lead_in=LEAD_IN)
    renderer.set_source(backing)
    renderer.start()
    seg = Segment(Extent(LEAD_IN * period, (LEAD_IN + 8) * period), "mix_take.wav")
    print("Four high ticks count you in, then recording starts — play along!")
    renderer.transport([seg]).wait()
    renderer.stop()
    _takes.append(seg.written_path)
    print(seg.recording.summary())
    print("Playing backing + aligned take...")
    pg.play(MixPE(backing, GainPE(seg.as_pe(), 0.8)))


def demo_punch_in_punch_out():
    print("=== Punch-in/punch-out: record beats 4-8 to a WAV file ===")
    renderer = make_renderer()
    period = s2s(60.0 / BPM)
    backing = click_track(8, lead_in=LEAD_IN)
    renderer.set_source(backing)
    renderer.start()
    a = (LEAD_IN + 4) * period
    b = (LEAD_IN + 8) * period
    seg = Segment(Extent(a, b), "punch_take.wav")
    print("Four high ticks count you in; recording punches in at beat 4!")
    transport = renderer.transport([seg])
    transport.wait()  # or transport.stop() to punch out early
    renderer.stop()
    _takes.append(seg.written_path)
    print(f"Wrote {seg.written_path} ({seg.captured} samples)")
    print("The file IS the musical region: sample 0 == beat 4.")
    print("Playing backing + take reloaded from disk...")
    reloaded = pg.DelayPE(pg.WavReaderPE(seg.written_path), seg.extent.start)
    pg.play(MixPE(backing, GainPE(reloaded, 0.8)))


DEMOS = [
    ("Record 8 beats against a click, then play the mix", demo_record_and_mix),
    ("Punch-in/punch-out: record beats 4-8 to a WAV file", demo_punch_in_punch_out),
]

if __name__ == "__main__":
    run_demos(DEMOS)
