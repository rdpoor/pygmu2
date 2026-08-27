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

from pygmu2 import (
    CropPE,
    DuplexRenderer,
    Extent,
    GainPE,
    MixPE,
    Segment,
    SinePE,
)
import pygmu2 as pg
from examples_helper import run_demos

pg.set_sample_rate(44100)
SAMPLE_RATE = 44100


def s2s(seconds):
    return int(round(seconds * SAMPLE_RATE))


def click_track(n_beats, bpm=90):
    """Short sine beeps, one per beat — something to play along with."""
    period = s2s(60.0 / bpm)
    beep = CropPE(SinePE(frequency=1000, amplitude=0.5), 0, s2s(0.03))
    return MixPE(*[pg.DelayPE(beep, i * period) for i in range(n_beats)])


def make_renderer():
    renderer = DuplexRenderer(sample_rate=SAMPLE_RATE)
    print("Measuring round-trip offset (a short noise click will play)...")
    offset = renderer.calibrate()
    print(f"Calibration: {offset} samples " f"({1000.0 * offset / SAMPLE_RATE:.1f} ms)")
    return renderer


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_record_and_mix():
    print("=== Record 8 beats against a click, then play the mix ===")
    renderer = make_renderer()
    backing = click_track(8)
    renderer.set_source(backing)
    renderer.start()
    print("Recording — play along with the click!")
    take = renderer.record_extent()
    renderer.stop()
    print(take.summary())
    print("Playing backing + aligned take...")
    pg.play(MixPE(backing, GainPE(take.as_pe(), 0.8)))


def demo_punch_in_punch_out():
    print("=== Punch-in/punch-out: record beats 4-8 to a WAV file ===")
    renderer = make_renderer()
    backing = click_track(8)
    renderer.set_source(backing)
    renderer.start()
    period = s2s(60.0 / 90)
    seg = Segment(Extent(4 * period, 8 * period), "punch_take.wav")
    print("Recording punches in at beat 4 — play along!")
    transport = renderer.transport([seg])
    transport.wait()  # or transport.stop() to punch out early
    renderer.stop()
    print(f"Wrote {seg.written_path} ({seg.captured} samples)")
    print("The file IS the musical region: sample 0 == beat 4.")
    print("Playing backing + take reloaded from disk...")
    reloaded = pg.DelayPE(pg.WavReaderPE(seg.written_path), seg.extent.start)
    pg.play(MixPE(backing, GainPE(reloaded, 0.8)))


DEMOS = [
    demo_record_and_mix,
    demo_punch_in_punch_out,
]

if __name__ == "__main__":
    run_demos(DEMOS)
