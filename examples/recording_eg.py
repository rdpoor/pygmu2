#!/usr/bin/env python3
"""
Recording with DuplexRenderer: calibrate, record, punch in/out, mix back.

The recorded input shares the render clock (one full-duplex stream), so a
take is sample-accurate relative to playback up to a constant offset.
calibrate() measures that offset exactly — output to mic, path included —
so takes land on the timeline they were performed against.

Run demo 1 (calibrate) once per session, with the mic coupled to the
playback path (e.g. resting in a headphone earcup); then move the mic to
its stand and run the recording demos. Monitor your instrument externally
(amp, direct monitor, acoustically); pygmu2 only plays the backing track
and captures the mic.

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


def ticks(n_beats, frequency, start_beat=0, bpm=BPM):
    """One tick per beat, starting at start_beat."""
    period = s2s(60.0 / bpm)
    return MixPE(
        *[
            pg.DelayPE(tick(frequency), (start_beat + i) * period)
            for i in range(n_beats)
        ]
    )


def main_clicks(n_beats):
    """The clicks of the recorded section (starts after the count-in)."""
    return ticks(n_beats, 1109, start_beat=LEAD_IN)


def count_in():
    """Lead-in ticks, an octave higher so the count-in is unmistakable."""
    return ticks(LEAD_IN, 2217)


# Take files are temporary demo output — remove them on exit.
_takes = []


def _cleanup_takes():
    for path in _takes:
        try:
            os.remove(path)
        except OSError:
            pass


atexit.register(_cleanup_takes)


# One renderer shared across demos, so demo 1's calibration carries into
# the recording demos.
_renderer = None


def get_renderer():
    global _renderer
    if _renderer is None:
        _renderer = DuplexRenderer(sample_rate=SAMPLE_RATE, device=DEVICE)
    return _renderer


def require_calibration():
    """The recording demos apply the measured offset — refuse to run
    without one rather than silently recording misaligned takes."""
    renderer = get_renderer()
    if renderer.calibration_offset is None:
        print("Not calibrated yet — run demo 1 first (mic in earcup).")
        return None
    return renderer


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_calibrate():
    print("=== Calibrate: measure the round-trip offset ===")
    print("Couple the mic to the playback path (rest it in a headphone")
    print("earcup); a short sweep will play.")
    renderer = get_renderer()
    offset = renderer.calibrate()
    print(f"Calibration: {offset} samples " f"({1000.0 * offset / SAMPLE_RATE:.1f} ms)")
    print("Done — return the mic to its stand. Calibration holds for the")
    print("whole session (re-run only if device/rate/blocksize change).")


def demo_record_and_mix():
    print("=== Record 8 beats against a click, then play the mix ===")
    renderer = require_calibration()
    if renderer is None:
        return
    period = s2s(60.0 / BPM)
    clicks = main_clicks(8)
    renderer.set_source(MixPE(count_in(), clicks))
    renderer.start()
    seg = Segment(Extent(LEAD_IN * period, (LEAD_IN + 8) * period), "mix_take.wav")
    print("Four high ticks count you in, then recording starts — play along!")
    renderer.transport([seg]).wait()
    renderer.stop()
    _takes.append(seg.written_path)
    print(seg.recording.summary())
    print("Playing clicks + aligned take (count-in omitted)...")
    pg.play(MixPE(clicks, GainPE(seg.as_pe(), 0.8)))


def demo_record_stereo_compare():
    print("=== Record 8 beats; playback: clicks left, take right ===")
    renderer = require_calibration()
    if renderer is None:
        return
    period = s2s(60.0 / BPM)
    clicks = main_clicks(8)
    renderer.set_source(MixPE(count_in(), clicks))
    renderer.start()
    seg = Segment(Extent(LEAD_IN * period, (LEAD_IN + 8) * period), "stereo_take.wav")
    print("Four high ticks count you in, then recording starts — play along!")
    renderer.transport([seg]).wait()
    renderer.stop()
    _takes.append(seg.written_path)
    print(seg.recording.summary())
    print("Playing clicks in the LEFT channel, take in the RIGHT...")
    left = pg.SpatialPE(clicks, method=pg.SpatialLinear(azimuth=-90.0))
    right = pg.SpatialPE(seg.as_pe(), method=pg.SpatialLinear(azimuth=90.0))
    pg.play(MixPE(left, right))


def demo_punch_in_punch_out():
    print("=== Punch-in/punch-out: record beats 4-8 to a WAV file ===")
    renderer = require_calibration()
    if renderer is None:
        return
    period = s2s(60.0 / BPM)
    clicks = main_clicks(8)
    renderer.set_source(MixPE(count_in(), clicks))
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
    print("Playing clicks + take reloaded from disk (count-in omitted)...")
    reloaded = pg.DelayPE(pg.WavReaderPE(seg.written_path), seg.extent.start)
    pg.play(MixPE(clicks, GainPE(reloaded, 0.8)))


DEMOS = [
    ("Calibrate: measure the round-trip offset (mic in earcup)", demo_calibrate),
    ("Record 8 beats against a click, then play the mix", demo_record_and_mix),
    ("Record 8 beats; playback: clicks left, take right", demo_record_stereo_compare),
    ("Punch-in/punch-out: record beats 4-8 to a WAV file", demo_punch_in_punch_out),
]

if __name__ == "__main__":
    run_demos(DEMOS)
