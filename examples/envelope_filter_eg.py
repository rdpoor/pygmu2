"""
envelope_filter_eg.py — Envelope-controlled filter: louder hits sound brighter.

Each percussive transient in claps.wav simultaneously:
  • retriggers choir.wav playback via TriggerRestartPE, and
  • modulates a low-pass filter cutoff so loud hits sound bright and
    soft hits sound dark.

Signal chain (Demo 3):

    claps.wav ──► WavReaderPE
                      │
                 EnvelopePE
                      │
                   CachePE ──────────────────────────────────────────────┐
                      │                                             TransformPE
               SignalToGatePE                                       (env → Hz)
                  CachePE ─────────────────────────────────┐             │
                      │                                 (gain)           │
               GateToTriggerPE   choir.wav                 │             │
                      │              └──► WavReaderPE      │             │
                      └──────────────► TriggerRestartPE    │             │
                                               └───────► GainPE          │
                                                              │          │
                                                       BiquadPE(LOWPASS)◄┘
                                                              │
                                                        GainPE(0.5)
                                                              │
                                                    CropPE clip=False (+tail)
                                                              │
                                                         pg.play()

Usage:
    uv run python examples/envelope_filter_eg.py        # interactive menu
    uv run python examples/envelope_filter_eg.py 1      # claps only
    uv run python examples/envelope_filter_eg.py 2      # choir only
    uv run python examples/envelope_filter_eg.py 3      # full effect
    uv run python examples/envelope_filter_eg.py a      # all demos

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import pygmu2 as pg
from pathlib import Path
from examples_helper import run_demos

SRATE = 44100
pg.set_sample_rate(SRATE)

AUDIO = Path(__file__).parent / "audio"

CLAPS_FILE = AUDIO / "claps.wav"
CHOIR_FILE = AUDIO / "choir.wav"

RELEASE = 0.2  # seconds — envelope release time
LOW_FREQ = 200.0  # Hz — dark (soft clap / envelope near 0)
HIGH_FREQ = 4000.0  # Hz — bright (loud clap / envelope near 1)


# ── Demos ─────────────────────────────────────────────────────────────────────


def demo_claps():
    """Play claps.wav unmodified."""
    pg.play(pg.WavReaderPE(str(CLAPS_FILE)))


def demo_choir():
    """Play choir.wav unmodified."""
    pg.play(pg.WavReaderPE(str(CHOIR_FILE)))


def demo_envelope_filter():
    """Clap-driven choir retrigger with envelope-tracked filter cutoff."""
    claps = pg.WavReaderPE(str(CLAPS_FILE))
    choir = pg.WavReaderPE(str(CHOIR_FILE))

    # Envelope follows clap energy.  CachePE lets it fan out to two branches
    # (SignalToGatePE and TransformPE) without the impure-multiple-sinks error.
    envelope = pg.CachePE(pg.EnvelopePE(claps, attack=0.002, release=RELEASE))

    # Trigger branch: retrigger choir on each rising edge of the gate.  CachePE
    # lets it fan out to two branches (GateToTriggerPE and GainPE).
    gate = pg.CachePE(
        pg.SignalToGatePE(envelope, low_threshold=0.02, high_threshold=0.05)
    )

    # Restart choir on each positive going edge of the gate singal
    retriggered = pg.TriggerRestartPE(pg.GateToTriggerPE(gate), choir)

    # Use the gate to turn on and off the choir
    gated = pg.GainPE(retriggered, gate)

    # Filter branch: cutoff tracks envelope → bright on loud hits, dark on soft
    freq_pe = pg.TransformPE(
        envelope,
        func=lambda x: LOW_FREQ + x * (HIGH_FREQ - LOW_FREQ),
        name="env_to_freq",
    )

    filtered = pg.BiquadPE(gated, frequency=freq_pe, q=3.0, mode=pg.BiquadMode.LOWPASS)

    # Extend by release tail so the last clap decays fully before the file ends
    claps_end = claps.extent().end
    release_samples = int((RELEASE + 0.4) * SRATE)

    output = pg.CropPE(
        pg.GainPE(filtered, 0.5),
        start=0,
        duration=claps_end + release_samples,
        clip=False,
    )

    pg.play(output)


# ── Entry point ───────────────────────────────────────────────────────────────

DEMOS = [
    ("Claps — source audio unmodified", demo_claps),
    ("Choir — source audio unmodified", demo_choir),
    ("Envelope filter: clap-driven choir retrigger + brightness", demo_envelope_filter),
]

README = """\
Envelope-following filter driven by percussive transients.

Clap hits from claps.wav retrigger choir.wav playback via TriggerRestartPE
and simultaneously modulate a low-pass filter cutoff: loud claps sound
bright, soft claps sound dark.  The signal chain uses EnvelopePE,
SignalToGatePE, GateToTriggerPE, and BiquadPE(LOWPASS).
"""

if __name__ == "__main__":
    run_demos(DEMOS, readme=README)
