#!/usr/bin/env python3
"""
Function generator outputs for teaching.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import math

from pygmu2 import (
    CropPE,
    FunctionGenPE,
    GainPE,
    SinePE,
    pitch_to_freq,
)
import pygmu2 as pg
from examples_helper import run_demos

pg.AudioRenderer.list_devices()

pg.set_sample_rate(44100)
SAMPLE_RATE = 44100


def s2s(seconds):
    return int(round(seconds * SAMPLE_RATE))


EXAMPLE_FREQUENCY = 220
EXAMPLE_DURATION = s2s(10)

# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_sine_wave():
    print("=== Sine Wave, 220 Hz ===")
    tone = SinePE(frequency=EXAMPLE_FREQUENCY, amplitude=0.5)
    pg.play(CropPE(tone, 0, EXAMPLE_DURATION))


def demo_sine_wave_440():
    print("=== Sine Wave, 440 Hz ===")
    tone = SinePE(frequency=EXAMPLE_FREQUENCY * 2, amplitude=0.5)
    pg.play(CropPE(tone, 0, EXAMPLE_DURATION))


def demo_sine_wave_half_amp():
    print("=== Sine Wave, 220 Hz, half amplitude ===")
    tone = SinePE(frequency=EXAMPLE_FREQUENCY, amplitude=0.25)
    pg.play(CropPE(tone, 0, EXAMPLE_DURATION))


def demo_triangle_wave():
    print("=== Triangle Wave, 220 Hz ===")
    tone = FunctionGenPE(
        frequency=EXAMPLE_FREQUENCY, waveform="sawtooth", duty_cycle=0.5
    )
    pg.play(CropPE(GainPE(tone, gain=0.5), 0, EXAMPLE_DURATION))


def demo_square_wave():
    print("=== Square Wave, 220 Hz ===")
    tone = FunctionGenPE(
        frequency=EXAMPLE_FREQUENCY, waveform="rectangle", duty_cycle=0.5
    )
    pg.play(CropPE(GainPE(tone, gain=0.5), 0, EXAMPLE_DURATION))


def demo_sawtooth_wave():
    print("=== Sawtooth Wave, 220 Hz ===")
    tone = FunctionGenPE(
        frequency=EXAMPLE_FREQUENCY, waveform="sawtooth", duty_cycle=0.0
    )
    pg.play(CropPE(GainPE(tone, gain=0.5), 0, EXAMPLE_DURATION))


def demo_pulse_wave():
    print("=== Pulse Wave, 220 Hz, 5% ===")
    tone = FunctionGenPE(
        frequency=EXAMPLE_FREQUENCY, waveform="rectangle", duty_cycle=0.05
    )
    pg.play(CropPE(GainPE(tone, gain=0.5), 0, EXAMPLE_DURATION))


def set_headphone_output():
    tone = SinePE(frequency=EXAMPLE_FREQUENCY, amplitude=0.5)
    pg.play(CropPE(tone, 0, s2s(0.5)), device=0)


def set_macbook_output():
    tone = SinePE(frequency=EXAMPLE_FREQUENCY, amplitude=0.5)
    pg.play(CropPE(tone, 0, s2s(0.5)), device=None)


# ---------------------------------------------------------------------------
# Demo registry
# ---------------------------------------------------------------------------

DEMOS = [
    ("=== Sine Wave, 220 Hz ===", demo_sine_wave),
    ("=== Sine Wave, 440 Hz ===", demo_sine_wave_440),
    ("=== Sine Wave, 220 Hz ===", demo_sine_wave_half_amp),
    ("=== Triangle Wave, 220 Hz ===", demo_triangle_wave),
    ("=== Square Wave, 220 Hz ===", demo_square_wave),
    ("=== Sawtooth Wave, 220 Hz ===", demo_sawtooth_wave),
    ("=== Pulse Wave, 220 Hz, 5% ===", demo_pulse_wave),
    ("Set Headpohone Output", set_headphone_output),
    ("Set MacBook Output", set_macbook_output),
]

if __name__ == "__main__":
    run_demos(DEMOS)
