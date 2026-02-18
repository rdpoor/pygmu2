#!/usr/bin/env python3
"""
Example 31: TriggerPE with swept LFO

Uses choir.wav as the signal and a sine wave sweeping from 1 Hz to 8 Hz
over 8 seconds as the trigger. Demos: original (ungated), ONE_SHOT,
GATED, and RETRIGGER modes.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path

from pygmu2 import (
    CropPE,
    Extent,
    PiecewisePE,
    SinePE,
    TriggerPE,
    TriggerMode,
    WavReaderPE,
)
import pygmu2 as pg
pg.set_sample_rate(44100)


AUDIO_DIR = Path(__file__).parent / "audio"
WAV_FILE = AUDIO_DIR / "choir.wav"

DURATION_SECONDS = 8.0


def _make_trigger_sweep(sample_rate: int, duration_samples: int):
    """Swept sine 1 Hz -> 8 Hz over duration_samples (trigger uses channel 0, >0 = on)."""
    freq_ramp = PiecewisePE([(0, 1.0), (duration_samples, 8.0)])
    return SinePE(frequency=freq_ramp, amplitude=1.0)


def demo_original():
    """Play choir without trigger (original signal)."""
    print("=== Trigger Example 31: Original (ungated choir) ===")
    source = WavReaderPE(str(WAV_FILE))
    sample_rate = source.file_sample_rate or 44100
    duration_samples = int(DURATION_SECONDS * sample_rate)
    pg.play(CropPE(source, 0, duration_samples), sample_rate)


def demo_one_shot():
    """ONE_SHOT: first positive edge starts choir; it then runs indefinitely."""
    print("=== Trigger Example 31: ONE_SHOT (choir starts on first trigger) ===")
    source = WavReaderPE(str(WAV_FILE))
    sample_rate = source.file_sample_rate or 44100
    duration_samples = int(DURATION_SECONDS * sample_rate)
    trigger = _make_trigger_sweep(sample_rate, duration_samples)
    triggered = TriggerPE(source, trigger, trigger_mode=TriggerMode.ONE_SHOT)
    pg.play(CropPE(triggered, 0, duration_samples), sample_rate)


def demo_gated():
    """GATED: choir plays only while trigger > 0; one gate per session, no retrigger."""
    print("=== Trigger Example 31: GATED (choir only when trigger > 0) ===")
    source = WavReaderPE(str(WAV_FILE))
    sample_rate = source.file_sample_rate or 44100
    duration_samples = int(DURATION_SECONDS * sample_rate)
    trigger = _make_trigger_sweep(sample_rate, duration_samples)
    triggered = TriggerPE(source, trigger, trigger_mode=TriggerMode.GATED)
    pg.play(CropPE(triggered, 0, duration_samples), sample_rate)


def demo_retrigger():
    """RETRIGGER: each positive edge restarts choir; stops when trigger <= 0."""
    print("=== Trigger Example 31: RETRIGGER (choir restarts on every trigger edge) ===")
    source = WavReaderPE(str(WAV_FILE))
    sample_rate = source.file_sample_rate or 44100
    duration_samples = int(DURATION_SECONDS * sample_rate)
    trigger = _make_trigger_sweep(sample_rate, duration_samples)
    triggered = TriggerPE(source, trigger, trigger_mode=TriggerMode.RETRIGGER)
    pg.play(CropPE(triggered, 0, duration_samples), sample_rate)


def demo_all():
    demo_original()
    demo_one_shot()
    demo_gated()
    demo_retrigger()


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "Original (ungated choir)", demo_original),
        ("2", "ONE_SHOT", demo_one_shot),
        ("3", "GATED", demo_gated),
        ("4", "RETRIGGER", demo_retrigger),
        ("a", "All demos", demo_all),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("Example 31: TriggerPE with swept LFO (choir.wav, 1–8 Hz over 8 s)")
        print("---------------------------------------------------------------")
        for key, name, _ in demos:
            print(f"  {key}: {name}")
        print()
        choice = input("Choice (1-4 or 'a'): ").strip().lower()

    for key, _name, fn in demos:
        if key == choice:
            fn()
            break
    else:
        print("Invalid choice.")
