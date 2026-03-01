"""
mag_freq_eg.py — FFT-domain magnitude and phase manipulation via MagFreqPE.

Demonstrates what happens when you modify magnitudes and/or phases of an FFT
before converting back to the time domain.

Source file is a drum beat at 99 BPM (1.65 beats/s, ~26727 samples/beat).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import pygmu2 as pg
from examples_helper import run_demos

SAMPLE_RATE = 44100
pg.set_sample_rate(SAMPLE_RATE)

AUDIO_DIR = Path(__file__).parent / "audio"
DRUM_WAV = pg.WavReaderPE(str(AUDIO_DIR / "LOA_99_Drums_DoubleDown.wav"))


# ── Helper ────────────────────────────────────────────────────────────────────

def negate_phases_fn(f_lo: float, f_hi: float):
    """Return a mangler that negates phases between f_lo and f_hi Hz."""
    def mangler_fn(magnitudes, phases):
        sr = pg.get_sample_rate()
        fft_len = len(magnitudes)
        bin_lo = int(round(f_lo * fft_len / sr))
        bin_hi = int(round(f_hi * fft_len / sr))
        print(f"freq[{f_lo}, {f_hi}] => bin[{bin_lo}, {bin_hi}]")
        phases[bin_lo:bin_hi] = -phases[bin_lo:bin_hi]
        return magnitudes, phases
    return mangler_fn


# ── Demos ─────────────────────────────────────────────────────────────────────

def demo_drums_dry():
    pg.play(pg.GainPE(DRUM_WAV, gain=0.71), SAMPLE_RATE)

def demo_reverse_low_frequencies():
    mangled = pg.MagFreqPE(DRUM_WAV, negate_phases_fn(0, 850), normalize_peak=0.33)
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_reverse_high_frequencies():
    mangled = pg.MagFreqPE(DRUM_WAV, negate_phases_fn(850, 20000), normalize_peak=0.33)
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_reverse_mid_frequencies():
    mangled = pg.MagFreqPE(DRUM_WAV, negate_phases_fn(100, 800), normalize_peak=0.33)
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_shift_increasing_frequencies():
    def mangler(magnitudes, phases):
        for i in range(len(phases) - 1):
            shift = 0.3 * float(i) / len(phases)
            phases[i + 1] += phases[i + 1] * shift  # don't touch DC
        return magnitudes, phases
    mangled = pg.MagFreqPE(DRUM_WAV, mangler, normalize_peak=0.33)
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_shift_decreasing_frequencies():
    def mangler(magnitudes, phases):
        for i in range(len(phases) - 1):
            shift = 0.3 * float(i) / len(phases)
            phases[i + 1] += phases[i + 1] * (1.0 - shift)  # don't touch DC
        return magnitudes, phases
    mangled = pg.MagFreqPE(DRUM_WAV, mangler, normalize_peak=0.33)
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_tralfam():
    def mangler(magnitudes, phases):
        rng = np.random.default_rng()
        phases = rng.random((len(phases), 2)) * 2.0 * np.pi
        return magnitudes, phases
    mangled = pg.MagFreqPE(DRUM_WAV, mangler, normalize_peak=0.33)
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_alternate_phases():
    def mangler(magnitudes, phases):
        for i in range(100, len(phases) - 1, 2):
            phases[i], phases[i + 1] = phases[i + 1], phases[i]
        return magnitudes, phases
    mangled = pg.MagFreqPE(DRUM_WAV, mangler, normalize_peak=0.33)
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_negate_every_other_phase():
    def mangler(magnitudes, phases):
        phases[10::2] *= -1
        return magnitudes, phases
    mangled = pg.MagFreqPE(DRUM_WAV, mangler, normalize_peak=0.33)
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)


DEMOS = [
    ("Dry drums",                                    demo_drums_dry),
    ("Reverse low frequency phases (0–850 Hz)",      demo_reverse_low_frequencies),
    ("Reverse high frequency phases (850–20k Hz)",   demo_reverse_high_frequencies),
    ("Reverse mid frequency phases (100–800 Hz)",    demo_reverse_mid_frequencies),
    ("Progressively phase-shift higher frequencies", demo_shift_increasing_frequencies),
    ("Progressively phase-shift lower frequencies",  demo_shift_decreasing_frequencies),
    ("Randomise phases (tralfam)",                   demo_tralfam),
    ("Alternate adjacent phases",                    demo_alternate_phases),
    ("Negate every other phase",                     demo_negate_every_other_phase),
]

if __name__ == "__main__":
    run_demos(DEMOS)
