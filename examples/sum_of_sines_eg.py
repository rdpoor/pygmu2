"""
Example: Demo sine summation for popular waveforms (square, triangle, saw, pulse)

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import pygmu2 as pg
from examples_helper import run_demos

pg.set_sample_rate(44100)

DURATION_SECONDS = 1.0
DURATION_SAMPLES = DURATION_SECONDS * pg.get_sample_rate()


def make_demo_eg(freq_amp_pairs, analytical_generator):
    # play analytical version for 1 second
    # short pause
    # build up sum of sines
    # hold last value
    # short pause
    # play analytical version again
    DURATION_SAMPLES * len(freq_amp_pairs)
    tones = []
    start_samp = 0

    # opening analytical tone
    tones.append(pg.CropPE(analytical_generator(), start_samp, DURATION_SAMPLES))
    # short silence
    start_samp += 2 * DURATION_SAMPLES
    # play each sinewave by itself as an arpeggio
    for freq, amp in freq_amp_pairs:
        tones.append(
            pg.CropPE(
                pg.SinePE(frequency=freq, amplitude=amp * 0.5),
                start_samp,
                DURATION_SAMPLES,
            )
        )
        start_samp += DURATION_SAMPLES
    # another short silence
    start_samp += 2 * DURATION_SAMPLES
    # play each sine wave additively
    end_samp = start_samp + DURATION_SAMPLES * (len(freq_amp_pairs) + 1)
    for freq, amp in freq_amp_pairs:
        tones.append(
            pg.CropPE(
                pg.SinePE(frequency=freq, amplitude=amp * 0.5),
                start_samp,
                end_samp - start_samp,
            )
        )
        start_samp += DURATION_SAMPLES
    # another short silence
    start_samp += DURATION_SAMPLES * 2
    # closing analytical tone
    tones.append(pg.CropPE(analytical_generator(), start_samp, DURATION_SAMPLES))
    return pg.MixPE(tones)


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────

F0 = 220
SQUARE_WAVE = [
    (F0 * 1.0, 1.0 / 1.0),
    (F0 * 3.0, 1.0 / 3.0),
    (F0 * 5.0, 1.0 / 5.0),
    (F0 * 7.0, 1.0 / 7.0),
    (F0 * 9.0, 1.0 / 9.0),
    (F0 * 11.0, 1.0 / 11.0),
]

TRIANGLE_WAVE = [
    (F0 * 1.0, 1.0 / 1.0),
    (F0 * 3.0, -1.0 / 3.0**2),
    (F0 * 5.0, 1.0 / 5.0**2),
    (F0 * 7.0, -1.0 / 7.0**2),
    (F0 * 9.0, 1.0 / 9.0**2),
    (F0 * 11.0, 1.0 / 11.0**2),
]

SAWTOOTH_WAVE = [
    (F0 * 1.0, 0.5 / 1.0),
    (F0 * 2.0, -0.5 / 2.0),
    (F0 * 3.0, 0.5 / 3.0),
    (F0 * 4.0, -0.5 / 4.0),
    (F0 * 5.0, 0.5 / 5.0),
    (F0 * 6.0, -0.5 / 6.0),
    (F0 * 7.0, 0.5 / 7.0),
    (F0 * 8.0, 0.5 / 8.0),
    (F0 * 9.0, 0.5 / 9.0),
    (F0 * 10.0, 0.5 / 10.0),
    (F0 * 11.0, 0.5 / 11.0),
]

PULSE_WAVE = [
    (F0 * 1.0, 0.2),
    (F0 * 2.0, -0.2),
    (F0 * 3.0, 0.2),
    (F0 * 4.0, -0.2),
    (F0 * 5.0, 0.2),
    (F0 * 6.0, -0.2),
    (F0 * 7.0, 0.2),
    (F0 * 8.0, -0.2),
    (F0 * 9.0, 0.2),
    (F0 * 10.0, -0.2),
    (F0 * 11.0, 0.2),
    (F0 * 12.0, -0.2),
    (F0 * 13.0, 0.2),
    (F0 * 14.0, -0.2),
]


def demo_square():
    """Approximate square waveform as a sum of sines"""

    def square_wave():
        return pg.GainPE(
            pg.AnalogOscPE(F0, duty_cycle=0.5, waveform="rectangle", antialias=False),
            0.5,
        )

    pg.play(make_demo_eg(SQUARE_WAVE, square_wave))


def demo_triangle():
    """Approximate triangle waveform as a sum of sines"""

    def triangle_wave():
        return pg.GainPE(
            pg.AnalogOscPE(F0, duty_cycle=0.5, waveform="sawtooth", antialias=False),
            0.5,
        )

    pg.play(make_demo_eg(TRIANGLE_WAVE, triangle_wave))


def demo_sawtooth():
    """Approximate sawtooth waveform as a sum of sines"""

    def sawtooth_wave():
        return pg.GainPE(
            pg.AnalogOscPE(F0, duty_cycle=0.0, waveform="sawtooth", antialias=False),
            0.5,
        )

    pg.play(make_demo_eg(SAWTOOTH_WAVE, sawtooth_wave))


def demo_pulse():
    """Approximate a pulse waveform as a sum of sines"""

    def pulse_wave():
        return pg.GainPE(
            pg.AnalogOscPE(F0, duty_cycle=0.05, waveform="rectangle", antialias=False),
            0.5,
        )

    pg.play(make_demo_eg(PULSE_WAVE, pulse_wave))


DEMOS = [
    ("Approximate square waveform as a sum of sines", demo_square),
    ("Approximate triangle waveform as asum of sines", demo_triangle),
    ("Approximate sawtooth waveform as a sum of sines", demo_sawtooth),
    ("Approximate a pulse waveform as a sum of sines", demo_pulse),
]

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demos(DEMOS)
