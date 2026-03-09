"""
Example: Ring Modulator — Sideband synthesis and vocal morphing

RingModulatorPE multiplies carrier × (modulator + bias), producing the classic
analog ring modulation effect found on Moog, Roland, Doepfer, and Buchla systems.

  wet    = carrier × (modulator + bias)
  output = (1 − mix) × carrier + mix × wet

  bias=0, mix=1  → pure ring modulation (sidebands only, carrier suppressed)
  bias=1, mix=1  → amplitude modulation (carrier present in output)
  mix=0          → dry carrier pass-through (unprocessed voice)

Signal routing used in all demos:
  carrier  = spoken voice (the signal being processed — mix=0 recovers it)
  modulator = sine wave  (the ring modulator oscillator)

The voice is stereo; the sine is mono and is automatically broadcast to both
channels inside RingModulatorPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

# Tips for using RingModulatorPE:
# - bias=0, mix=1: classic ring mod — metallic, robotic quality on voice
# - bias=1, mix=1: amplitude modulation — carrier pitch is preserved better
# - Modulator frequency shapes which sidebands appear — try 100–2000 Hz on voice
# - Dynamic bias or mix PEs let you morph continuously between modes
# - SinePE amplitude=1.0 gives full ring mod depth; lower values reduce depth

from pathlib import Path

import pygmu2 as pg
from examples_helper import pad_clip, run_demos

pg.set_sample_rate(44100)

AUDIO_DIR = Path(__file__).parent / "audio"
VOICE_FILE = AUDIO_DIR / "spoken_voice44.wav"

# Modulator frequency used for the bias, mix, and dynamic demos
MOD_FREQ = 440.0  # Hz

# Probe the voice file once at import time to get its length
VOICE_SAMPLES = pg.WavReaderPE(str(VOICE_FILE)).extent().end  # 181753 @ 44100


def make_voice():
    """Return a fresh WavReaderPE for the voice file."""
    return pg.WavReaderPE(str(VOICE_FILE))


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────

def demo_dry_voice():
    """Play the unprocessed voice for reference."""
    pg.play(pg.GainPE(make_voice(), gain=2.14))


def demo_modulator_frequency():
    """Ring-mod voice with a sine at 100, 200, 440, and 800 Hz.

    bias=0, mix=1 — pure ring modulation at four modulator frequencies.
    Lower frequencies add a subtle metallic colouring while preserving some
    intelligibility.  Higher frequencies create the classic robotic or bell-like
    character.  Ring modulation output contains sum and difference sidebands
    (voice_freq ± mod_freq) with the original voice frequencies suppressed.
    """
    for freq in [100, 200, 440, 800]:
        voice = make_voice()
        sine = pg.SinePE(frequency=float(freq), amplitude=1.0)
        rm = pg.RingModulatorPE(voice, sine, bias=0.0, mix=1.0)
        print(f"  modulator = {freq} Hz ...", flush=True)
        pg.play(pg.GainPE(pad_clip(rm), gain=2.15))
    print("Done!")


def demo_static_bias():
    """Step bias through 0.0, 0.5, and 1.0 with mix=1.

    Increasing bias injects a DC offset into the modulator before
    multiplication, which progressively re-introduces the carrier frequencies:

      bias=0.0  → pure ring modulation (carrier suppressed, sidebands only)
      bias=0.5  → partial amplitude modulation (some original frequencies present)
      bias=1.0  → amplitude modulation (full carrier — more like a tremolo)
    """
    for bias in [0.0, 0.5, 1.0]:
        voice = make_voice()
        sine = pg.SinePE(frequency=MOD_FREQ, amplitude=1.0)
        rm = pg.RingModulatorPE(voice, sine, bias=bias, mix=1.0)
        print(f"  bias = {bias} ...", flush=True)
        pg.play(pg.GainPE(pad_clip(rm), gain=1.18))
    print("Done!")


def demo_static_mix():
    """Step mix through 0.0, 0.5, and 1.0 with bias=0.

    mix controls the dry/wet balance between the original voice and the
    ring-modulated signal:

      mix=0.0  → dry voice only (ring mod inaudible)
      mix=0.5  → equal blend of dry voice and ring-modulated signal
      mix=1.0  → pure ring modulation, no dry signal
    """
    for mix in [0.0, 0.5, 1.0]:
        voice = make_voice()
        sine = pg.SinePE(frequency=MOD_FREQ, amplitude=1.0)
        rm = pg.RingModulatorPE(voice, sine, bias=0.0, mix=mix)
        print(f"  mix = {mix} ...", flush=True)
        pg.play(pg.GainPE(pad_clip(rm), gain=2.15))
    print("Done!")


def demo_dynamic_mix():
    """Ramp mix from 0 → 1 over the full voice duration.

    A PiecewisePE ramp controls mix in real time.  The voice starts completely
    dry, then the ring modulation gradually takes over until the end of the
    clip is fully ring-modulated.  bias=0 (pure ring mod throughout).
    """
    mix_ramp = pg.PiecewisePE([(0, 0.0), (VOICE_SAMPLES, 1.0)])
    voice = make_voice()
    sine = pg.SinePE(frequency=MOD_FREQ, amplitude=1.0)
    rm = pg.RingModulatorPE(voice, sine, bias=0.0, mix=mix_ramp)
    pg.play(pg.GainPE(rm, gain=2.36))
    print("Done!")


def demo_dynamic_bias():
    """Ramp bias from 0 → 1 over the full voice duration.

    A PiecewisePE ramp controls bias in real time.  The voice starts as pure
    ring modulation (carrier suppressed), then the original carrier frequencies
    gradually leak back in as bias rises, ending in amplitude modulation.
    mix=1 throughout so the full wet signal is heard.
    """
    bias_ramp = pg.PiecewisePE([(0, 0.0), (VOICE_SAMPLES, 1.0)])
    voice = make_voice()
    sine = pg.SinePE(frequency=MOD_FREQ, amplitude=1.0)
    rm = pg.RingModulatorPE(voice, sine, bias=bias_ramp, mix=1.0)
    pg.play(pg.GainPE(rm, gain=1.23))
    print("Done!")


DEMOS = [
    ("Dry voice (reference)", demo_dry_voice),
    ("Modulator frequency sweep — pure ring mod (bias=0, mix=1)", demo_modulator_frequency),
    ("Static bias steps  (mix=1.0  |  modulator=440 Hz)", demo_static_bias),
    ("Static mix steps  (bias=0.0  |  modulator=440 Hz)", demo_static_mix),
    ("Dynamic mix ramp — dry -> full ring mod", demo_dynamic_mix),
    ("Dynamic bias ramp — ring mod -> amplitude modulation", demo_dynamic_bias),
]

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

README = """\
RingModulatorPE — sideband synthesis and vocal morphing.

Multiplies carrier x (modulator + bias) for classic analog ring modulation.
bias=0 gives pure ring mod (sidebands only); bias=1 gives amplitude
modulation (carrier preserved).  mix controls dry/wet balance.

Demos sweep modulator frequency, step through bias and mix values, and
ramp bias and mix dynamically over time using PiecewisePE.
"""

if __name__ == "__main__":
    run_demos(DEMOS, readme=README)
