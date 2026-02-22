"""
Example: Super Saw - Rich, detuned unison oscillator

Inspired by the Roland JP-8000's "Supersaw" waveform, SuperSawPE creates
multiple slightly-detuned sawtooth oscillators mixed together, a sonic staple of
trance, EDM, and synth pads.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

# Tips for using SuperSawPE:
# - 7 voices with 15-25 cents detune is a good starting point
# - Use a lowpass filter to tame the high frequencies
# - Stack multiple notes for rich chord pads
# - 'center_heavy' mix mode gives a more focused sound
# - Higher voice counts (9-11) add richness but cost more CPU


from pathlib import Path

import pygmu2 as pg
from examples_helper import s, pad_clip, run_demos

pg.set_sample_rate(44100)

RANDOM_SEED = 123
PLAY_DUR = 4.0
SILENCE_DUR = 1.0

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# write your demos here, list demo names and demo functions in DEMOS

def demo_voice_count():
    print("Demo SuperSawPE with varying voice count")
    print("--------")
    for n_voices in range(1, 12):
        ss = pg.SuperSawPE(
            frequency=pg.pitch_to_freq(45),
            amplitude=0.5,
            voices=n_voices,
            seed = RANDOM_SEED,
        )
        print(f"n voices = {n_voices}...")
        pg.play(pad_clip(pg.CropPE(ss, 0, s(PLAY_DUR)), silence_secs=SILENCE_DUR))
    print(f"Done!")

def demo_mix_mode():
    print("Demo SuperSawPE with three mix modes")
    print("--------")
    for mix_mode in [pg.SuperSawPE.MIX_EQUAL, pg.SuperSawPE.MIX_LINEAR, pg.SuperSawPE.MIX_CENTER_HEAVY]:
        ss = pg.SuperSawPE(
            frequency=pg.pitch_to_freq(46),
            amplitude=0.5,
            voices=7,
            mix_mode = mix_mode,
            seed = RANDOM_SEED,
        )
        print(f"mix_mode = {mix_mode}...")
        pg.play(pad_clip(pg.CropPE(ss, 0, s(PLAY_DUR)), silence_secs=SILENCE_DUR))
    print(f"Done!")

def demo_detune_amounts():
    print("Demo SuperSawPE with varying degrees of detuning")
    print("--------")
    for detune_cents in [3, 12, 20, 30, 40]:
        ss = pg.SuperSawPE(
            frequency=pg.pitch_to_freq(47),
            amplitude=0.5,
            voices=7,
            detune_cents = detune_cents,
            mix_mode = pg.SuperSawPE.MIX_CENTER_HEAVY,
            seed = RANDOM_SEED,
        )
        print(f"detune_cents = {detune_cents}...")
        pg.play(pad_clip(pg.CropPE(ss, 0, s(PLAY_DUR)), silence_secs=SILENCE_DUR))
    print(f"Done!")

def demo_randomize_phase():
    print("Demo SuperSawPE with and without randomized initial phase")
    print("--------")
    for randomize_phase in [True, False]:
        ss = pg.SuperSawPE(
            frequency=pg.pitch_to_freq(48),
            amplitude=0.5,
            voices=7,
            detune_cents = 20,
            mix_mode = pg.SuperSawPE.MIX_CENTER_HEAVY,
            randomize_phase = randomize_phase,
            seed = RANDOM_SEED,
        )
        print(f"randomize_phase = {randomize_phase}...")
        pg.play(pad_clip(pg.CropPE(ss, 0, s(PLAY_DUR)), silence_secs=SILENCE_DUR))
    print(f"Done!")

DEMOS = [
    ("Demo SuperSawPE with varying voice count", demo_voice_count),
    ("Demo SuperSawPE with three mix modes", demo_mix_mode),
    ("Demo SuperSawPE with varying degrees of detuning", demo_detune_amounts),
    ("Demo SuperSawPE with and without randomized initial phase", demo_randomize_phase),
]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    run_demos(DEMOS)
