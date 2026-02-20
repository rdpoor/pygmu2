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
SRATE = 44100
pg.set_sample_rate(SRATE)

RANDOM_SEED = 123
PLAY_DUR = 4.0
SILENCE_DUR = 1.0

def crop_and_pad_pe(pe, play_dur, silence_dur):
    # crop PE to a finite length, followed by a silence
    cropped_pe = pg.CropPE(
        pe, 0, pg.seconds_to_samples(PLAY_DUR, SRATE))
    # pad with silence
    padded_pe = pg.SetExtentPE(
        cropped_pe, 0, pg.seconds_to_samples(PLAY_DUR + SILENCE_DUR, SRATE))
    return padded_pe

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# write your demos here, list demo names and demo functions in DEMOS

def demo_voice_count():
    print("Demo SuperSawPE with varying voice count")
    print("--------")
    for n_voices in range(1, 12):
        ss = pg.SuperSawPE(
            frequency=pg.pitch_to_freq(45),
            voices=n_voices,
            seed = RANDOM_SEED,
        )
        print(f"n voices = {n_voices}...")
        pg.play(crop_and_pad_pe(ss, PLAY_DUR, SILENCE_DUR))
    print(f"Done!")

def demo_mix_mode():
    print("Demo SuperSawPE with three mix modes")
    print("--------")
    for mix_mode in [pg.SuperSawPE.MIX_EQUAL, pg.SuperSawPE.MIX_LINEAR, pg.SuperSawPE.MIX_CENTER_HEAVY]:
        ss = pg.SuperSawPE(
            frequency=pg.pitch_to_freq(46),
            voices=7,
            mix_mode = mix_mode,
            seed = RANDOM_SEED,
        )
        print(f"mix_mode = {mix_mode}...")
        pg.play(crop_and_pad_pe(ss, PLAY_DUR, SILENCE_DUR))
    print(f"Done!")

def demo_detune_amounts():
    print("Demo SuperSawPE with varying degrees of detuning")
    print("--------")
    for detune_cents in [3, 12, 20, 30, 40]:
        ss = pg.SuperSawPE(
            frequency=pg.pitch_to_freq(47),
            voices=7,
            detune_cents = detune_cents,
            mix_mode = pg.SuperSawPE.MIX_CENTER_HEAVY,
            seed = RANDOM_SEED,
        )
        print(f"detune_cents = {detune_cents}...")
        pg.play(crop_and_pad_pe(ss, PLAY_DUR, SILENCE_DUR))
    print(f"Done!")

def demo_randomize_phase():
    print("Demo SuperSawPE with and without randomized initial phase")
    print("--------")
    for randomize_phase in [True, False]:
        ss = pg.SuperSawPE(
            frequency=pg.pitch_to_freq(48),
            voices=7,
            detune_cents = 20,
            mix_mode = pg.SuperSawPE.MIX_CENTER_HEAVY,
            randomize_phase = randomize_phase,
            seed = RANDOM_SEED,
        )
        print(f"randomize_phase = {randomize_phase}...")
        pg.play(crop_and_pad_pe(ss, PLAY_DUR, SILENCE_DUR))
    print(f"Done!")

DEMOS = {
    "Demo SuperSawPE with varying voice count": demo_voice_count,
    "Demo SuperSawPE with three mix modes": demo_mix_mode,
    "Demo SuperSawPE with varying degrees of detuning": demo_detune_amounts,
    "Demo SuperSawPE with and without randomized initial phase": demo_randomize_phase,
}

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    import sys

    def resolve_choice(choice: str):
        """
        Return (name, fn) on valid choice, (None, None) otherwise.
        """
        item_list = list(DEMOS.items())

        if choice.isdigit():
            # Numerical choice (one-based)
            idx = int(choice)
            if 1 <= idx <= len(item_list):
                return item_list[idx - 1]
            else:
                return None, None

        if choice in DEMOS:
            # String match
            return choice, DEMOS[choice]
        else:
            return None, None

    def print_menu():
        names = list(DEMOS.keys())
        print("Available demos:")
        for i, name in enumerate(names, start=1):
            print(f"  {i}: {name}")
        print("  ?: show list")
        print("  a: run all")
        print("  q: quit")

    def choose_and_play():
        """
        DEMOS: dict[demo_name, demo_function]
        Present list of demo names, call user's choice.  Loop until 'q'
        """
        while True:
            choice = input("Select demo (name or number): ").strip()
            if choice.lower() == "q":
                break
            if choice.lower() == "a":
                for fn in DEMOS.values():
                    fn()
                continue
            if choice == "?":
                print_menu()
                continue

            _name, fn = resolve_choice(choice)
            if fn is not None:
                fn()
            else:
                print(f"unrecognized choice {choice}, '?' to see choices")

    if len(sys.argv) > 1:
        # Command line choice: Run single demo (or 'a' for all') and quit
        choice = sys.argv[1].strip().lower()
        if choice == "a":
            for fn in DEMOS.values():
                fn()
            raise SystemExit(0)
        _name, fn = resolve_choice(choice)
        if fn is not None:
            fn()
        else:
            print(f"Invalid choice '{choice}'")
    else:
        # Enter interactive loop
        print_menu()
        choose_and_play()
