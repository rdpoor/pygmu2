"""
I'm Lucky -- a pygmu2 re-creation of Thomas Dolby's synth part in Joan
Armatrading's "I'm Lucky".  See https://www.youtube.com/watch?v=HxQSfuGTCdM

The goal of this was to put pygmu2 through its paces to validate the design
of the modules, etc.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pygmu2 as pg
from pathlib import Path

SRATE = 44100
pg.set_sample_rate(SRATE)

# ===== MIDI notes
Cm1, Dfm1, Dm1, Efm1, Em1, Fm1, Gfm1, Gm1, Afm1, Am1, Bfm1, Bm1 = range(0, 12)
C0, Df0, D0, Ef0, E0, F0, Gf0, G0, Af0, A0, Bf0, B0 = range(12, 24)
C1, Df1, D1, Ef1, E1, F1, Gf1, G1, Af1, A1, Bf1, B1 = range(24, 36)
C2, Df2, D2, Ef2, E2, F2, Gf2, G2, Af2, A2, Bf2, B2 = range(36, 48)
C3, Df3, D3, Ef3, E3, F3, Gf3, G3, Af3, A3, Bf3, B3 = range(48, 60)
C4, Df4, D4, Ef4, E4, F4, Gf4, G4, Af4, A4, Bf4, B4 = range(60, 72)
C5, Df5, D5, Ef5, E5, F5, Gf5, G5, Af5, A5, Bf5, B5 = range(72, 84)
C6, Df6, D6, Ef6, E6, F6, Gf6, G6, Af6, A6, Bf6, B6 = range(84, 96)

# Sharp equivalents
Asm1, Csm1, Dsm1, Fsm1, Gsm1 = [Bfm1, Dfm1, Efm1, Gfm1, Afm1]
As0, Cs0, Ds0, Fs0, Gs0 = [Bf0, Df0, Ef0, Gf0, Af0]
As1, Cs1, Ds1, Fs1, Gs1 = [Bf1, Df1, Ef1, Gf1, Af1]
As2, Cs2, Ds2, Fs2, Gs2 = [Bf2, Df2, Ef2, Gf2, Af2]
As3, Cs3, Ds3, Fs3, Gs3 = [Bf3, Df3, Ef3, Gf3, Af3]
As4, Cs4, Ds4, Fs4, Gs4 = [Bf4, Df4, Ef4, Gf4, Af4]
As5, Cs5, Ds5, Fs5, Gs5 = [Bf5, Df5, Ef5, Gf5, Af5]
As6, Cs6, Ds6, Fs6, Gs6 = [Bf6, Df6, Ef6, Gf6, Af6]

REST = -1  # an out-of-band note value

# ===== Durations, expressed as beats
WHOLE = 4.0
HALF = 2.0
QUARTER = 1.0
EIGHTH = 0.5
SIXTEENTH = 0.25
THIRTY_SECOND = 0.125
DOTTED = 1.5  # multiplicative modifier
TRIPLET = 2.0 / 3.0  # multiplicative modifier

# ===== Articulation, modifies durations

LEGATO = 1.2  # mild overlap
CONNECTED = 1.0  # notes directly abut one another, no overlap
DETACHED = 0.7  # slight space between notes
STACCATO = 0.5  # shortened notes

BPM = 114


def _b2sam(beat):
    """convert beat to samples"""
    seconds = beat * 60.0 / BPM
    return _sec2sam(seconds)


def _sec2sam(seconds):
    """convert seconds to samples"""
    return int(round(seconds * SRATE))


PORTAMENTO_SECONDS = 0.05
PORTAMENTO_SAMPLES = _sec2sam(PORTAMENTO_SECONDS)

IR_PATH = Path(__file__).parent / "audio" / "plate_ir44.wav"
IR = pg.WavReaderPE(str(IR_PATH))

im_lucky_synth = [
    # pitch, duration, articulation
    (Bf1, QUARTER, DETACHED),
    (Bf2, QUARTER, DETACHED),
    (Ef2, QUARTER, DETACHED),
    (F2, EIGHTH, DETACHED),
    (Fs2, EIGHTH, DETACHED),
    (Bf1, QUARTER, DETACHED),
    (Bf2, HALF * DOTTED, DETACHED),
    (Bf1, QUARTER, DETACHED),
    (Bf2, QUARTER, DETACHED),
    (Ef2, QUARTER, DETACHED),
    (F2, EIGHTH, DETACHED),
    (Fs2, EIGHTH, DETACHED),
    (Bf1, QUARTER, DETACHED),
    (Bf2, QUARTER, DETACHED),
    (Ef4, EIGHTH, DETACHED),
    (D4, EIGHTH, DETACHED),
    (Ef4, EIGHTH, DETACHED),
    (F4, EIGHTH + HALF, DETACHED),
    (Ef2, QUARTER, DETACHED),
    (F2, QUARTER, DETACHED),
    (Bf1, HALF * DOTTED, DETACHED),
    (F4, QUARTER, DETACHED),
    (G4, QUARTER, DETACHED),
    (F4, HALF * DOTTED - EIGHTH, DETACHED),
    (Bf1, EIGHTH + WHOLE, DETACHED),
    (Bf4, QUARTER * DOTTED, DETACHED),
    (A4, QUARTER, DETACHED),
    (F4, EIGHTH, DETACHED),
    (D4, QUARTER, DETACHED),
    (G4, QUARTER * DOTTED, DETACHED),
    (F4, QUARTER, DETACHED),
    (D4, EIGHTH, DETACHED),
    (Bf3, QUARTER, DETACHED),
    (Bf1, QUARTER, DETACHED),
    (Bf2, QUARTER, DETACHED),
    (Ef2, QUARTER, DETACHED),
    (F2, EIGHTH, DETACHED),
    (Fs2, EIGHTH, DETACHED),
    (Bf1, QUARTER, DETACHED),
    (Bf2, HALF * DOTTED, DETACHED),
    (Bf1, QUARTER, DETACHED),
    (Bf2, QUARTER, DETACHED),
    (Ef2, QUARTER, DETACHED),
    (F2, EIGHTH, DETACHED),
    (Fs2, EIGHTH, DETACHED),
    (Bf1, WHOLE, DETACHED),
]

# Synth is monophonic (dual oscillators at octave).  Use a single AdsrGatedPE
# to generate envelope for the gain of two SuperSaw oscillators


# Build a list of (start, duration) pairs, used to trigger the synth's ADSR
beat_points = []
beat = 0
for pitch, duration, articulation in im_lucky_synth:
    if pitch != REST:
        beat_points.append((_b2sam(beat), _b2sam(duration * articulation)))
    beat += duration

# Build a list of (start, pitch) pairs, used to control the pitch of
# the synth.  Includes portamento (glide) between pitches.
pitch_points = []
beat = 0
prev_pitch = None

for pitch, duration, articulation in im_lucky_synth:
    if pitch != REST:
        pitch = pitch - 0.48  # Match the pitch of Thomas Dolby's synth...
        # Apply portamento to all notes except for the first:
        if prev_pitch is None:
            # first point starts at the target pitch...
            pitch_points.append((_b2sam(beat), pitch))
        else:
            # subsequent points: gliss from previous pitch to current pitch
            pitch_points.append((_b2sam(beat), prev_pitch))
            pitch_points.append((_b2sam(beat) + PORTAMENTO_SAMPLES, pitch))
        prev_pitch = pitch
        # Now hold the pitch flat for the duration of the note
        pitch_points.append((_b2sam(beat + duration), pitch))
    beat += duration

# print(f'beat_points = {beat_points}')
# print(f'pitch_points = {pitch_points}')
gates = pg.ScheduledGatePE(beat_points)

# Create a stream of pitch values.  Ues a linear ramp to transition between
# pitches (for portamento), appropriate since we're using pitches and not
# frequencies.
pitches = pg.PiecewisePE(pitch_points)

# pitch_points => TransformPE => SuperSaw.freq => GainPE.source
#                     gate_signals => adsrGate => GainPE.gain


def make_synth(gates, pitches):
    # Generate ADSR ramps
    adsr = pg.CachePE(
        pg.AdsrGatedPE(
            gates,
            attack_time=0.02,
            decay_time=0.05,
            sustain_level=0.75,
            release_time=0.5,
        )
    )

    # Mix two SuperSaws running an octave apart
    f_lo = pg.CachePE(pg.TransformPE(pitches, func=pg.pitch_to_freq))
    synth_lo = pg.SuperSawPE(frequency=f_lo, detune_cents=8)
    f_hi = pg.TransformPE(f_lo, func=lambda x: x * 2)
    synth_hi = pg.SuperSawPE(frequency=f_hi, detune_cents=8)
    synth = pg.MixPE(synth_lo, synth_hi)

    # Apply ADSR to the mixed synths
    mix = pg.GainPE(synth, adsr)

    # Individual voiced stems (ADSR-shaped, same gain as the mix)
    stem_lo = pg.GainPE(pg.GainPE(synth_lo, adsr), 0.1)
    stem_hi = pg.GainPE(pg.GainPE(synth_hi, adsr), 0.1)

    # Not too loud, please...
    return pg.GainPE(mix, 0.1), stem_lo, stem_hi


# Now for the final mix...
duration = pitch_points[-1][0] + _sec2sam(5)  # account for reverb tail

mono_mix, stem_lo, stem_hi = make_synth(gates, pitches)
# a bit more bass...
eq_mix = pg.BiquadPE(mono_mix, frequency=200.0, q=0.707, mode=pg.BiquadMode.LOWSHELF)
# convert to stereo to match the plate reverb
stereo_mix = pg.SpatialPE(eq_mix, method=pg.SpatialAdapter(channels=2))
# add plate reverb
wet_mix = pg.ReverbPE(source=stereo_mix, ir=IR, mix=0.3)


def main():
    # Render supersaw stems to separate files
    stem_dir = Path(__file__).parent / "audio"
    cropped_lo = pg.CropPE(stem_lo, 0, duration)
    cropped_hi = pg.CropPE(stem_hi, 0, duration)
    lo_path = str(stem_dir / "im_lucky_supersaw_lo.wav")
    hi_path = str(stem_dir / "im_lucky_supersaw_hi.wav")
    print(f"Rendering supersaw lo stem → {lo_path}")
    pg.render_to_file(cropped_lo, lo_path)
    print(f"Rendering supersaw hi stem → {hi_path}")
    pg.render_to_file(cropped_hi, hi_path)
    print("Stems written.")

    # Render and play the full mix...
    pg.browse(wet_mix)


if __name__ == "__main__":
    main()
