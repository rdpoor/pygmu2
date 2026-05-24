"""
Example: Random Step 2 — Musical ratios with parallel stepped LPF and slew.

Two independent RandomStepPE sources modulate pitch and Moog ladder filter
cutoff simultaneously.  Pitch is quantized to just-intonation intervals above
a root, and filter cutoff is quantized to harmonics of that same root, so
every discrete state is acoustically related.

Each stepped control signal passes through a SlewLimiterPE that adds a brief
linear ramp between states — turning hard voltage jumps into short glides.
Pitch and filter slew at slightly different rates, so the glides feel
independent even when the underlying clocks happen to fire together.

A final demo replaces the Poisson pitch clock with a ScheduledGatePE running
on a strict rhythmic grid, while the filter stays random and rapid, creating
a melodic foreground against a timbral shimmer background.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import random as _random

import numpy as np
import pygmu2 as pg
from examples_helper import s, pad_clip, run_demos

pg.set_sample_rate(44100)

# Each run draws a fresh seed from OS entropy so the demos sound different
# every time.  Copy the printed value and paste it here to reproduce a result.
RUN_SEED: int = _random.randrange(2**31)

# ──────────────────────────────────────────────────────────────────────────────
# Musical constants
# ──────────────────────────────────────────────────────────────────────────────

ROOT_FREQ = 110.0  # A2 — shared harmonic root for pitch and filter

# Just-intonation ratios spanning two octaves of a Ptolemaic major scale.
# Using exact integer ratios keeps every pitch spectrally related to ROOT_FREQ.
PITCH_RATIOS = np.array([
    1 / 1,   # unison        (A2)
    9 / 8,   # major 2nd     (B2)
    5 / 4,   # major 3rd     (C#3)
    4 / 3,   # perfect 4th   (D3)
    3 / 2,   # perfect 5th   (E3)
    5 / 3,   # major 6th     (F#3)
    15 / 8,  # major 7th     (G#3)
    2 / 1,   # octave        (A3)
    9 / 4,   # major 9th     (B3)
    5 / 2,   # major 10th    (C#4)
    3 / 1,   # perfect 12th  (E4)
    4 / 1,   # double octave (A4)
])

# Cutoff frequencies drawn from the harmonic series above ROOT_FREQ.
# Choosing harmonics means the filter resonance peak always rings at a
# musically related partial.
FILTER_RATIOS = np.array([2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32], dtype=float)
FILTER_FREQS = ROOT_FREQ * FILTER_RATIOS  # ~220 Hz … ~3520 Hz


# ──────────────────────────────────────────────────────────────────────────────
# Mapping functions
# ──────────────────────────────────────────────────────────────────────────────

def random_to_pitch_freq(r: np.ndarray) -> np.ndarray:
    """Quantize 0..1 array to a just-intonation frequency above ROOT_FREQ."""
    n = len(PITCH_RATIOS)
    idx = np.clip(np.floor(r * n).astype(int), 0, n - 1)
    return ROOT_FREQ * PITCH_RATIOS[idx]


def random_to_filter_freq(r: np.ndarray) -> np.ndarray:
    """Quantize 0..1 array to a harmonic-series cutoff frequency."""
    n = len(FILTER_FREQS)
    idx = np.clip(np.floor(r * n).astype(int), 0, n - 1)
    return FILTER_FREQS[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Slew constants
# ──────────────────────────────────────────────────────────────────────────────

# Maximum rate of change applied by SlewLimiterPE (LINEAR mode, units = Hz/s).
# Pitch range is ~110–880 Hz; at 12 000 Hz/s a two-octave jump takes ~64 ms.
# Filter range is ~220–3520 Hz; at 48 000 Hz/s a full-range sweep takes ~69 ms.
PITCH_SLEW_RATE = 12_000.0   # Hz/s
FILTER_SLEW_RATE = 48_000.0  # Hz/s


# ──────────────────────────────────────────────────────────────────────────────
# Shared signal chain builder
# ──────────────────────────────────────────────────────────────────────────────

def build_chain(pitch_rate, filter_rate, resonance=0.5, duration=s(8),
                pitch_seed=None, filter_seed=None):
    """
    Construct a BlitSaw → LadderPE chain driven by two RandomStepPEs.

    Each control path is:
        RandomStepPE → TransformPE (quantise to Hz) → SlewLimiterPE → …

    The SlewLimiterPE adds a brief linear ramp between discrete states so that
    pitch and filter transitions glide rather than snap.

    Args:
        pitch_rate:  Rate (float or PE) for the pitch step generator.
        filter_rate: Rate (float or PE) for the filter cutoff step generator.
        resonance:   Ladder resonance (0..1); higher values give a singing peak.
        duration:    Length in samples.
        pitch_seed:  Optional RNG seed for reproducibility.
        filter_seed: Optional RNG seed for reproducibility.

    Returns:
        Cropped, gain-adjusted PE ready for play_offline.
    """
    pitch_step = pg.RandomStepPE(rate=pitch_rate, seed=pitch_seed)
    filter_step = pg.RandomStepPE(rate=filter_rate, seed=filter_seed)

    freq_stepped = pg.TransformPE(pitch_step, func=random_to_pitch_freq)
    cutoff_stepped = pg.TransformPE(filter_step, func=random_to_filter_freq)

    # Brief linear glide between discrete pitch / cutoff states
    freq = pg.SlewLimiterPE(freq_stepped, rate=PITCH_SLEW_RATE,
                            mode=pg.SlewMode.LINEAR)
    cutoff = pg.SlewLimiterPE(cutoff_stepped, rate=FILTER_SLEW_RATE,
                              mode=pg.SlewMode.LINEAR)

    osc = pg.BlitSawPE(frequency=freq)
    filtered = pg.LadderPE(osc, frequency=cutoff, resonance=resonance,
                           mode=pg.LadderMode.LP24)

    cropped = pg.CropPE(pg.GainPE(filtered, 0.12), 0, duration)
    return pg.GainPE(pad_clip(cropped), 2.5)


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────

def demo_lockstep():
    """Both clocks at the same rate — pitch and timbre always change together."""
    print("Demo: Lockstep — pitch rate = filter rate = 4 Hz")
    pg.play_offline(source=build_chain(
        pitch_rate=4.0, filter_rate=4.0,
        resonance=0.45, duration=s(8),
        pitch_seed=RUN_SEED + 1, filter_seed=RUN_SEED + 2,
    ))


def demo_polyrhythmic():
    """
    Pitch steps at a prime rate, filter at another prime — they drift in and
    out of alignment, creating evolving rhythmic combinations.
    """
    print("Demo: Polyrhythmic — pitch=3 Hz, filter=7 Hz")
    pg.play_offline(source=build_chain(
        pitch_rate=3.0, filter_rate=7.0,
        resonance=0.5, duration=s(12),
        pitch_seed=RUN_SEED + 3, filter_seed=RUN_SEED + 4,
    ))


def demo_high_resonance():
    """
    Resonance near self-oscillation makes the filter peak audible as a
    pitched tone — the harmonic cutoff values become almost melodic.
    """
    print("Demo: High resonance — filter sings the harmonic series")
    pg.play_offline(source=build_chain(
        pitch_rate=2.0, filter_rate=5.0,
        resonance=0.82, duration=s(12),
        pitch_seed=RUN_SEED + 9, filter_seed=RUN_SEED + 10,
    ))


def demo_rhythmic_gate():
    """
    Pitch follows a strict rhythmic grid; immediately after each pitch change
    the filter sweeps rapidly through a wide harmonic range, then both the
    step rate and the accessible frequency range decay across the beat.

    Two quantities are modulated by the same beat-phase envelope
    (1 at onset → 0 at end of beat):

        filter_rate  = MAX_RATE × env         — steps fire often, then rarely
        range_factor = MIN_RANGE + (1−MIN_RANGE) × env
                                              — full harmonic range → bottom quarter

    The range_factor is multiplied into the raw [0,1] CV from RandomStepPE
    before quantisation via RingModulatorPE.  Because this scales the *held*
    value as well, the filter drifts continuously toward darker harmonics
    between firings, reinforcing the sense of settling.

    Signal flow:

        gate (brief pulse at each beat)
          → AdsrGatedPE  (attack≈0, sustain=1, release=one beat) → env [1→0]
              ├─ × MAX_RATE                    → filter_rate
              └─ MIN_RANGE + (1−MIN_RANGE)×env → range_factor
                    ↕
        RandomStepPE(rate=filter_rate) × range_factor
          → quantise → SlewLimiterPE → LadderPE
    """
    print("Demo: Rhythmic gate — rapid wide filter at onset, settling to dark/slow")

    BPM = 120
    beat_samples = int(s(60.0 / BPM))   # samples per quarter note
    beat_secs = 60.0 / BPM              # seconds per quarter note
    gate_len = beat_samples // 8        # ~15 ms pulse to latch the noise value
    duration = s(12)

    # Extra beats ensure the ADSR release tail is defined past the crop window
    n_beats = int(duration // beat_samples) + 2
    notes = [(i * beat_samples, gate_len) for i in range(n_beats)]
    gate = pg.ScheduledGatePE(notes)

    # ── Pitch: sample noise at each beat onset ────────────────────────────────
    noise_cv = pg.NoisePE(min_value=0.0, max_value=1.0, seed=RUN_SEED + 11)
    pitch_cv = pg.TrackHoldPE(noise_cv, gate, initial_value=0.5)
    freq_stepped = pg.TransformPE(pitch_cv, func=random_to_pitch_freq)
    freq = pg.SlewLimiterPE(freq_stepped, rate=PITCH_SLEW_RATE,
                            mode=pg.SlewMode.LINEAR)

    # ── Beat-phase envelope: 1 at onset, linear decay to 0 over one beat ─────
    # CachePE allows the stateful AdsrGatedPE to fan out to two downstream PEs
    # (filter_rate and range_factor) without triggering the "multiple sinks"
    # validation error.  The cache returns the same Snippet for both consumers
    # within each render block, so the ADSR state advances exactly once.
    rate_env = pg.CachePE(pg.AdsrGatedPE(
        gate,
        attack_time=0.001,
        decay_time=0.001,   # sustain_level=1 → decay slope = 0, value OK
        sustain_level=1.0,
        release_time=beat_secs,
    ))

    # ── Filter rate: high at onset, falls to zero ─────────────────────────────
    MAX_FILTER_RATE = 24.0
    filter_rate = pg.TransformPE(
        rate_env,
        func=lambda x: MAX_FILTER_RATE * x,
    )

    # ── Range factor: 1.0 at onset, decays to MIN_RANGE ──────────────────────
    # Scales the raw [0,1] filter CV so that accessible harmonics progressively
    # compress toward the lower (darker) end of FILTER_FREQS.
    MIN_RANGE = 0.25   # bottom 25 % of the table ≈ 3 harmonics at beat end
    range_factor = pg.TransformPE(
        rate_env,
        func=lambda x: MIN_RANGE + (1.0 - MIN_RANGE) * x,
    )

    # ── Filter CV: step × range_factor ──────────────────────────────────────
    filter_step = pg.RandomStepPE(rate=filter_rate, seed=RUN_SEED + 12)
    # RingModulatorPE with bias=0, mix=1: output = filter_step × range_factor
    scaled_cv = pg.RingModulatorPE(filter_step, range_factor, bias=0.0, mix=1.0)
    cutoff_stepped = pg.TransformPE(scaled_cv, func=random_to_filter_freq)
    cutoff = pg.SlewLimiterPE(cutoff_stepped, rate=FILTER_SLEW_RATE,
                              mode=pg.SlewMode.LINEAR)

    osc = pg.BlitSawPE(frequency=freq)
    filtered = pg.LadderPE(osc, frequency=cutoff, resonance=0.55,
                           mode=pg.LadderMode.LP24)

    cropped = pg.CropPE(pg.GainPE(filtered, 0.12), 0, duration)
    pg.play_offline(source=pg.GainPE(pad_clip(cropped), 2.5))


DEMOS = [
    ("Lockstep (same rate)", demo_lockstep),
    ("Polyrhythmic (pitch=3, filter=7)", demo_polyrhythmic),
    ("High resonance (filter sings harmonics)", demo_high_resonance),
    ("Rhythmic gate (quarter-note pitch, rapid filter)", demo_rhythmic_gate),
]

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Run seed: {RUN_SEED}  (set RUN_SEED = {RUN_SEED} at the top of the file to reproduce this run)")
    run_demos(DEMOS)
