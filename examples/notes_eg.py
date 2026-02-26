"""
notes_eg.py — NotesPE: play MIDI notes from a source sample.

NotesPE pitch-shifts and time-positions a single source sound to play every
note in a MIDI file.  Each note is handled by a pure ResamplePE chain so any
number of simultaneous voices can overlap without interference.

Demo 1 — Supersaw source
    Plays the Earle of Salisbury using a 256 Hz blitsaw wave as the sound source.
    tempo=45 BPM, native_pitch=77 (the pitch the sine would represent at rate 1).

Demo 2 — WAV source (edit WAV_FILE to point at a real sample)
    Same MIDI, same parameters, but driven from a recorded instrument sample.

Usage:
    uv run python examples/notes_eg.py        # interactive menu
    uv run python examples/notes_eg.py 1      # run demo 1 only
    uv run python examples/notes_eg.py a      # run all demos

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path

import pygmu2 as pg
from examples_helper import run_demos

pg.set_sample_rate(44100)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# MIDI file lives in <project_root>/vault/
MIDI_FILE = Path(__file__).parent / "midi" / "earle_of_salisbury.mid"

# Optional recorded instrument sample (mono, tuned to MIDI pitch 77 = F5)
WAV_FILE = Path(__file__).parent / "audio" / "uke_54.wav"

# ---------------------------------------------------------------------------
# Shared parameters (matching the original snippet)
# ---------------------------------------------------------------------------

TEMPO        = 80     # BPM
GAIN_FACTOR  = 0.25   # master amplitude scale
NATIVE_PITCH = 48     # MIDI pitch that corresponds to the source sound at rate 1.0


def _load_notes():
    """Parse the MIDI file and return a list of Note objects."""
    if not MIDI_FILE.exists():
        raise FileNotFoundError(
            f"MIDI file not found: {MIDI_FILE}\n"
            "Place the file at vault/earle_of_salisbury.mid relative to the project root."
        )
    notes = pg.get_notes_from_midi(str(MIDI_FILE))
    notes = notes[:75]
    print(f"  Loaded {len(notes)} notes from {MIDI_FILE.name}")
    return notes


# ---------------------------------------------------------------------------
# Demo 1: Supersaw wave source
# ---------------------------------------------------------------------------

def demo_saw_source():
    """Play MIDI notes using a 512 Hz sine wave as the source sound."""
    print("Demo: NotesPE — sine source (512 Hz)")
    notes = _load_notes()

    saw_pe = pg.BlitSawPE(frequency=256.0, amplitude=0.15)
    
    sr = pg.get_sample_rate()
    ramp_samples = int(0.7 * sr)   # 1200 ms

    freq_ramp_pe = pg.PiecewisePE(
        [(0, 9000.0), (ramp_samples, 400.0)],
        transition_type=pg.TransitionType.EXPONENTIAL,
        extend_mode=pg.ExtendMode.HOLD_LAST,
    )

   # filtered_pe = pg.BiquadPE(saw_pe, mode=pg.BiquadMode.LOWPASS, frequency=freq_ramp_pe, q=2)
    
    filtered_pe = pg.MovingAveragePE(saw_pe, window=pg.window_for_cutoff(freq_ramp_pe, sr))  # ~1 kHz

    spatial_pe = pg.SpatialPE(filtered_pe, method=pg.SpatialLinear(azimuth=-90.0))
    
    audio_dir = Path(__file__).parent / "audio"
    ir_path = audio_dir / "long_ir44.wav"
    ir = pg.WavReaderPE(str(ir_path))
    reverb_pe = pg.ReverbPE(spatial_pe, ir, mix=0.75, normalize_ir=True)

    music = pg.NotesPE(
        reverb_pe,
        notes,
        tempo=TEMPO,
        gain_factor=GAIN_FACTOR,
        native_pitch=NATIVE_PITCH,
    )

    print(f"  Duration: {music.extent().end / pg.get_sample_rate():.2f} s")
    pg.play_offline(music)


# ---------------------------------------------------------------------------
# Demo 2: recorded WAV source
# ---------------------------------------------------------------------------

def demo_wav_source():
    """Play MIDI notes using a recorded instrument sample as the source."""
    print("Demo: NotesPE — WAV source")

    if not WAV_FILE.exists():
        print(f"  WAV file not found: {WAV_FILE}")
        print("  Edit WAV_FILE in this script to point at a real sample, then re-run.")
        return

    notes = _load_notes()

    source = pg.WavReaderPE(str(WAV_FILE))
    
    delayed_pe = pg.DelayPE(source, delay=22000)
    delayed_pe2 = pg.DelayPE(source, delay=33000)
    
    gain_pe = pg.GainPE(delayed_pe, gain=0.25)
    gain_pe2 = pg.GainPE(delayed_pe2, gain=0.16)
    mixed_pe = pg.MixPE(source, gain_pe, gain_pe2)
                
    music = pg.NotesPE(
        mixed_pe,
        notes,
        tempo=TEMPO,
        gain_factor=GAIN_FACTOR,
        native_pitch=NATIVE_PITCH,
    )

    print(f"  Duration: {music.extent().end / pg.get_sample_rate():.2f} s")
    pg.play_offline(music)


# ---------------------------------------------------------------------------
# DEMOS registry and entry point
# ---------------------------------------------------------------------------

DEMOS = [
    ("NotesPE with blitsaw source", demo_saw_source),
    ("NotesPE with WAV source",  demo_wav_source),
]

if __name__ == "__main__":
    run_demos(DEMOS)
