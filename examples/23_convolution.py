#!/usr/bin/env python3
"""
Example 23: ConvolvePE - convolution reverb (room impulse responses)

This example demonstrates ConvolvePE by convolving dry sources with room
impulse responses (IRs).

Expected files (place these in examples/audio/, all 48KHz sample rate):
- spoken_voice48.wav
- acoustic_drums48.wav
- plate_ir48.wav
- long_ir48.wav

Notes:
- IRs should be finite and start at time 0 (standard WAV files satisfy this).
- For simplicity, this demo assumes source and IR sample rates match.
- IRs should be mono, or match the source's channel count.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import logging
from pathlib import Path

from pygmu2 import (
    AudioRenderer,
    ConvolvePE,
    CropPE,
    DelayPE,
    DiracPE,
    Extent,
    GainPE,
    LimiterPE,
    MixPE,
    seconds_to_samples,
    SpatialAdapter,
    SpatialLinear,
    SpatialPE,
    WavReaderPE,
)
import pygmu2 as pg
pg.set_sample_rate(44100)



AUDIO_DIR = Path(__file__).parent / "audio"

SPOKEN_PATH = AUDIO_DIR / "spoken_voice44.wav"
DRUMS_PATH = AUDIO_DIR / "acoustic_drums44.wav"
DRUMS_MONO_PATH = AUDIO_DIR / "acoustic_drums_mono44.wav"
PLATE_IR_PATH = AUDIO_DIR / "plate_ir44.wav"
LONG_IR_PATH = AUDIO_DIR / "long_ir44.wav"

logger = logging.getLogger(__name__)


def create_pingpong_ir_pe(sample_rate: int, beats_per_minute: float = 92):
    """
    Create a stereo 'ping pong' impluse response.  It produces two impulse
    responses:
    After one beat, it emits a unit impulse in the left channel.
    After two beats, it emits a unit impulse in the right channel.
    """
    logger.debug(
        "create_pingpong_ir_pe(sample_rate=%s, beats_per_minute=%s)",
        sample_rate,
        beats_per_minute,
    )
    seconds_per_beat = 60.0 / beats_per_minute
    impulse = DiracPE(channels=1)
    delay_1_beat = int(round(float(seconds_to_samples(seconds_per_beat, sample_rate))))
    delay_2_beats = int(round(float(seconds_to_samples(2 * seconds_per_beat, sample_rate))))
    logger.debug(
        "create_pingpong_ir_pe: delay_1_beat=%s, delay_2_beats=%s",
        delay_1_beat,
        delay_2_beats,
    )
    beat_1_impulse = DelayPE(impulse, delay=delay_1_beat)
    beat_1_impulse = SpatialPE(
        beat_1_impulse, method=SpatialLinear(azimuth=-90.0)
    )
    beat_2_impulse = DelayPE(impulse, delay=delay_2_beats)
    beat_2_impulse = SpatialPE(
        beat_2_impulse, method=SpatialLinear(azimuth=90.0)
    )
    mix_stream = MixPE(beat_1_impulse, beat_2_impulse)
    pe = CropPE(mix_stream, 0, (delay_2_beats+1) - (0))
    ext = pe.extent()
    logger.debug(
        "create_pingpong_ir_pe: channels=%s, extent=(%s, %s)",
        pe.channel_count(),
        ext.start,
        ext.end,
    )
    return pe


def _load_wav(path: Path) -> WavReaderPE:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return WavReaderPE(str(path))


def _assert_sample_rate_match(source: WavReaderPE, ir: WavReaderPE) -> None:
    sr_s = getattr(source, "file_sample_rate", None)
    sr_i = getattr(ir, "file_sample_rate", None)
    if sr_s is None or sr_i is None:
        return
    if int(sr_s) != int(sr_i):
        raise ValueError(
            f"Sample rate mismatch: source={sr_s} Hz, IR={sr_i} Hz. "
            f"Please provide an IR at the same sample rate as the source for this demo."
        )


def _play(source_pe, sample_rate: int) -> None:
    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(source_pe)
    with renderer:
        renderer.start()
        renderer.play_extent()


def _demo_dry(source_path: Path, *, gain: float = 0.8) -> None:
    source_stream = _load_wav(source_path)
    sample_rate = int(source_stream.file_sample_rate)
    out_stream = GainPE(source_stream, gain=gain)

    print(f"Source: {source_path.name}")
    print("IR:     (dry)")
    print()

    _play(out_stream, sample_rate)


def _demo_wet(
    source_path: Path,
    ir_path: Path | None = None,
    *,
    ir_pe=None,
    wet_gain: float = 0.25,
) -> None:
    logger.debug(
        "_demo_wet(source_path=%s, ir_path=%s, ir_pe=%s, wet_gain=%s)",
        source_path,
        ir_path,
        "PE" if ir_pe is not None else None,
        wet_gain,
    )
    if ir_pe is None and ir_path is None:
        raise ValueError("Provide either ir_path or ir_pe")
    source_stream = _load_wav(source_path)
    ir_stream = _load_wav(ir_path) if ir_path is not None else ir_pe
    logger.debug("_demo_wet: ir_stream from %s", "ir_path" if ir_path is not None else "ir_pe")
    if ir_path is not None:
        _assert_sample_rate_match(source_stream, ir_stream)

    sample_rate = int(source_stream.file_sample_rate)
    logger.debug("_demo_wet: sample_rate=%s", sample_rate)

    # Compute IR energy norm for normalization
    ir_energy = ConvolvePE.ir_energy_norm(ir_stream)
    logger.debug("_demo_wet: ir_energy=%s", ir_energy)

    # Create wet signal (convolved with IR), normalized by energy
    wet_stream = ConvolvePE(source_stream, ir_stream)
    wet_stream = GainPE(wet_stream, gain=wet_gain / ir_energy)
    # Note: lookahead=0 required because ConvolvePE is stateful
    # wet_gained_limited = LimiterPE(wet_gained, ceiling=1.0, release=15.0, attack=0.01, lookahead=0)

    # Create dry signal at (1 - wet_gain) level
    dry_gain = 1.0 - wet_gain
    dry_stream = GainPE(source_stream, gain=dry_gain)
    # coerce dry_stream to have the same # of channels as wet_stream
    dry_stream = SpatialPE(
        dry_stream, 
        method=SpatialAdapter(channels=wet_stream.channel_count()))

    # Mix dry and wet signals
    out_stream = MixPE(dry_stream, wet_stream)

    ir_label = ir_path.name if ir_path is not None else "ping-pong IR (PE)"
    print(f"Source: {source_path.name}")
    print(f"IR:     {ir_label}")
    print(f"IR energy norm: {ir_energy:.2f}")
    print(f"Dry gain: {dry_gain:.2f}")
    print(f"Wet gain: {wet_gain:.2f} (effective: {wet_gain / ir_energy:.4f})")
    print()

    _play(out_stream, sample_rate)

def demo_spoken_dry():
    print("=== Demo: spoken voice (dry) ===")
    _demo_dry(SPOKEN_PATH, gain=0.8)


def demo_spoken_short():
    print("=== Demo: spoken voice * plate_ir ===")
    _demo_wet(SPOKEN_PATH, PLATE_IR_PATH, wet_gain=0.30)


def demo_spoken_long():
    print("=== Demo: spoken voice * long_ir ===")
    _demo_wet(SPOKEN_PATH, LONG_IR_PATH, wet_gain=0.30)

def demo_drums_dry():
    print("=== Demo: drums (dry) ===")
    _demo_dry(DRUMS_PATH, gain=0.8)

def demo_drums_short():
    print("=== Demo: drums * plate_ir ===")
    _demo_wet(DRUMS_PATH, PLATE_IR_PATH, wet_gain=0.35)

def demo_drums_long():
    print("=== Demo: drums * long_ir ===")
    _demo_wet(DRUMS_PATH, LONG_IR_PATH, wet_gain=0.20)

def demo_drums_mono_dry():
    print("=== Demo: mono drums (dry) ===")
    _demo_dry(DRUMS_MONO_PATH, gain=0.8)

def demo_drums_mono_to_stereo():
    print("=== Demo: drums (mono) spread to stereo via ping-pong IR (PE) ===")
    source_stream = _load_wav(DRUMS_MONO_PATH)
    sample_rate = int(source_stream.file_sample_rate)
    ir_pe = create_pingpong_ir_pe(sample_rate, beats_per_minute=91)
    _demo_wet(DRUMS_MONO_PATH, ir_pe=ir_pe, wet_gain=0.65)

def demo_all():
    demo_spoken_dry()
    demo_spoken_short()
    demo_spoken_long()
    demo_drums_dry()
    demo_drums_short()
    demo_drums_long()
    demo_drums_mono_dry()
    demo_drums_mono_to_stereo()

if __name__ == "__main__":
    import os
    import sys

    # Enable DEBUG to see create_pingpong_ir_pe and _demo_wet flow (e.g. PYGMU_DEBUG=1)
    if os.environ.get("PYGMU_DEBUG"):
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    demos = [
        ("1", "spoken voice, dry", demo_spoken_dry),
        ("2", "spoken voice * plate_ir", demo_spoken_short),
        ("3", "spoken voice * long_ir", demo_spoken_long),
        ("4", "drums, dry", demo_drums_dry),
        ("5", "drums * plate_ir", demo_drums_short),
        ("6", "drums * long_ir", demo_drums_long),
        ("7", "drums (mono) dry", demo_drums_mono_dry),
        ("8", "drums (mono) spread to stereo via reverb", demo_drums_mono_to_stereo),
        ("a", "All demos", demo_all),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("pygmu2 Convolution Examples (ConvolvePE)")
        print("--------------------------------------")
        for key, name, _ in demos:
            print(f"{key}: {name}")
        print()
        choice = input(f"Choose a demo (1-{len(demos)-1} or 'a' for all): ").strip().lower()
        print()

    try:
        for key, _name, fn in demos:
            if key == choice:
                fn()
                break
        else:
            print("Invalid choice.")
    except FileNotFoundError as e:
        print(str(e))
        print()
        print("Place your impulse response WAV files in:")
        print(f"  {AUDIO_DIR}")
        print()
        print("Expected:")
        print(f"  - {PLATE_IR_PATH.name}")
        print(f"  - {LONG_IR_PATH.name}")
    except Exception as e:
        print(f"Error: {e}")

