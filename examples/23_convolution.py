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

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pathlib import Path

import numpy as np

from pygmu2 import (
    AudioRenderer,
    ConvolvePE,
    GainPE,
    LimiterPE,
    MixPE,
    WavReaderPE,
)


AUDIO_DIR = Path(__file__).parent / "audio"

SPOKEN_PATH = AUDIO_DIR / "spoken_voice44.wav"
DRUMS_PATH = AUDIO_DIR / "acoustic_drums44.wav"
PLATE_IR_PATH = AUDIO_DIR / "plate_ir44.wav"
LONG_IR_PATH = AUDIO_DIR / "long_ir44.wav"


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


def _compute_ir_energy_norm(ir_stream: WavReaderPE) -> float:
    """
    Compute the energy norm of the IR: sqrt(sum of squares).
    
    This measures the total energy in the IR. Dividing by this value
    normalizes the convolution output to have similar energy to the input.
    
    Returns:
        The energy norm (sqrt of sum of squared samples).
    """
    extent = ir_stream.extent()
    if extent.start is None or extent.end is None:
        return 1.0
    
    duration = extent.end - extent.start
    ir_data = ir_stream.render(extent.start, duration).data
    
    # Energy norm = sqrt(sum of squares)
    energy_norm = np.sqrt(np.sum(ir_data ** 2))
    
    return energy_norm if energy_norm > 1e-10 else 1.0


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


def _demo_wet(source_path: Path, ir_path: Path, *, wet_gain: float = 0.25) -> None:
    source_stream = _load_wav(source_path)
    ir_stream = _load_wav(ir_path)
    _assert_sample_rate_match(source_stream, ir_stream)

    sample_rate = int(source_stream.file_sample_rate)

    # Compute IR energy norm for normalization
    ir_energy = _compute_ir_energy_norm(ir_stream)

    # Create wet signal (convolved with IR), normalized by energy
    wet_stream = ConvolvePE(source_stream, ir_stream)
    wet_gained = GainPE(wet_stream, gain=wet_gain / ir_energy)
    # Note: lookahead=0 required because ConvolvePE is stateful
    # wet_gained_limited = LimiterPE(wet_gained, ceiling=1.0, release=15.0, attack=0.01, lookahead=0)

    # Create dry signal at (1 - wet_gain) level
    dry_gain = 1.0 - wet_gain
    dry_gained = GainPE(source_stream, gain=dry_gain)

    # Mix dry and wet signals
    out_stream = MixPE(dry_gained, wet_gained)

    print(f"Source: {source_path.name}")
    print(f"IR:     {ir_path.name}")
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


def demo_all():
    demo_spoken_dry()
    demo_spoken_short()
    demo_spoken_long()
    demo_drums_dry()
    demo_drums_short()
    demo_drums_long()


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "spoken voice, dry", demo_spoken_dry),
        ("2", "spoken voice * plate_ir", demo_spoken_short),
        ("3", "spoken voice * long_ir", demo_spoken_long),
        ("4", "drums, dry", demo_drums_dry),
        ("5", "drums * plate_ir", demo_drums_short),
        ("6", "drums * long_ir", demo_drums_long),
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
        choice = input("Choose a demo (1-6 or 'a' for all): ").strip().lower()
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

