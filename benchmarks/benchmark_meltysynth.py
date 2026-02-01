#!/usr/bin/env python3
"""
Benchmark meltysynth render time (simple_chord-style: C+E+G for 3s).

Measures synthesizer.render(left, right) only (no WAV write). Use to compare
non-vectorized vs vectorized code: run once with vectorization stashed (baseline),
then again after restoring vectorization.

Block sizes: 64, 256, 1024 (per VECTORIZATION_REPORT.md).

Usage (from project root):
  uv run python benchmarks/benchmark_meltysynth.py [path/to/soundfont.sf2]
  uv run python benchmarks/benchmark_meltysynth.py --runs 5 --warmup 2

Requires a .sf2 file; default is examples/audio/TimGM6mb.sf2 if present.
"""

import sys
import time
from pathlib import Path

# Add src for development
sys.path.insert(0, "src")

from pygmu2.meltysynth import (
    SoundFont,
    Synthesizer,
    SynthesizerSettings,
    create_buffer,
)


def _default_soundfont_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    in_repo = repo_root / "examples" / "audio" / "TimGM6mb.sf2"
    if in_repo.exists():
        return str(in_repo)
    return "TimGM6mb.sf2"


def run_render_benchmark(
    sound_font_path: str,
    block_sizes: list[int] = (64, 256, 1024),
    duration_s: float = 3.0,
    sample_rate: int = 44100,
    num_runs: int = 5,
    warmup_runs: int = 2,
) -> None:
    if not Path(sound_font_path).exists():
        print(f"SoundFont not found: {sound_font_path}", file=sys.stderr)
        print("Download a .sf2 (e.g. TimGM6mb.sf2) or pass its path.", file=sys.stderr)
        sys.exit(1)

    sound_font = SoundFont.from_file(sound_font_path)
    length_samples = int(duration_s * sample_rate)

    print("Meltysynth render benchmark (simple_chord-style, no WAV write)")
    print(f"  SoundFont: {sound_font_path}")
    print(f"  Duration: {duration_s}s ({length_samples} samples)")
    print(f"  Runs: {num_runs} (warmup {warmup_runs})")
    print()

    left = create_buffer(length_samples)
    right = create_buffer(length_samples)

    for block_size in block_sizes:
        settings = SynthesizerSettings(sample_rate)
        settings.block_size = block_size

        # Warmup: one synth, multiple renders
        syn = Synthesizer(sound_font, settings)
        syn.note_on(0, 60, 100)
        syn.note_on(0, 64, 100)
        syn.note_on(0, 67, 100)
        for _ in range(warmup_runs):
            syn.render(left, right)

        times_s: list[float] = []
        for _ in range(num_runs):
            syn = Synthesizer(sound_font, settings)
            syn.note_on(0, 60, 100)
            syn.note_on(0, 64, 100)
            syn.note_on(0, 67, 100)
            start = time.perf_counter()
            syn.render(left, right)
            elapsed = time.perf_counter() - start
            times_s.append(elapsed)

        mean_s = sum(times_s) / len(times_s)
        realtime_s = length_samples / sample_rate
        ratio = realtime_s / mean_s if mean_s > 0 else 0
        mean_ms = mean_s * 1000
        print(f"  block_size={block_size:4d}  mean={mean_ms:7.2f} ms  realtime_ratio={ratio:.2f}x")

    print()
    print("(realtime_ratio > 1 = faster than realtime)")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    parser.add_argument(
        "soundfont",
        nargs="?",
        default=None,
        help="Path to .sf2 (default: examples/audio/TimGM6mb.sf2 if present)",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per block size")
    parser.add_argument(
        "--blocks",
        default="64,256,1024",
        help="Comma-separated block sizes (default: 64,256,1024)",
    )
    parser.add_argument("--duration", type=float, default=3.0, help="Render duration in seconds")
    args = parser.parse_args()

    soundfont = args.soundfont if args.soundfont is not None else _default_soundfont_path()
    block_sizes = [int(x.strip()) for x in args.blocks.split(",")]

    run_render_benchmark(
        soundfont,
        block_sizes=block_sizes,
        duration_s=args.duration,
        num_runs=args.runs,
        warmup_runs=args.warmup,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
