"""
37_sequence_eg.py

SequencePE example showing OVERLAP and NON_OVERLAP modes with audio material.
"""

from pathlib import Path

import pygmu2 as pg


def _build_sources():
    audio_dir = Path(__file__).parent / "audio"
    choir_path = audio_dir / "choir.wav"
    source = pg.WavReaderPE(str(choir_path))
    sample_rate = source.file_sample_rate or 44100

    # Original choir
    choir = source

    # Pitch-shift down three semitones (rate = 2^(-3/12))
    rate = 2 ** (-3 / 12)
    choir_down = pg.TimeWarpPE(source, rate=rate)

    # Pitch-shift up four semitones (rate = 2^(-4/12))
    rate = 2 ** (4 / 12)
    choir_up = pg.TimeWarpPE(source, rate=rate)

    return choir, choir_down, choir_up, sample_rate

def _play(source, sample_rate):
    renderer = pg.AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(pg.CropPE(source, pg.Extent(0, int(3.5 * sample_rate))))
    with renderer:
        renderer.start()
        renderer.play_extent()

def demo_overlap():
    print("SequencePE with mode=OVERLAP")
    print("----------------------------")
    choir, choir_down, choir_up, sample_rate = _build_sources()

    seq = pg.SequencePE(
        (choir, 0),
        (choir_down, int(1.0 * sample_rate)),
        (choir_up, int(2.0 * sample_rate)),
        mode=pg.SequenceMode.OVERLAP,
    )
    _play(seq, sample_rate)


def demo_non_overlap():
    print("SequencePE with mode=NON_OVERLAP")
    print("--------------------------------")
    choir, choir_down, choir_up, sample_rate = _build_sources()

    seq = pg.SequencePE(
        (choir, 0),
        (choir_down, int(1.0 * sample_rate)),
        (choir_up, int(2.0 * sample_rate)),
        mode=pg.SequenceMode.NON_OVERLAP,
    )
    _play(seq, sample_rate)


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "Demo overlap", demo_overlap),
        ("2", "Demo non-overlap", demo_non_overlap),
        ("a", "Demo all", None),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("SequencePE Examples")
        print("-------------------")
        for key, name, _ in demos:
            print(f"  {key}: {name}")
        print()
        choice = input(f"Choice (1-{len(demos)-1} or 'a'): ").strip().lower()

    if choice == "a":
        for _key, _name, fn in demos:
            if fn is not None:
                fn()
    else:
        for key, _name, fn in demos:
            if key == choice and fn is not None:
                fn()
                break
        else:
            print("Invalid choice.")
