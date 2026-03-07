"""
trafalm_etude3.py

Diagnostic version: pick two random raw slices from the remote Strudel
catalog, blur each slice with the Tralfam chain, apply fade in/out after the
blur, then sequence the two slices with 30% overlap.

Usage:
  uv run python examples/trafalm_etude3.py
"""

import random

import pygmu2 as pg

SAMPLE_RATE = 44100
pg.set_sample_rate(SAMPLE_RATE)

STRUDEL_JSON_URL = "https://software.tomandandy.com/strudel.json"
RANDOM_SEED = 42
NUM_FILES = 1
NUM_SLICES = 2
MIN_SLICE_SECONDS = 1.20
MAX_SLICE_SECONDS = 3.20
SLICE_FADE_IN_SECONDS = 1.0
SLICE_FADE_OUT_SECONDS = 1.0
TRALFAM_TAIL_SECONDS = 2.0
TRALFAM_LOOP_COUNT = 3
TRALFAM_NORMALIZE_PEAK = 0.33
OVERLAP_RATIO = 0.30
MASTER_GAIN = 0.62


def catalog_wav_entries(library):
    """Return concrete (sound name, variant index, relative path) WAV entries."""
    entries = []
    for sound_name, variants in library._audio_paths.items():
        for index, rel_path in enumerate(variants):
            if str(rel_path).lower().endswith(".wav"):
                entries.append((sound_name, index, rel_path))
    return entries


def choose_sources(library, rng):
    """Resolve random catalog entries to local WAV paths and readers."""
    entries = catalog_wav_entries(library)
    if not entries:
        raise RuntimeError("No WAV entries found in the Strudel catalog.")

    if len(entries) < NUM_FILES:
        raise RuntimeError(
            f"Requested {NUM_FILES} files, but catalog only has {len(entries)} WAV entries."
        )

    chosen = []
    shuffled_entries = entries[:]
    rng.shuffle(shuffled_entries)
    for sound_name, index, rel_path in shuffled_entries:
        try:
            resolved_path = library.resolve(sound_name, index=index)
            reader = pg.WavReaderPE(resolved_path)
        except Exception as exc:
            print(f"Skipping {sound_name}: {exc}", flush=True)
            continue
        file_sample_rate = reader.file_sample_rate
        if file_sample_rate != SAMPLE_RATE:
            continue
        chosen.append(
            {
                "sound_name": sound_name,
                "index": index,
                "catalog_path": rel_path,
                "resolved_path": resolved_path,
                "reader": reader,
                "frames": reader.extent().end,
                "file_sample_rate": file_sample_rate,
            }
        )
        if len(chosen) == NUM_FILES:
            return chosen

    raise RuntimeError(
        f"Only found {len(chosen)} WAV files at {SAMPLE_RATE} Hz in the Strudel catalog; "
        f"need {NUM_FILES}."
    )


def make_random_slice(reader, rng):
    """Extract a random slice from a file with adjustable duration bounds."""
    file_frames = reader.extent().end
    file_sample_rate = reader.file_sample_rate
    min_duration = int(round(MIN_SLICE_SECONDS * file_sample_rate))
    max_duration = int(round(MAX_SLICE_SECONDS * file_sample_rate))

    if file_frames <= 0:
        raise RuntimeError(f"Cannot slice empty file: {reader.path}")

    max_allowed_duration = min(max_duration, file_frames)
    min_allowed_duration = min(min_duration, max_allowed_duration)
    duration = rng.randint(min_allowed_duration, max_allowed_duration)
    start = rng.randint(0, file_frames - duration)

    slice_pe = pg.SlicePE(
        reader,
        start,
        duration,
    )
    return slice_pe, start, duration


def apply_probe_fade(pe):
    """Apply an explicit gain envelope after the blur stage."""
    duration = pe.extent().end
    fade_in = min(int(round(SLICE_FADE_IN_SECONDS * SAMPLE_RATE)), duration)
    fade_out = min(int(round(SLICE_FADE_OUT_SECONDS * SAMPLE_RATE)), duration)

    if duration <= 0:
        raise RuntimeError("Slice has no duration.")

    sustain_end = max(fade_in, duration - fade_out)
    envelope = pg.PiecewisePE(
        [
            (0, 0.0),
            (fade_in, 1.0),
            (sustain_end, 1.0),
            (duration, 0.0),
        ],
        transition_type=pg.TransitionType.LINEAR,
    )
    return pg.GainPE(pe, gain=envelope)


def make_blurry_slice(slice_pe, slice_duration, rng):
    """Blur one raw slice via SetExtentPE -> TralfamPE -> LoopPE."""
    padded_duration = slice_duration + int(round(TRALFAM_TAIL_SECONDS * slice_pe.sample_rate))
    padded = pg.SetExtentPE(slice_pe, 0, padded_duration)
    tralfam = pg.TralfamPE(
        padded,
        seed=rng.randint(0, 2**31 - 1),
        normalize_peak=TRALFAM_NORMALIZE_PEAK,
    )
    return pg.LoopPE(tralfam, count=TRALFAM_LOOP_COUNT)


def build_two_slice_sequence(reader, rng):
    """Create two blurred/ramped slices and overlap them by OVERLAP_RATIO."""
    processed = []
    for slice_num in range(NUM_SLICES):
        slice_pe, start, duration = make_random_slice(reader, rng)
        blurry = make_blurry_slice(slice_pe, duration, rng)
        ramped = apply_probe_fade(blurry)
        processed.append(
            {
                "slice_num": slice_num + 1,
                "start": start,
                "raw_duration": duration,
                "pe": ramped,
            }
        )

    first = processed[0]
    second = processed[1]
    first_duration = first["pe"].extent().end
    second_offset = max(1, int(round(first_duration * (1.0 - OVERLAP_RATIO))))

    print(
        f"Slice 1: start={first['start'] / SAMPLE_RATE:.2f}s, "
        f"raw_dur={first['raw_duration'] / SAMPLE_RATE:.2f}s, "
        f"blurred_dur={first_duration / SAMPLE_RATE:.2f}s, "
        f"placed @ 0.00s",
        flush=True,
    )
    print(
        f"Slice 2: start={second['start'] / SAMPLE_RATE:.2f}s, "
        f"raw_dur={second['raw_duration'] / SAMPLE_RATE:.2f}s, "
        f"blurred_dur={second['pe'].extent().end / SAMPLE_RATE:.2f}s, "
        f"placed @ {second_offset / SAMPLE_RATE:.2f}s",
        flush=True,
    )

    return pg.MixPE(
        first["pe"],
        pg.DelayPE(second["pe"], delay=second_offset),
    )


def build_etude(seed=RANDOM_SEED):
    rng = random.Random(seed)

    print(f"Fetching catalog: {STRUDEL_JSON_URL}", flush=True)
    library = pg.AudioLibrary.from_url(STRUDEL_JSON_URL)

    source_info = choose_sources(library, rng)[0]
    file_sample_rate = source_info["file_sample_rate"]
    print(
        f"Selected source: {source_info['sound_name']} ({source_info['catalog_path']}) "
        f"{source_info['frames'] / file_sample_rate:.2f}s @ {file_sample_rate} Hz",
        flush=True,
    )

    print(
        f"Building {NUM_SLICES} slices with {OVERLAP_RATIO:.0%} overlap, "
        f"fade_in={SLICE_FADE_IN_SECONDS:.2f}s, "
        f"fade_out={SLICE_FADE_OUT_SECONDS:.2f}s",
        flush=True,
    )
    sequence = build_two_slice_sequence(source_info["reader"], rng)
    return pg.GainPE(sequence, gain=MASTER_GAIN)


def main():
    etude = build_etude()
    total_duration = etude.extent().end / SAMPLE_RATE
    print(f"\nPlaying etude ({total_duration:.2f} seconds)...", flush=True)
    pg.play(etude, sample_rate=SAMPLE_RATE)


if __name__ == "__main__":
    main()