"""
tralfam_etude1.py

Build a randomized etude from the remote Strudel catalog:
1. download the catalog
2. choose random WAV files
3. extract random slices from each file
4. extend each slice with the SetExtentPE -> TralfamPE -> LoopPE pattern
5. sequence the processed slices with 30% overlap using MixPE
6. compress the full mix
7. play it

Usage:
  uv run python examples/tralfam_etude1.py
"""

import random

import pygmu2 as pg

SAMPLE_RATE = 44100
pg.set_sample_rate(SAMPLE_RATE)

STRUDEL_JSON_URL = "https://software.tomandandy.com/strudel.json"
RANDOM_SEED = 42
NUM_FILES = 4
SLICES_PER_FILE = 2
MIN_SLICE_SECONDS = 1.20
MAX_SLICE_SECONDS = 3.20
SLICE_FADE_IN_SECONDS = 1
SLICE_FADE_OUT_SECONDS = 1
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


def apply_post_blur_fade(pe):
    """Apply fade in/out after the blur stage has created the full texture."""
    duration = pe.extent().end
    return pg.SlicePE(
        pe,
        0,
        duration,
        fade_in_seconds=SLICE_FADE_IN_SECONDS,
        fade_out_seconds=SLICE_FADE_OUT_SECONDS,
    )


def make_blurry_slice(slice_pe, slice_duration, rng):
    """Turn a short slice into a longer blurry texture."""
    padded_duration = slice_duration + int(round(TRALFAM_TAIL_SECONDS * slice_pe.sample_rate))
    padded = pg.SetExtentPE(slice_pe, 0, padded_duration)
    tralfam = pg.TralfamPE(
        padded,
        seed=rng.randint(0, 2**31 - 1),
        normalize_peak=TRALFAM_NORMALIZE_PEAK,
    )
    looped = pg.LoopPE(tralfam, count=TRALFAM_LOOP_COUNT)
    return apply_post_blur_fade(looped)


def collect_processed_slices(chosen_sources, rng):
    """Extract, blur, and annotate slices from each chosen source file."""
    processed = []
    for source_info in chosen_sources:
        file_sample_rate = source_info["file_sample_rate"]
        print(
            f"Source: {source_info['sound_name']} -> {source_info['catalog_path']} "
            f"({source_info['frames'] / file_sample_rate:.2f}s @ "
            f"{file_sample_rate} Hz)",
            flush=True,
        )
        for slice_num in range(SLICES_PER_FILE):
            slice_pe, start, duration = make_random_slice(source_info["reader"], rng)
            blurry = make_blurry_slice(slice_pe, duration, rng)
            processed.append(
                {
                    "source_name": source_info["sound_name"],
                    "slice_num": slice_num + 1,
                    "start": start,
                    "duration": duration,
                    "pe": blurry,
                }
            )
            print(
                f"  slice {slice_num + 1}: start={start / file_sample_rate:.2f}s, "
                f"dur={duration / file_sample_rate:.2f}s, "
                f"blurred_dur={blurry.extent().end / file_sample_rate:.2f}s",
                flush=True,
            )
    return processed


def sequence_slices(processed_slices, rng):
    """Shuffle slices, position them with overlap, and mix into one stream."""
    if not processed_slices:
        raise RuntimeError("No processed slices were created.")

    play_order = processed_slices[:]
    rng.shuffle(play_order)

    positioned = []
    cursor = 0
    print("\nSequence order:", flush=True)
    for item in play_order:
        duration = item["pe"].extent().end
        positioned.append(pg.DelayPE(item["pe"], delay=cursor))
        print(
            f"  {item['source_name']} slice {item['slice_num']} @ {cursor / SAMPLE_RATE:.2f}s "
            f"for {duration / SAMPLE_RATE:.2f}s",
            flush=True,
        )
        cursor += max(1, int(round(duration * (1.0 - OVERLAP_RATIO))))

    return pg.MixPE(*positioned)


def build_etude(seed=RANDOM_SEED):
    rng = random.Random(seed)

    print(f"Fetching catalog: {STRUDEL_JSON_URL}", flush=True)
    library = pg.AudioLibrary.from_url(STRUDEL_JSON_URL)

    chosen_sources = choose_sources(library, rng)
    print(f"Selected {len(chosen_sources)} files:", flush=True)
    for source_info in chosen_sources:
        print(
            f"  {source_info['sound_name']} ({source_info['catalog_path']})",
            flush=True,
        )

    processed_slices = collect_processed_slices(chosen_sources, rng)
    mixed = sequence_slices(processed_slices, rng)
    compressed = pg.CompressorPE(
        mixed,
        threshold=-24,
        ratio=6,
        attack=0.02,
        release=0.25,
        knee=6,
        makeup_gain="auto",
    )
    return pg.GainPE(compressed, gain=MASTER_GAIN)


def main():
    etude = build_etude()
    total_duration = etude.extent().end / SAMPLE_RATE
    print(f"\nPlaying etude ({total_duration:.2f} seconds)...", flush=True)
    pg.play(etude, sample_rate=SAMPLE_RATE)


if __name__ == "__main__":
    main()