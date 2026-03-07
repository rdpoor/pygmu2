"""
trafalm_etude3.py

Build a randomized etude from the remote Strudel catalog:
1. choose multiple random source files
2. extract multiple raw slices from each file
3. blur each slice with the Tralfam chain
4. apply fade in/out after the blur
5. sequence all slices in random order with 30% overlap
6. render the full piece to a WAV file

Usage:
  uv run python examples/trafalm_etude3.py
"""

import random
from pathlib import Path

import pygmu2 as pg

RANDOM_SEED = 44

SAMPLE_RATE = 44100
pg.set_sample_rate(SAMPLE_RATE)

EXAMPLES_DIR = Path(__file__).parent
IR_PATH = EXAMPLES_DIR / "audio" / "long_ir44.wav"

STRUDEL_JSON_URL = "https://software.tomandandy.com/strudel.json"
NUM_FILES = 4
SLICES_PER_FILE = 3
FORCED_SOURCE_NAME = None
MIN_SLICE_SECONDS = 1.20
MAX_SLICE_SECONDS = 3.20
SLICE_FADE_IN_SECONDS = 2.0
SLICE_FADE_OUT_SECONDS = 4.0
TRALFAM_TAIL_SECONDS = 5.0
TRALFAM_LOOP_COUNT = 7
TRALFAM_NORMALIZE_PEAK = 0.53
OVERLAP_RATIO = 0.30
USE_OVERLAP = True
MASTER_GAIN = 0.62
ENABLE_MASTER_COMPRESSION = True
COMPRESSOR_THRESHOLD = -28
COMPRESSOR_RATIO = 6
COMPRESSOR_ATTACK = 0.02
COMPRESSOR_RELEASE = 0.25
COMPRESSOR_KNEE = 6.0
REVERB_MIX = 0.28
OUTPUT_PATH = Path(__file__).with_name("trafalm_etude3_render.wav")


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

    if FORCED_SOURCE_NAME is not None:
        entries = [entry for entry in entries if entry[0] == FORCED_SOURCE_NAME]
        if not entries:
            raise RuntimeError(f"Forced source {FORCED_SOURCE_NAME!r} not found in the catalog.")

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


def apply_sequence_crossfade(pe, fade_in_samples=0, fade_out_start=None):
    """Apply an additional sequence-stage crossfade envelope."""
    duration = pe.extent().end
    if duration <= 0:
        raise RuntimeError("Slice has no duration.")

    if fade_out_start is None:
        fade_out_start = duration
    fade_in_samples = max(0, min(int(fade_in_samples), duration))
    fade_out_start = max(0, min(int(fade_out_start), duration))

    points = [(0, 1.0)]
    if fade_in_samples > 0:
        points = [(0, 0.0), (fade_in_samples, 1.0)]
    if fade_out_start > fade_in_samples:
        points.append((fade_out_start, 1.0))
    if fade_out_start < duration:
        points.append((duration, 0.0))

    envelope = pg.PiecewisePE(
        points,
        transition_type=pg.TransitionType.CONSTANT_POWER,
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


def load_reverb_ir():
    """Load the plate IR used by reverb_eg.py."""
    if not IR_PATH.exists():
        raise FileNotFoundError(f"Missing reverb IR: {IR_PATH}")
    ir = pg.WavReaderPE(str(IR_PATH))
    if ir.file_sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"IR sample rate mismatch: {IR_PATH} is {ir.file_sample_rate} Hz, "
            f"expected {SAMPLE_RATE} Hz."
        )
    return ir


def collect_processed_slices(sources, rng):
    """Extract, blur, ramp, and annotate slices from all chosen sources."""
    processed = []
    for source_info in sources:
        file_sample_rate = source_info["file_sample_rate"]
        print(
            f"Source: {source_info['sound_name']} ({source_info['catalog_path']}) "
            f"{source_info['frames'] / file_sample_rate:.2f}s",
            flush=True,
        )
        for slice_num in range(SLICES_PER_FILE):
            slice_pe, start, duration = make_random_slice(source_info["reader"], rng)
            blurry = make_blurry_slice(slice_pe, duration, rng)
            ramped = apply_post_blur_fade(blurry)
            processed.append(
                {
                    "source_name": source_info["sound_name"],
                    "slice_num": slice_num + 1,
                    "start": start,
                    "raw_duration": duration,
                    "pe": ramped,
                }
            )
            print(
                f"  slice {slice_num + 1}: start={start / file_sample_rate:.2f}s, "
                f"raw_dur={duration / file_sample_rate:.2f}s, "
                f"blurred_dur={ramped.extent().end / file_sample_rate:.2f}s",
                flush=True,
            )
    return processed


def sequence_with_overlap(processed_slices, rng):
    """Shuffle slices, crossfade overlaps, and mix into one stream."""
    if not processed_slices:
        raise RuntimeError("No processed slices were created.")

    play_order = processed_slices[:]
    rng.shuffle(play_order)

    positioned = []
    cursor = 0
    print("\nSequence order:", flush=True)
    for index, item in enumerate(play_order):
        duration = item["pe"].extent().end
        pe_to_place = item["pe"]

        if USE_OVERLAP and index < len(play_order) - 1:
            next_offset = max(1, int(round(duration * (1.0 - OVERLAP_RATIO))))
            pe_to_place = apply_sequence_crossfade(pe_to_place, fade_out_start=next_offset)
            overlap_samples = duration - next_offset
            print(
                f"  {item['source_name']} slice {item['slice_num']} @ {cursor / SAMPLE_RATE:.2f}s "
                f"for {duration / SAMPLE_RATE:.2f}s, "
                f"crossfade={overlap_samples / SAMPLE_RATE:.2f}s",
                flush=True,
            )
            cursor_step = next_offset
        else:
            print(
                f"  {item['source_name']} slice {item['slice_num']} @ {cursor / SAMPLE_RATE:.2f}s "
                f"for {duration / SAMPLE_RATE:.2f}s",
                flush=True,
            )
            cursor_step = duration

        if USE_OVERLAP and index > 0:
            previous_duration = play_order[index - 1]["pe"].extent().end
            previous_step = max(1, int(round(previous_duration * (1.0 - OVERLAP_RATIO))))
            fade_in_samples = previous_duration - previous_step
            pe_to_place = apply_sequence_crossfade(pe_to_place, fade_in_samples=fade_in_samples)

        positioned.append(pg.DelayPE(pe_to_place, delay=cursor))
        cursor += cursor_step

    return pg.MixPE(*positioned)


def build_etude(seed=RANDOM_SEED):
    rng = random.Random(seed)

    print(f"Fetching catalog: {STRUDEL_JSON_URL}", flush=True)
    library = pg.AudioLibrary.from_url(STRUDEL_JSON_URL)

    print(
        f"Selected {NUM_FILES} sources, building {SLICES_PER_FILE} slices per file with "
        f"{(f'{OVERLAP_RATIO:.0%} overlap' if USE_OVERLAP else 'no overlap')}, "
        f"fade_in={SLICE_FADE_IN_SECONDS:.2f}s, "
        f"fade_out={SLICE_FADE_OUT_SECONDS:.2f}s, "
        f"source_mode={'forced' if FORCED_SOURCE_NAME else 'random'}, "
        f"compressor={'on' if ENABLE_MASTER_COMPRESSION else 'off'}, "
        f"reverb_mix={REVERB_MIX:.2f}, "
        f"ir={IR_PATH.name}",
        flush=True,
    )
    sources = choose_sources(library, rng)
    processed_slices = collect_processed_slices(sources, rng)
    sequence = sequence_with_overlap(processed_slices, rng)
    mixed = sequence
    if ENABLE_MASTER_COMPRESSION:
        mixed = pg.CompressorPE(
            mixed,
            threshold=COMPRESSOR_THRESHOLD,
            ratio=COMPRESSOR_RATIO,
            attack=COMPRESSOR_ATTACK,
            release=COMPRESSOR_RELEASE,
            knee=COMPRESSOR_KNEE,
            makeup_gain="auto",
        )
    reverb = pg.ReverbPE(mixed, load_reverb_ir(), mix=REVERB_MIX, normalize_ir=True)
    return pg.GainPE(reverb, gain=MASTER_GAIN)


def main():
    etude = build_etude()
    total_duration = etude.extent().end / SAMPLE_RATE
    print(f"\nRendering etude ({total_duration:.2f} seconds)...", flush=True)
    pg.render_to_file(etude, str(OUTPUT_PATH), sample_rate=SAMPLE_RATE)
    print(f"Rendered to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()