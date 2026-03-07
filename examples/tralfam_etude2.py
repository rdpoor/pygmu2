"""
tralfam_etude2.py

Build a randomized etude from the remote Strudel audio catalog:

1. Download the catalog of available .wav files
2. Choose N files at random
3. Extract X random slices from each file (adjustable min/max duration)
4. Blur each slice via SetExtentPE -> TralfamPE -> LoopPE
5. Sequence the processed slices in random order with 30% overlap (MixPE)
6. Compress the full mix
7. Play

Usage:
  uv run python examples/tralfam_etude2.py
"""

import random

import pygmu2 as pg

SAMPLE_RATE = 44100
pg.set_sample_rate(SAMPLE_RATE)

# ---------------------------------------------------------------------------
# Adjustable parameters
# ---------------------------------------------------------------------------
STRUDEL_JSON_URL = "https://software.tomandandy.com/strudel.json"
RANDOM_SEED = 42

NUM_FILES = 4
SLICES_PER_FILE = 3
MIN_SLICE_SECONDS = 1.20
MAX_SLICE_SECONDS = 3.20
SLICE_FADE_IN_SECONDS = 1
SLICE_FADE_OUT_SECONDS = 1

TRALFAM_TAIL_SECONDS = 2.0
TRALFAM_LOOP_COUNT = 5
TRALFAM_NORMALIZE_PEAK = 0.33

OVERLAP_RATIO = 0.30
MASTER_GAIN = 0.62

COMPRESSOR_THRESHOLD = -24
COMPRESSOR_RATIO = 6
COMPRESSOR_ATTACK = 0.02
COMPRESSOR_RELEASE = 0.25
COMPRESSOR_KNEE = 6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def catalog_wav_entries(library):
    """Return (sound_name, variant_index, relative_path) for every WAV entry."""
    entries = []
    for sound_name, variants in library._audio_paths.items():
        for idx, rel_path in enumerate(variants):
            if str(rel_path).lower().endswith(".wav"):
                entries.append((sound_name, idx, rel_path))
    return entries


def choose_sources(library, rng):
    """Resolve NUM_FILES random catalog entries to local paths and WavReaderPEs."""
    entries = catalog_wav_entries(library)
    if len(entries) < NUM_FILES:
        raise RuntimeError(
            f"Need {NUM_FILES} WAV entries but catalog only has {len(entries)}."
        )

    rng.shuffle(entries)
    chosen = []
    for sound_name, idx, rel_path in entries:
        try:
            path = library.resolve(sound_name, index=idx)
            reader = pg.WavReaderPE(path)
        except Exception as exc:
            print(f"  skipping {sound_name}: {exc}", flush=True)
            continue
        if reader.file_sample_rate != SAMPLE_RATE:
            continue
        chosen.append({
            "name": sound_name,
            "index": idx,
            "rel_path": rel_path,
            "reader": reader,
            "frames": reader.extent().end,
        })
        if len(chosen) == NUM_FILES:
            return chosen

    raise RuntimeError(
        f"Only {len(chosen)} WAV files at {SAMPLE_RATE} Hz found; need {NUM_FILES}."
    )


def random_slice(reader, rng):
    """Cut a random segment from *reader* with adjustable min/max duration."""
    total = reader.extent().end
    if total <= 0:
        raise RuntimeError("Cannot slice an empty file.")

    lo = int(round(MIN_SLICE_SECONDS * SAMPLE_RATE))
    hi = int(round(MAX_SLICE_SECONDS * SAMPLE_RATE))
    hi = min(hi, total)
    lo = min(lo, hi)

    dur = rng.randint(lo, hi)
    start = rng.randint(0, total - dur)
    return pg.SlicePE(reader, start, dur), dur


def apply_post_blur_fade(pe):
    """Apply the fade after the blur stage has created the full texture."""
    dur = pe.extent().end
    return pg.SlicePE(
        pe,
        0,
        dur,
        fade_in_seconds=SLICE_FADE_IN_SECONDS,
        fade_out_seconds=SLICE_FADE_OUT_SECONDS,
    )


def blurrify(slice_pe, slice_dur, rng):
    """Pad, phase-randomize (TralfamPE), and loop a slice."""
    tail = int(round(TRALFAM_TAIL_SECONDS * SAMPLE_RATE))
    padded = pg.SetExtentPE(slice_pe, 0, slice_dur + tail)
    tralfam = pg.TralfamPE(
        padded,
        seed=rng.randint(0, 2**31 - 1),
        normalize_peak=TRALFAM_NORMALIZE_PEAK,
    )
    looped = pg.LoopPE(tralfam, count=TRALFAM_LOOP_COUNT)
    return apply_post_blur_fade(looped)


def gather_slices(sources, rng):
    """Extract, blur, and collect all slices from the chosen source files."""
    slices = []
    for src in sources:
        print(
            f"  {src['name']} ({src['rel_path']})  "
            f"{src['frames'] / SAMPLE_RATE:.2f}s",
            flush=True,
        )
        for n in range(SLICES_PER_FILE):
            seg, dur = random_slice(src["reader"], rng)
            blurred = blurrify(seg, dur, rng)
            slices.append({
                "name": src["name"],
                "slice": n + 1,
                "dur_s": dur / SAMPLE_RATE,
                "blurred_dur_s": blurred.extent().end / SAMPLE_RATE,
                "pe": blurred,
            })
            print(
                f"    slice {n + 1}: {dur / SAMPLE_RATE:.2f}s -> "
                f"{blurred.extent().end / SAMPLE_RATE:.2f}s blurred",
                flush=True,
            )
    return slices


def sequence_with_overlap(slices, rng):
    """Shuffle slices, position them with OVERLAP_RATIO overlap, mix."""
    rng.shuffle(slices)
    positioned = []
    cursor = 0
    print("\nSequence:", flush=True)
    for item in slices:
        dur = item["pe"].extent().end
        positioned.append(pg.DelayPE(item["pe"], delay=cursor))
        print(
            f"  {item['name']} #{item['slice']} @ {cursor / SAMPLE_RATE:.2f}s "
            f"({dur / SAMPLE_RATE:.2f}s)",
            flush=True,
        )
        cursor += max(1, int(round(dur * (1.0 - OVERLAP_RATIO))))
    return pg.MixPE(*positioned)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_etude(seed=RANDOM_SEED):
    rng = random.Random(seed)

    print(f"Fetching catalog: {STRUDEL_JSON_URL}", flush=True)
    library = pg.AudioLibrary.from_url(STRUDEL_JSON_URL)

    print(f"Choosing {NUM_FILES} files...", flush=True)
    sources = choose_sources(library, rng)

    print(f"Extracting {SLICES_PER_FILE} slices per file...", flush=True)
    slices = gather_slices(sources, rng)

    mixed = sequence_with_overlap(slices, rng)

    compressed = pg.CompressorPE(
        mixed,
        threshold=COMPRESSOR_THRESHOLD,
        ratio=COMPRESSOR_RATIO,
        attack=COMPRESSOR_ATTACK,
        release=COMPRESSOR_RELEASE,
        knee=COMPRESSOR_KNEE,
        makeup_gain="auto",
    )
    return pg.GainPE(compressed, gain=MASTER_GAIN)


if __name__ == "__main__":
    etude = build_etude()
    total = etude.extent().end / SAMPLE_RATE
    print(f"\nPlaying etude ({total:.2f}s)...", flush=True)
    pg.play(etude, sample_rate=SAMPLE_RATE)
    print("Done.", flush=True)
