"""
tralfam_etude4.py

Build a randomized etude from the remote Strudel catalog:
1. choose multiple random source files
2. extract multiple raw slices from each file
3. blur each slice with the Tralfam chain
4. apply fade in/out after the blur
5. estimate a tonic/root frequency per slice via periodicity analysis
6. sequence all slices in random order with 30% overlap
7. add a low supersaw bass layer tuned to each slice tonic
8. render the full piece to a WAV file

Usage:
  uv run python examples/tralfam_etude4.py
"""

import random
from pathlib import Path

import numpy as np
import pygmu2 as pg

RANDOM_SEED = 46

SAMPLE_RATE = 44100
pg.set_sample_rate(SAMPLE_RATE)

EXAMPLES_DIR = Path(__file__).parent
IR_PATH = EXAMPLES_DIR / "audio" / "long_ir44.wav"

STRUDEL_JSON_URL = "https://software.tomandandy.com/strudel.json"
NUM_FILES = 4
SLICES_PER_FILE = 3
FORCED_SOURCE_NAME = None
MIN_SLICE_SECONDS = 0.80
MAX_SLICE_SECONDS = 2.20
SLICE_FADE_IN_SECONDS = 3.0
SLICE_FADE_OUT_SECONDS = 6.0
TRALFAM_TAIL_SECONDS = 4.0
TRALFAM_LOOP_COUNT = 5
TRALFAM_NORMALIZE_PEAK = 0.63
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
OUTPUT_PATH = Path(__file__).with_name("tralfam_etude4_render.wav")

# Tonic estimation + bass layer controls.
ANALYSIS_WINDOW_SECONDS = 2.5
MIN_TONIC_HZ = 40.0
MAX_TONIC_HZ = 850.0
PERIODICITY_THRESHOLD = 0.10
DEFAULT_TONIC_HZ = 55.0

BASS_MIN_HZ = 38.0
BASS_MAX_HZ = 96.0
BASS_GAIN = 0.18
BASS_GLOBAL_GAIN = 0.14
BASS_VOICES = 7
BASS_DETUNE_CENTS = 14.0
BASS_LPF_MIN_CUTOFF_HZ = 120.0
BASS_LPF_MAX_CUTOFF_HZ = 6000.0
BASS_LPF_ENV_CURVE = 0.5
BASS_LPF_ENV_ATTACK_SECONDS = 0.01
BASS_LPF_ENV_RELEASE_SECONDS = 0.06
BASS_LPF_Q = 2.07
BASS_ATTACK_SECONDS = 0.02
BASS_RELEASE_SECONDS = 0.10


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


def make_spatialized_slice_pair(pe):
    """Create a simultaneous stereo pair, hard-panned left and right."""
    mono = pg.SpatialPE(pe, method=pg.SpatialAdapter(channels=1))
    left = pg.SpatialPE(mono, method=pg.SpatialLinear(azimuth=-90.0))
    right = pg.SpatialPE(mono, method=pg.SpatialLinear(azimuth=90.0))
    return pg.MixPE(left, right)


def _estimate_tonic_hz_from_pe(pe):
    """
    Estimate tonic/root frequency from periodicity using autocorrelation.

    Returns:
        (tonic_hz, periodicity_score)
    """
    duration = pe.extent().end
    if duration <= 0:
        return DEFAULT_TONIC_HZ, 0.0

    window = min(int(round(ANALYSIS_WINDOW_SECONDS * SAMPLE_RATE)), duration)
    if window < 256:
        return DEFAULT_TONIC_HZ, 0.0

    # Analyze near the center where the blur + fades are usually most stable.
    start = max(0, (duration - window) // 2)
    snippet = pe.render(start, window)
    samples = snippet.data.astype(np.float64)
    if samples.ndim == 1:
        mono = samples
    else:
        mono = samples.mean(axis=1)

    mono = mono - np.mean(mono)
    rms = float(np.sqrt(np.mean(mono * mono)))
    if rms < 1e-5:
        return DEFAULT_TONIC_HZ, 0.0

    # Windowing reduces edge artifacts in autocorrelation.
    mono = mono * np.hanning(window)
    fft_size = 1 << (2 * window - 1).bit_length()
    spectrum = np.fft.rfft(mono, n=fft_size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[:window]
    if autocorr[0] <= 0:
        return DEFAULT_TONIC_HZ, 0.0
    autocorr = autocorr / autocorr[0]

    min_lag = max(1, int(SAMPLE_RATE / MAX_TONIC_HZ))
    max_lag = min(window - 1, int(SAMPLE_RATE / MIN_TONIC_HZ))
    if max_lag <= min_lag:
        return DEFAULT_TONIC_HZ, 0.0

    lag_band = autocorr[min_lag : max_lag + 1]
    best_idx = int(np.argmax(lag_band))
    best_lag = min_lag + best_idx
    periodicity = float(lag_band[best_idx])

    autocorr_tonic_hz = SAMPLE_RATE / float(best_lag)
    autocorr_tonic_hz = float(np.clip(autocorr_tonic_hz, MIN_TONIC_HZ, MAX_TONIC_HZ))

    # Fallback: use dominant spectral bin in the tonic search range.
    # This avoids collapsing many noisy slices to one fixed default pitch.
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / SAMPLE_RATE)
    mags = np.abs(spectrum)
    band = np.where((freqs >= MIN_TONIC_HZ) & (freqs <= MAX_TONIC_HZ))[0]
    if band.size > 0:
        peak_idx = int(band[np.argmax(mags[band])])
        spectral_tonic_hz = float(freqs[peak_idx])
    else:
        spectral_tonic_hz = DEFAULT_TONIC_HZ

    if periodicity >= PERIODICITY_THRESHOLD:
        return autocorr_tonic_hz, periodicity
    return spectral_tonic_hz, periodicity


def _fold_to_bass_register(freq_hz):
    """Fold a frequency into a low-bass octave range."""
    freq = float(max(1.0, freq_hz))
    while freq > BASS_MAX_HZ:
        freq *= 0.5
    while freq < BASS_MIN_HZ:
        freq *= 2.0
    return freq


def _make_bass_supersaw(tonic_hz, duration, rng):
    """Create a short-lived bass supersaw note matched to the slice duration."""
    bass_hz = _fold_to_bass_register(tonic_hz)
    bass = pg.SuperSawPE(
        frequency=bass_hz,
        amplitude=BASS_GAIN,
        voices=BASS_VOICES,
        detune_cents=BASS_DETUNE_CENTS,
        mix_mode=pg.SuperSawPE.MIX_CENTER_HEAVY,
        channels=1,
        seed=rng.randint(0, 2**31 - 1),
    )
    bass = pg.CropPE(bass, 0, duration)

    def env_to_cutoff(env):
        env = np.clip(env, 0.0, 1.0)
        return BASS_LPF_MIN_CUTOFF_HZ + (
            BASS_LPF_MAX_CUTOFF_HZ - BASS_LPF_MIN_CUTOFF_HZ
        ) * (env ** BASS_LPF_ENV_CURVE)

    cutoff_control = pg.TransformPE(
        pg.EnvelopePE(
            bass,
            attack=BASS_LPF_ENV_ATTACK_SECONDS,
            release=BASS_LPF_ENV_RELEASE_SECONDS,
            mode=pg.DetectionMode.PEAK,
        ),
        func=env_to_cutoff,
        name="bass_env_to_cutoff",
    )

    bass = pg.BiquadPE(
        bass,
        frequency=cutoff_control,
        q=BASS_LPF_Q,
        mode=pg.BiquadMode.LOWPASS,
    )

    attack = min(int(round(BASS_ATTACK_SECONDS * SAMPLE_RATE)), duration)
    release = min(int(round(BASS_RELEASE_SECONDS * SAMPLE_RATE)), duration)
    sustain_end = max(attack, duration - release)
    env = pg.PiecewisePE(
        [
            (0, 0.0),
            (attack, 1.0),
            (sustain_end, 0.90),
            (duration, 0.0),
        ],
        transition_type=pg.TransitionType.LINEAR,
    )
    bass = pg.GainPE(bass, gain=env)
    bass = pg.GainPE(bass, gain=BASS_GLOBAL_GAIN)
    return bass, bass_hz


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
    """Extract, blur, ramp, estimate tonic, and annotate slices."""
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
            tonic_hz, periodicity = _estimate_tonic_hz_from_pe(ramped)
            bass_hz = _fold_to_bass_register(tonic_hz)
            processed.append(
                {
                    "source_name": source_info["sound_name"],
                    "slice_num": slice_num + 1,
                    "start": start,
                    "raw_duration": duration,
                    "pe": ramped,
                    "base_duration": ramped.extent().end,
                    "tonic_hz": tonic_hz,
                    "periodicity": periodicity,
                    "bass_hz": bass_hz,
                }
            )
            print(
                f"  slice {slice_num + 1}: start={start / file_sample_rate:.2f}s, "
                f"raw_dur={duration / file_sample_rate:.2f}s, "
                f"blurred_dur={ramped.extent().end / file_sample_rate:.2f}s, "
                f"tonic={tonic_hz:.2f}Hz, periodicity={periodicity:.3f}, "
                f"bass={bass_hz:.2f}Hz",
                flush=True,
            )
    return processed


def sequence_with_overlap(processed_slices, rng):
    """Shuffle slices, crossfade overlaps, add bass, and mix into one stream."""
    if not processed_slices:
        raise RuntimeError("No processed slices were created.")

    play_order = processed_slices[:]
    rng.shuffle(play_order)

    positioned = []
    cursor = 0
    print("\nSequence order:", flush=True)
    for index, item in enumerate(play_order):
        duration = item["base_duration"]
        fade_in_samples = 0
        fade_out_start = None

        if USE_OVERLAP and index < len(play_order) - 1:
            next_offset = max(1, int(round(duration * (1.0 - OVERLAP_RATIO))))
            overlap_samples = duration - next_offset
            fade_out_start = next_offset
            print(
                f"  {item['source_name']} slice {item['slice_num']} @ {cursor / SAMPLE_RATE:.2f}s "
                f"for {duration / SAMPLE_RATE:.2f}s, "
                f"tonic={item['tonic_hz']:.2f}Hz, bass={item['bass_hz']:.2f}Hz, "
                f"crossfade={overlap_samples / SAMPLE_RATE:.2f}s",
                flush=True,
            )
            cursor_step = next_offset
        else:
            print(
                f"  {item['source_name']} slice {item['slice_num']} @ {cursor / SAMPLE_RATE:.2f}s "
                f"for {duration / SAMPLE_RATE:.2f}s, "
                f"tonic={item['tonic_hz']:.2f}Hz, bass={item['bass_hz']:.2f}Hz",
                flush=True,
            )
            cursor_step = duration

        if USE_OVERLAP and index > 0:
            previous_duration = play_order[index - 1]["base_duration"]
            previous_step = max(1, int(round(previous_duration * (1.0 - OVERLAP_RATIO))))
            fade_in_samples = previous_duration - previous_step

        source_pe = item["pe"]
        if USE_OVERLAP and (fade_in_samples > 0 or fade_out_start is not None):
            source_pe = apply_sequence_crossfade(
                source_pe,
                fade_in_samples=fade_in_samples,
                fade_out_start=fade_out_start,
            )
        paired = make_spatialized_slice_pair(source_pe)

        bass_mono, _ = _make_bass_supersaw(item["tonic_hz"], duration, rng)
        if USE_OVERLAP and (fade_in_samples > 0 or fade_out_start is not None):
            bass_mono = apply_sequence_crossfade(
                bass_mono,
                fade_in_samples=fade_in_samples,
                fade_out_start=fade_out_start,
            )
        bass_stereo = pg.SpatialPE(bass_mono, method=pg.SpatialAdapter(channels=2))

        combined = pg.MixPE(paired, bass_stereo)
        positioned.append(pg.DelayPE(combined, delay=cursor))
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
        f"ir={IR_PATH.name}, "
        f"bass=supersaw",
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
