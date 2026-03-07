"""
tralfam_etude4.py

Build a randomized multi-voice etude from the remote Strudel catalog:
1. for each voice, choose random source files
2. extract random slices from each source
3. blur each slice with the Tralfam chain
4. estimate a tonic/root frequency per slice
5. add a supersaw accompaniment in that voice's configured range
6. sequence each voice with overlap
7. apply compressor + convolution reverb per voice
8. mix voices together and apply a final compressor
9. render the full piece to a WAV file

Usage:
  uv run python examples/tralfam_etude4.py
"""

import random
from pathlib import Path

import numpy as np
import pygmu2 as pg
from pygmu2.annotations import AnnotationRecord, write_annotation_sidecar

RANDOM_SEED = 51
NUM_FILES = 4
SLICES_PER_FILE = 2
REVERB_MIX = 0.76



SAMPLE_RATE = 44100
pg.set_sample_rate(SAMPLE_RATE)

EXAMPLES_DIR = Path(__file__).parent
IR_PATH = EXAMPLES_DIR / "audio" / "long_ir44.wav"

STRUDEL_JSON_URL = "https://software.tomandandy.com/strudel.json"
FORCED_SOURCE_NAME = None
MIN_SLICE_SECONDS = 0.80
MAX_SLICE_SECONDS = 3.20
SLICE_FADE_IN_SECONDS = 6.0
SLICE_FADE_OUT_SECONDS = 12.0
TRALFAM_TAIL_SECONDS = 4.0
TRALFAM_LOOP_COUNT = 5
TRALFAM_NORMALIZE_PEAK = 0.63
OVERLAP_RATIO = 0.30
USE_OVERLAP = True
MASTER_GAIN = 1.0
ENABLE_VOICE_COMPRESSION = True
ENABLE_MASTER_COMPRESSION = True
VOICE_COMPRESSOR_THRESHOLD = -18
VOICE_COMPRESSOR_RATIO = 6
VOICE_COMPRESSOR_ATTACK = 0.04
VOICE_COMPRESSOR_RELEASE = 0.25
VOICE_COMPRESSOR_KNEE = 6.0
MASTER_COMPRESSOR_THRESHOLD = -28
MASTER_COMPRESSOR_RATIO = 4
MASTER_COMPRESSOR_ATTACK = 0.05
MASTER_COMPRESSOR_RELEASE = 0.25
MASTER_COMPRESSOR_KNEE = 6.0
OUTPUT_PATH = Path(__file__).with_name("tralfam_etude4_render.wav")
RENDER_VOICE_STEMS = False
VOICE_STEMS_DIR = OUTPUT_PATH.parent
VOICE_STEMS_PREFIX = OUTPUT_PATH.stem
VOICE_BANDPASS_Q = 3
MIN_CROSSFADE_LPF_FREQ = 180.0
MAX_CROSSFADE_LPF_FREQ = 12000.0
LPF_FADE_IN_SECONDS = 12.0
LPF_FADE_OUT_SECONDS = 12.0
CROSSFADE_LPF_Q = 2.4
CROSSFADE_LPF_POW = 2.0

# Voice-level configuration. Add more entries to create more voices.
VOICE_CONFIGS = [
    {
        "name": "voice_1",
        "num_files": NUM_FILES,
        "slices_per_file": SLICES_PER_FILE,
        "forced_source_name": FORCED_SOURCE_NAME,
        "min_slice_seconds": MIN_SLICE_SECONDS,
        "max_slice_seconds": MAX_SLICE_SECONDS,
        "slice_fade_in_seconds": SLICE_FADE_IN_SECONDS,
        "slice_fade_out_seconds": SLICE_FADE_OUT_SECONDS,
        "tralfam_tail_seconds": TRALFAM_TAIL_SECONDS,
        "tralfam_loop_count": TRALFAM_LOOP_COUNT,
        "tralfam_normalize_peak": TRALFAM_NORMALIZE_PEAK,
        "overlap_ratio": OVERLAP_RATIO,
        "voice_filter_min_hz": 60.0,
        "voice_filter_max_hz": 12000.0,
        "min_crossfade_lpf_freq": MIN_CROSSFADE_LPF_FREQ,
        "max_crossfade_lpf_freq": MAX_CROSSFADE_LPF_FREQ,
        "lpf_fade_in_seconds": LPF_FADE_IN_SECONDS,
        "lpf_fade_out_seconds": LPF_FADE_OUT_SECONDS,
        "voice_gain": 1.0,
        "accompaniment_min_hz": 68.0,
        "accompaniment_max_hz": 196.0,
        "accompaniment_gain": 0.14,
        "reverb_wet_dry": REVERB_MIX,
    },
    {
        "name": "voice_2",
        "num_files": NUM_FILES,
        "slices_per_file": SLICES_PER_FILE,
        "forced_source_name": FORCED_SOURCE_NAME,
        "min_slice_seconds": MIN_SLICE_SECONDS,
        "max_slice_seconds": MAX_SLICE_SECONDS,
        "slice_fade_in_seconds": SLICE_FADE_IN_SECONDS,
        "slice_fade_out_seconds": SLICE_FADE_OUT_SECONDS,
        "tralfam_tail_seconds": TRALFAM_TAIL_SECONDS,
        "tralfam_loop_count": TRALFAM_LOOP_COUNT,
        "tralfam_normalize_peak": TRALFAM_NORMALIZE_PEAK,
        "overlap_ratio": OVERLAP_RATIO,
        "voice_filter_min_hz": 60.0,
        "voice_filter_max_hz": 12000.0,
        "min_crossfade_lpf_freq": MIN_CROSSFADE_LPF_FREQ,
        "max_crossfade_lpf_freq": MAX_CROSSFADE_LPF_FREQ,
        "lpf_fade_in_seconds": LPF_FADE_IN_SECONDS,
        "lpf_fade_out_seconds": LPF_FADE_OUT_SECONDS,
        "voice_gain": 1.0,
        "accompaniment_min_hz": 390.0,
        "accompaniment_max_hz": 1260.0,
        "accompaniment_gain": 0.07,
        "reverb_wet_dry": REVERB_MIX,
    },
    {
        "name": "voice_3",
        "num_files": NUM_FILES * 2,
        "slices_per_file": SLICES_PER_FILE / 2,
        "forced_source_name": FORCED_SOURCE_NAME,
        "min_slice_seconds": MIN_SLICE_SECONDS / 2,
        "max_slice_seconds": MAX_SLICE_SECONDS / 2,
        "slice_fade_in_seconds": SLICE_FADE_IN_SECONDS,
        "slice_fade_out_seconds": SLICE_FADE_OUT_SECONDS,
        "tralfam_tail_seconds": TRALFAM_TAIL_SECONDS,
        "tralfam_loop_count": TRALFAM_LOOP_COUNT * 2,
        "tralfam_normalize_peak": TRALFAM_NORMALIZE_PEAK,
        "overlap_ratio": OVERLAP_RATIO,
        "voice_filter_min_hz": 560.0,
        "voice_filter_max_hz": 900.0,
        "min_crossfade_lpf_freq": 2000,
        "max_crossfade_lpf_freq": 5000,
        "lpf_fade_in_seconds": LPF_FADE_IN_SECONDS,
        "lpf_fade_out_seconds": LPF_FADE_OUT_SECONDS,
        "voice_gain": 1.0,
        "accompaniment_min_hz": 1390.0,
        "accompaniment_max_hz": 3260.0,
        "accompaniment_gain": 0.05,
        "reverb_wet_dry": REVERB_MIX,
    },
]

# Tonic estimation + accompaniment controls.
ANALYSIS_WINDOW_SECONDS = 2.5
MIN_TONIC_HZ = 40.0
MAX_TONIC_HZ = 850.0
PERIODICITY_THRESHOLD = 0.10
DEFAULT_TONIC_HZ = 55.0

BASS_GAIN = 0.18
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


def _log_constraint(voice_name, message):
    print(f"  [constraint][{voice_name}] {message}", flush=True)


def _sanitize_voice_config(raw_config, voice_index):
    """Return a validated voice config, logging every constrained value."""
    cfg = dict(raw_config)
    voice_name = str(cfg.get("name", f"voice_{voice_index + 1}"))
    cfg["name"] = voice_name

    def as_int(key, default, minimum=1):
        value = cfg.get(key, default)
        try:
            value = int(value)
        except (TypeError, ValueError):
            _log_constraint(voice_name, f"{key}={value!r} invalid; using {default}")
            value = int(default)
        if value < minimum:
            _log_constraint(voice_name, f"{key}={value} too low; clamped to {minimum}")
            value = minimum
        return value

    def as_float(key, default, minimum=None, maximum=None):
        value = cfg.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            _log_constraint(voice_name, f"{key}={value!r} invalid; using {default}")
            value = float(default)
        original = value
        if minimum is not None and value < minimum:
            value = minimum
        if maximum is not None and value > maximum:
            value = maximum
        if value != original:
            _log_constraint(voice_name, f"{key}={original} constrained to {value}")
        return value

    cfg["num_files"] = as_int("num_files", NUM_FILES, minimum=1)
    cfg["slices_per_file"] = as_int("slices_per_file", SLICES_PER_FILE, minimum=1)
    cfg["forced_source_name"] = cfg.get("forced_source_name", FORCED_SOURCE_NAME)

    cfg["min_slice_seconds"] = as_float("min_slice_seconds", MIN_SLICE_SECONDS, minimum=0.01)
    cfg["max_slice_seconds"] = as_float("max_slice_seconds", MAX_SLICE_SECONDS, minimum=0.01)
    if cfg["max_slice_seconds"] < cfg["min_slice_seconds"]:
        _log_constraint(
            voice_name,
            "max_slice_seconds < min_slice_seconds; setting max_slice_seconds=min_slice_seconds",
        )
        cfg["max_slice_seconds"] = cfg["min_slice_seconds"]

    cfg["slice_fade_in_seconds"] = as_float("slice_fade_in_seconds", SLICE_FADE_IN_SECONDS, minimum=0.0)
    cfg["slice_fade_out_seconds"] = as_float("slice_fade_out_seconds", SLICE_FADE_OUT_SECONDS, minimum=0.0)
    cfg["tralfam_tail_seconds"] = as_float("tralfam_tail_seconds", TRALFAM_TAIL_SECONDS, minimum=0.0)
    cfg["tralfam_loop_count"] = as_int("tralfam_loop_count", TRALFAM_LOOP_COUNT, minimum=1)
    cfg["tralfam_normalize_peak"] = as_float("tralfam_normalize_peak", TRALFAM_NORMALIZE_PEAK, minimum=1e-6)
    cfg["overlap_ratio"] = as_float("overlap_ratio", OVERLAP_RATIO, minimum=0.0, maximum=0.95)

    cfg["voice_filter_min_hz"] = as_float("voice_filter_min_hz", 60.0, minimum=20.0)
    cfg["voice_filter_max_hz"] = as_float("voice_filter_max_hz", 12000.0, minimum=20.0)
    if cfg["voice_filter_max_hz"] <= cfg["voice_filter_min_hz"]:
        corrected = cfg["voice_filter_min_hz"] * 2.0
        _log_constraint(
            voice_name,
            f"voice_filter_max_hz <= voice_filter_min_hz; setting voice_filter_max_hz={corrected}",
        )
        cfg["voice_filter_max_hz"] = corrected

    cfg["min_crossfade_lpf_freq"] = as_float(
        "min_crossfade_lpf_freq",
        MIN_CROSSFADE_LPF_FREQ,
        minimum=20.0,
    )
    cfg["max_crossfade_lpf_freq"] = as_float(
        "max_crossfade_lpf_freq",
        MAX_CROSSFADE_LPF_FREQ,
        minimum=20.0,
    )
    if cfg["max_crossfade_lpf_freq"] <= cfg["min_crossfade_lpf_freq"]:
        corrected = cfg["min_crossfade_lpf_freq"] * 2.0
        _log_constraint(
            voice_name,
            "max_crossfade_lpf_freq <= min_crossfade_lpf_freq; "
            f"setting max_crossfade_lpf_freq={corrected}",
        )
        cfg["max_crossfade_lpf_freq"] = corrected
    cfg["lpf_fade_in_seconds"] = as_float(
        "lpf_fade_in_seconds",
        LPF_FADE_IN_SECONDS,
        minimum=0.0,
    )
    cfg["lpf_fade_out_seconds"] = as_float(
        "lpf_fade_out_seconds",
        LPF_FADE_OUT_SECONDS,
        minimum=0.0,
    )

    cfg["accompaniment_min_hz"] = as_float("accompaniment_min_hz", 38.0, minimum=20.0)
    cfg["accompaniment_max_hz"] = as_float("accompaniment_max_hz", 96.0, minimum=20.0)
    if cfg["accompaniment_max_hz"] <= cfg["accompaniment_min_hz"]:
        corrected = cfg["accompaniment_min_hz"] * 2.0
        _log_constraint(
            voice_name,
            f"accompaniment_max_hz <= accompaniment_min_hz; setting accompaniment_max_hz={corrected}",
        )
        cfg["accompaniment_max_hz"] = corrected

    cfg["accompaniment_gain"] = as_float("accompaniment_gain", 0.1, minimum=0.0, maximum=4.0)
    cfg["voice_gain"] = as_float("voice_gain", 1.0, minimum=0.0, maximum=4.0)
    cfg["reverb_wet_dry"] = as_float(
        "reverb_wet_dry",
        cfg.get("reverb_mix", REVERB_MIX),
        minimum=0.0,
        maximum=1.0,
    )

    return cfg


def _safe_stem_token(text):
    token = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(text))
    token = token.strip("_")
    return token or "voice"


def catalog_wav_entries(library):
    """Return concrete (sound name, variant index, relative path) WAV entries."""
    entries = []
    for sound_name, variants in library._audio_paths.items():
        for index, rel_path in enumerate(variants):
            if str(rel_path).lower().endswith(".wav"):
                entries.append((sound_name, index, rel_path))
    return entries


def choose_sources(library, rng, num_files=NUM_FILES, forced_source_name=FORCED_SOURCE_NAME):
    """Resolve random catalog entries to local WAV paths and readers."""
    entries = catalog_wav_entries(library)
    if not entries:
        raise RuntimeError("No WAV entries found in the Strudel catalog.")

    if forced_source_name is not None:
        entries = [entry for entry in entries if entry[0] == forced_source_name]
        if not entries:
            raise RuntimeError(f"Forced source {forced_source_name!r} not found in the catalog.")

    if len(entries) < num_files:
        raise RuntimeError(
            f"Requested {num_files} files, but catalog only has {len(entries)} WAV entries."
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
        if len(chosen) == num_files:
            return chosen

    raise RuntimeError(
        f"Only found {len(chosen)} WAV files at {SAMPLE_RATE} Hz in the Strudel catalog; "
        f"need {num_files}."
    )


def make_random_slice(reader, rng, voice_config):
    """Extract a random slice from a file with adjustable duration bounds."""
    voice_name = voice_config["name"]
    file_frames = reader.extent().end
    file_sample_rate = reader.file_sample_rate
    min_duration = int(round(voice_config["min_slice_seconds"] * file_sample_rate))
    max_duration = int(round(voice_config["max_slice_seconds"] * file_sample_rate))

    if file_frames <= 0:
        raise RuntimeError(f"Cannot slice empty file: {reader.path}")

    if min_duration < 1:
        _log_constraint(voice_name, f"min slice duration {min_duration} samples -> 1 sample")
        min_duration = 1
    if max_duration < 1:
        _log_constraint(voice_name, f"max slice duration {max_duration} samples -> 1 sample")
        max_duration = 1
    if min_duration > max_duration:
        _log_constraint(
            voice_name,
            f"min_duration ({min_duration}) > max_duration ({max_duration}); using max_duration",
        )
        min_duration = max_duration

    max_allowed_duration = min(max_duration, file_frames)
    min_allowed_duration = min(min_duration, max_allowed_duration)
    if max_duration > file_frames:
        _log_constraint(
            voice_name,
            f"requested max slice {max_duration / file_sample_rate:.2f}s exceeds source "
            f"{file_frames / file_sample_rate:.2f}s; clamped to source length",
        )
    if min_duration > max_allowed_duration:
        _log_constraint(
            voice_name,
            f"requested min slice {min_duration / file_sample_rate:.2f}s exceeds available "
            f"{max_allowed_duration / file_sample_rate:.2f}s; clamped",
        )

    duration = rng.randint(min_allowed_duration, max_allowed_duration)
    start = rng.randint(0, file_frames - duration)

    slice_pe = pg.SlicePE(
        reader,
        start,
        duration,
    )
    return slice_pe, start, duration


def apply_post_blur_fade(pe, voice_config):
    """Apply an explicit gain envelope after the blur stage."""
    voice_name = voice_config["name"]
    duration = pe.extent().end
    fade_in = min(int(round(voice_config["slice_fade_in_seconds"] * SAMPLE_RATE)), duration)
    fade_out = min(int(round(voice_config["slice_fade_out_seconds"] * SAMPLE_RATE)), duration)

    if duration <= 0:
        raise RuntimeError("Slice has no duration.")

    if fade_in + fade_out > duration:
        requested_in = fade_in
        requested_out = fade_out
        total = max(1, fade_in + fade_out)
        scale = duration / total
        fade_in = int(round(fade_in * scale))
        fade_in = max(0, min(fade_in, duration))
        fade_out = max(0, duration - fade_in)
        _log_constraint(
            voice_name,
            "fade_in + fade_out exceeded slice duration; "
            f"scaled from ({requested_in / SAMPLE_RATE:.2f}s, {requested_out / SAMPLE_RATE:.2f}s) "
            f"to ({fade_in / SAMPLE_RATE:.2f}s, {fade_out / SAMPLE_RATE:.2f}s)",
        )

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


def apply_sequence_crossfade_lpf(pe, voice_config):
    """Apply LPF sweep per slice-pair: low->high on fade-in, high->low on fade-out."""
    voice_name = voice_config["name"]
    duration = pe.extent().end
    if duration <= 0:
        raise RuntimeError("Slice has no duration.")

    min_freq = float(voice_config["min_crossfade_lpf_freq"])
    max_freq = float(voice_config["max_crossfade_lpf_freq"])
    fade_in = min(int(round(voice_config["lpf_fade_in_seconds"] * SAMPLE_RATE)), duration)
    fade_out = min(int(round(voice_config["lpf_fade_out_seconds"] * SAMPLE_RATE)), duration)

    if fade_in + fade_out > duration:
        requested_in = fade_in
        requested_out = fade_out
        total = max(1, fade_in + fade_out)
        scale = duration / total
        fade_in = int(round(fade_in * scale))
        fade_in = max(0, min(fade_in, duration))
        fade_out = max(0, duration - fade_in)
        _log_constraint(
            voice_name,
            "lpf_fade_in_seconds + lpf_fade_out_seconds exceeded slice duration; "
            f"scaled from ({requested_in / SAMPLE_RATE:.2f}s, {requested_out / SAMPLE_RATE:.2f}s) "
            f"to ({fade_in / SAMPLE_RATE:.2f}s, {fade_out / SAMPLE_RATE:.2f}s)",
        )

    sustain_end = max(fade_in, duration - fade_out)
    shape_env = pg.PiecewisePE(
        [
            (0, 0.0),
            (fade_in, 1.0),
            (sustain_end, 1.0),
            (duration, 0.0),
        ],
        transition_type=pg.TransitionType.LINEAR,
    )
    cutoff_env = pg.TransformPE(
        shape_env,
        func=lambda x: min_freq + (max_freq - min_freq) * np.power(np.clip(x, 0.0, 1.0), CROSSFADE_LPF_POW),
        name="crossfade_lpf_pow2",
    )
    return pg.BiquadPE(
        pe,
        frequency=cutoff_env,
        q=CROSSFADE_LPF_Q,
        mode=pg.BiquadMode.LOWPASS,
    )


def make_blurry_slice(slice_pe, slice_duration, rng, voice_config):
    """Blur one raw slice via SetExtentPE -> TralfamPE -> LoopPE."""
    voice_name = voice_config["name"]
    tail_samples = int(round(voice_config["tralfam_tail_seconds"] * slice_pe.sample_rate))
    if tail_samples < 0:
        _log_constraint(voice_name, f"tralfam_tail_seconds produced {tail_samples} samples; clamped to 0")
        tail_samples = 0
    padded_duration = slice_duration + tail_samples
    if padded_duration < slice_duration:
        _log_constraint(
            voice_name,
            f"padded_duration ({padded_duration}) < slice_duration ({slice_duration}); clamped",
        )
        padded_duration = slice_duration
    padded = pg.SetExtentPE(slice_pe, 0, padded_duration)
    tralfam = pg.TralfamPE(
        padded,
        seed=rng.randint(0, 2**31 - 1),
        normalize_peak=voice_config["tralfam_normalize_peak"],
    )
    return pg.LoopPE(tralfam, count=voice_config["tralfam_loop_count"])


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


def _fold_to_voice_register(freq_hz, min_hz, max_hz):
    """Fold a frequency into a voice-defined octave range."""
    freq = float(max(1.0, freq_hz))
    while freq > max_hz:
        freq *= 0.5
    while freq < min_hz:
        freq *= 2.0
    return freq


def _make_accompaniment_supersaw(tonic_hz, duration, rng, voice_config):
    """Create a short-lived supersaw accompaniment for one slice."""
    low = voice_config["accompaniment_min_hz"]
    high = voice_config["accompaniment_max_hz"]
    voice_gain = voice_config["accompaniment_gain"]
    bass_hz = _fold_to_voice_register(tonic_hz, low, high)
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
    bass = pg.GainPE(bass, gain=voice_gain)
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


def collect_processed_slices(sources, rng, voice_config):
    """Extract, blur, ramp, estimate tonic, and annotate slices."""
    processed = []
    low = voice_config["accompaniment_min_hz"]
    high = voice_config["accompaniment_max_hz"]
    for source_info in sources:
        file_sample_rate = source_info["file_sample_rate"]
        print(
            f"Source: {source_info['sound_name']} ({source_info['catalog_path']}) "
            f"{source_info['frames'] / file_sample_rate:.2f}s",
            flush=True,
        )
        for slice_num in range(voice_config["slices_per_file"]):
            slice_pe, start, duration = make_random_slice(source_info["reader"], rng, voice_config)
            blurry = make_blurry_slice(slice_pe, duration, rng, voice_config)
            ramped = apply_post_blur_fade(blurry, voice_config)
            tonic_hz, periodicity = _estimate_tonic_hz_from_pe(ramped)
            bass_hz = _fold_to_voice_register(tonic_hz, low, high)
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


def sequence_with_overlap(processed_slices, rng, voice_config):
    """Shuffle slices, crossfade overlaps, add accompaniment, and mix."""
    if not processed_slices:
        raise RuntimeError("No processed slices were created.")

    play_order = processed_slices[:]
    rng.shuffle(play_order)

    overlap_ratio = voice_config["overlap_ratio"]
    voice_name = voice_config["name"]
    if overlap_ratio >= 1.0:
        _log_constraint(voice_name, f"overlap_ratio={overlap_ratio} invalid; constrained to 0.95")
        overlap_ratio = 0.95

    positioned = []
    annotations = []
    cursor = 0
    print("\nSequence order:", flush=True)
    for index, item in enumerate(play_order):
        duration = item["base_duration"]
        fade_in_samples = 0
        fade_out_start = None

        if USE_OVERLAP and index < len(play_order) - 1:
            next_offset = max(1, int(round(duration * (1.0 - overlap_ratio))))
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
            crossfade_seconds = overlap_samples / SAMPLE_RATE
        else:
            print(
                f"  {item['source_name']} slice {item['slice_num']} @ {cursor / SAMPLE_RATE:.2f}s "
                f"for {duration / SAMPLE_RATE:.2f}s, "
                f"tonic={item['tonic_hz']:.2f}Hz, bass={item['bass_hz']:.2f}Hz",
                flush=True,
            )
            cursor_step = duration
            crossfade_seconds = 0.0

        if USE_OVERLAP and index > 0:
            previous_duration = play_order[index - 1]["base_duration"]
            previous_step = max(1, int(round(previous_duration * (1.0 - overlap_ratio))))
            fade_in_samples = previous_duration - previous_step

        source_pe = item["pe"]
        if USE_OVERLAP and (fade_in_samples > 0 or fade_out_start is not None):
            source_pe = apply_sequence_crossfade(
                source_pe,
                fade_in_samples=fade_in_samples,
                fade_out_start=fade_out_start,
            )
        paired = make_spatialized_slice_pair(source_pe)

        bass_mono, _ = _make_accompaniment_supersaw(
            item["tonic_hz"], duration, rng, item["voice_config"]
        )
        if USE_OVERLAP and (fade_in_samples > 0 or fade_out_start is not None):
            bass_mono = apply_sequence_crossfade(
                bass_mono,
                fade_in_samples=fade_in_samples,
                fade_out_start=fade_out_start,
            )
        bass_stereo = pg.SpatialPE(bass_mono, method=pg.SpatialAdapter(channels=2))

        combined = pg.MixPE(paired, bass_stereo)
        combined = apply_sequence_crossfade_lpf(combined, voice_config)
        positioned.append(pg.DelayPE(combined, delay=cursor))
        annotations.append(
            AnnotationRecord(
                id=f"{voice_name}_{index + 1:04d}",
                onset=cursor / SAMPLE_RATE,
                extent=duration / SAMPLE_RATE,
                label=f"{voice_name} {item['source_name']}#{item['slice_num']}",
                kind="slice",
                voice=voice_name,
                meta={
                    "source_name": item["source_name"],
                    "slice_num": item["slice_num"],
                    "tonic_hz": float(item["tonic_hz"]),
                    "periodicity": float(item["periodicity"]),
                    "bass_hz": float(item["bass_hz"]),
                    "duration_seconds": duration / SAMPLE_RATE,
                    "crossfade_seconds": float(crossfade_seconds),
                },
            )
        )
        cursor += cursor_step

    return pg.MixPE(*positioned), annotations


def build_voice(library, ir, rng, voice_config):
    """Build one complete voice: slices -> accompaniment -> compressor -> convolution."""
    voice_name = voice_config["name"]
    reverb_wet_dry = float(voice_config.get("reverb_wet_dry", voice_config.get("reverb_mix", REVERB_MIX)))
    reverb_wet_dry = min(1.0, max(0.0, reverb_wet_dry))
    print(
        f"\nBuilding {voice_name}: files={voice_config['num_files']}, "
        f"slices_per_file={voice_config['slices_per_file']}, "
        f"slice_dur={voice_config['min_slice_seconds']:.2f}-{voice_config['max_slice_seconds']:.2f}s, "
        f"overlap={voice_config['overlap_ratio']:.0%}, "
        f"voice_band={voice_config['voice_filter_min_hz']:.1f}-"
        f"{voice_config['voice_filter_max_hz']:.1f}Hz, "
        f"crossfade_lpf={voice_config['min_crossfade_lpf_freq']:.1f}-"
        f"{voice_config['max_crossfade_lpf_freq']:.1f}Hz "
        f"(in={voice_config['lpf_fade_in_seconds']:.2f}s, "
        f"out={voice_config['lpf_fade_out_seconds']:.2f}s), "
        f"accompaniment_range={voice_config['accompaniment_min_hz']:.1f}-"
        f"{voice_config['accompaniment_max_hz']:.1f}Hz, "
        f"accompaniment_gain={voice_config['accompaniment_gain']:.3f}, "
        f"voice_gain={voice_config['voice_gain']:.3f}, "
        f"reverb_wet_dry={reverb_wet_dry:.2f}",
        flush=True,
    )
    sources = choose_sources(
        library,
        rng,
        num_files=voice_config["num_files"],
        forced_source_name=voice_config["forced_source_name"],
    )
    processed_slices = collect_processed_slices(sources, rng, voice_config)
    for item in processed_slices:
        item["voice_config"] = voice_config

    voice, annotations = sequence_with_overlap(processed_slices, rng, voice_config)
    voice = pg.BiquadPE(
        voice,
        frequency=voice_config["voice_filter_min_hz"],
        q=VOICE_BANDPASS_Q,
        mode=pg.BiquadMode.HIGHPASS,
    )
    voice = pg.BiquadPE(
        voice,
        frequency=voice_config["voice_filter_max_hz"],
        q=VOICE_BANDPASS_Q,
        mode=pg.BiquadMode.LOWPASS,
    )
    if ENABLE_VOICE_COMPRESSION:
        voice = pg.CompressorPE(
            voice,
            threshold=VOICE_COMPRESSOR_THRESHOLD,
            ratio=VOICE_COMPRESSOR_RATIO,
            attack=VOICE_COMPRESSOR_ATTACK,
            release=VOICE_COMPRESSOR_RELEASE,
            knee=VOICE_COMPRESSOR_KNEE,
            makeup_gain="auto",
        )
    voice = pg.ReverbPE(voice, ir, mix=reverb_wet_dry, normalize_ir=True)
    voice = pg.GainPE(voice, gain=voice_config["voice_gain"])
    return voice, annotations


def build_etude(seed=RANDOM_SEED, return_voice_stems=False, return_annotations=False):
    rng = random.Random(seed)

    print(f"Fetching catalog: {STRUDEL_JSON_URL}", flush=True)
    library = pg.AudioLibrary.from_url(STRUDEL_JSON_URL)
    ir = load_reverb_ir()

    print(
        f"Building {len(VOICE_CONFIGS)} voices, "
        f"{'overlap enabled' if USE_OVERLAP else 'no overlap'}, "
        f"voice_comp={'on' if ENABLE_VOICE_COMPRESSION else 'off'}, "
        f"master_comp={'on' if ENABLE_MASTER_COMPRESSION else 'off'}, "
        f"ir={IR_PATH.name}",
        flush=True,
    )

    if not VOICE_CONFIGS:
        raise RuntimeError("VOICE_CONFIGS is empty; add at least one voice config.")
    sanitized_voice_configs = [
        _sanitize_voice_config(voice_config, index) for index, voice_config in enumerate(VOICE_CONFIGS)
    ]
    voice_results = [build_voice(library, ir, rng, voice_config) for voice_config in sanitized_voice_configs]
    voices = [voice_result[0] for voice_result in voice_results]
    annotations = [item for _, ann in voice_results for item in ann]
    mixed = pg.MixPE(*voices)
    if ENABLE_MASTER_COMPRESSION:
        mixed = pg.CompressorPE(
            mixed,
            threshold=MASTER_COMPRESSOR_THRESHOLD,
            ratio=MASTER_COMPRESSOR_RATIO,
            attack=MASTER_COMPRESSOR_ATTACK,
            release=MASTER_COMPRESSOR_RELEASE,
            knee=MASTER_COMPRESSOR_KNEE,
            makeup_gain="auto",
        )
    master = pg.GainPE(mixed, gain=MASTER_GAIN)
    if return_voice_stems and return_annotations:
        return master, list(zip(sanitized_voice_configs, voices)), annotations
    if return_voice_stems:
        return master, list(zip(sanitized_voice_configs, voices))
    if return_annotations:
        return master, annotations
    return master


def main():
    build_result = build_etude(
        seed=RANDOM_SEED,
        return_voice_stems=RENDER_VOICE_STEMS,
        return_annotations=True,
    )
    if RENDER_VOICE_STEMS:
        etude, voice_stems, annotations = build_result
    else:
        etude, annotations = build_result
        voice_stems = []

    if RENDER_VOICE_STEMS:
        VOICE_STEMS_DIR.mkdir(parents=True, exist_ok=True)
        print("\nRendering per-voice stems...", flush=True)
        for index, (voice_config, voice_pe) in enumerate(voice_stems, start=1):
            voice_name = _safe_stem_token(voice_config["name"])
            stem_path = VOICE_STEMS_DIR / f"{VOICE_STEMS_PREFIX}_{index:02d}_{voice_name}.wav"
            stem_duration = voice_pe.extent().end / SAMPLE_RATE
            print(
                f"  Rendering stem {index}: {voice_config['name']} "
                f"({stem_duration:.2f}s) -> {stem_path}",
                flush=True,
            )
            pg.render_to_file(voice_pe, str(stem_path), sample_rate=SAMPLE_RATE)

    total_duration = etude.extent().end / SAMPLE_RATE
    print(f"\nRendering etude ({total_duration:.2f} seconds)...", flush=True)
    pg.render_to_file(etude, str(OUTPUT_PATH), sample_rate=SAMPLE_RATE)
    sidecar_path = write_annotation_sidecar(
        OUTPUT_PATH,
        annotations=annotations,
        sample_rate=SAMPLE_RATE,
        total_frames=int(etude.extent().end),
    )
    print(f"Wrote {len(annotations)} annotations to {sidecar_path}", flush=True)
    print(f"Rendered to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
