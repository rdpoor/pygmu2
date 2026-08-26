"""
Offline noFFT spectrogram analysis helpers.

This module adapts the resonator-bank approach for file-based spectrogram
generation in desktop UIs where a seekable timeline view is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import soundfile as sf

FrequencyScale = Literal["musical", "logarithmic", "linear", "mel"]


@dataclass(frozen=True)
class NoFFTSpectrogramConfig:
    """Configuration for offline noFFT spectrogram generation."""

    min_freq: float = 100.0
    max_freq: float = 16000.0
    bins_per_semitone: float = 2.0
    frequency_scale: FrequencyScale = "musical"
    cycles_for_decay: float = 5.0
    hop_size: int = 128
    min_db: float = -80.0
    max_db: float = 0.0


@dataclass(frozen=True)
class NoFFTSpectrogramResult:
    """Result payload for spectrogram rendering."""

    intensities: np.ndarray
    frequencies_hz: np.ndarray
    sample_rate: int
    hop_size: int
    min_db: float
    max_db: float


def _generate_musical_frequencies(
    min_freq: float,
    max_freq: float,
    bins_per_semitone: float,
) -> np.ndarray:
    ratio = 2.0 ** (1.0 / (12.0 * bins_per_semitone))
    freqs: list[float] = []
    f = float(min_freq)
    while f <= max_freq:
        freqs.append(f)
        f *= ratio
    if not freqs:
        freqs = [min_freq]
    return np.asarray(freqs, dtype=np.float32)


def _generate_log_frequencies(
    min_freq: float,
    max_freq: float,
    bins_per_semitone: float,
) -> np.ndarray:
    bins_per_octave = max(1.0, bins_per_semitone * 12.0)
    ratio = 2.0 ** (1.0 / bins_per_octave)
    freqs: list[float] = []
    f = float(min_freq)
    while f <= max_freq:
        freqs.append(f)
        f *= ratio
    if not freqs:
        freqs = [min_freq]
    return np.asarray(freqs, dtype=np.float32)


def _generate_linear_frequencies(
    min_freq: float, max_freq: float, num_bins: int
) -> np.ndarray:
    num_bins = max(2, num_bins)
    return np.linspace(min_freq, max_freq, num_bins, dtype=np.float32)


def _generate_mel_frequencies(
    min_freq: float, max_freq: float, num_bins: int
) -> np.ndarray:
    num_bins = max(2, num_bins)

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    min_mel = hz_to_mel(min_freq)
    max_mel = hz_to_mel(max_freq)
    mel_values = np.linspace(min_mel, max_mel, num_bins, dtype=np.float32)
    hz_values = np.asarray(
        [mel_to_hz(float(mel)) for mel in mel_values], dtype=np.float32
    )
    return hz_values


def generate_frequencies(config: NoFFTSpectrogramConfig) -> np.ndarray:
    """Build per-bin frequencies using the requested scale."""

    min_freq = max(1.0, float(config.min_freq))
    max_freq = max(min_freq + 1.0, float(config.max_freq))
    bins_per_semitone = max(0.25, float(config.bins_per_semitone))
    estimated_bins = int(
        np.ceil(12.0 * bins_per_semitone * np.log2(max_freq / min_freq))
    )
    estimated_bins = max(24, min(estimated_bins, 512))

    if config.frequency_scale == "linear":
        return _generate_linear_frequencies(min_freq, max_freq, estimated_bins)
    if config.frequency_scale == "mel":
        return _generate_mel_frequencies(min_freq, max_freq, estimated_bins)
    if config.frequency_scale == "logarithmic":
        return _generate_log_frequencies(min_freq, max_freq, bins_per_semitone)
    return _generate_musical_frequencies(min_freq, max_freq, bins_per_semitone)


def _alpha_from_frequency(
    freq_hz: np.ndarray, sample_rate: int, cycles_for_decay: float
) -> np.ndarray:
    tau = np.maximum(1e-5, cycles_for_decay / np.maximum(freq_hz, 1e-5))
    return 1.0 - np.exp(-1.0 / (tau * sample_rate))


def _to_mono_float32(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32)


def compute_nofft_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    config: NoFFTSpectrogramConfig,
    target_frames: int | None = None,
    max_analysis_samples: int = 2_000_000,
) -> NoFFTSpectrogramResult:
    """
    Compute a noFFT spectrogram from mono samples.

    Returns intensities as a float32 array with shape [n_bins, n_frames]
    normalized to 0..1 for direct UI colormap mapping.
    """

    x = _to_mono_float32(samples)
    effective_sr = int(sample_rate)

    # Keep offline analysis responsive on long files.
    if x.size > max_analysis_samples:
        stride = int(np.ceil(x.size / max_analysis_samples))
        x = x[::stride]
        effective_sr = max(1, int(sample_rate // stride))

    hop_size = max(1, int(config.hop_size))
    if target_frames and target_frames > 0:
        hop_size = max(hop_size, int(np.ceil(max(1, x.size) / target_frames)))
    freqs = generate_frequencies(config)
    if x.size == 0:
        return NoFFTSpectrogramResult(
            intensities=np.zeros((len(freqs), 1), dtype=np.float32),
            frequencies_hz=freqs,
            sample_rate=sample_rate,
            hop_size=hop_size,
            min_db=config.min_db,
            max_db=config.max_db,
        )

    n_bins = len(freqs)
    n_frames = max(1, x.size // hop_size)

    two_pi_over_sr = np.float32(2.0 * np.pi / effective_sr)
    omega = freqs * two_pi_over_sr
    mult_real = np.cos(omega).astype(np.float32)
    mult_imag = (-np.sin(omega)).astype(np.float32)

    alpha = _alpha_from_frequency(
        freqs.astype(np.float32),
        sample_rate=effective_sr,
        cycles_for_decay=max(0.1, float(config.cycles_for_decay)),
    ).astype(np.float32)
    one_minus_alpha = (1.0 - alpha).astype(np.float32)

    phasor_real = np.ones(n_bins, dtype=np.float32)
    phasor_imag = np.zeros(n_bins, dtype=np.float32)
    resonator_real = np.zeros(n_bins, dtype=np.float32)
    resonator_imag = np.zeros(n_bins, dtype=np.float32)
    matrix = np.zeros((n_bins, n_frames), dtype=np.float32)

    frame_idx = 0
    for idx, sample in enumerate(x):
        p_real = phasor_real.copy()
        p_imag = phasor_imag.copy()
        phasor_real = p_real * mult_real - p_imag * mult_imag
        phasor_imag = p_real * mult_imag + p_imag * mult_real

        resonator_real = one_minus_alpha * resonator_real + alpha * sample * phasor_real
        resonator_imag = one_minus_alpha * resonator_imag + alpha * sample * phasor_imag

        if (idx + 1) % hop_size == 0:
            amps = np.sqrt(
                resonator_real * resonator_real + resonator_imag * resonator_imag
            )
            matrix[:, frame_idx] = amps
            frame_idx += 1
            if frame_idx >= n_frames:
                break

    db = 20.0 * np.log10(matrix + 1e-10)
    db = np.clip(db, config.min_db, config.max_db)
    den = max(1e-6, (config.max_db - config.min_db))
    intensities = (db - config.min_db) / den
    intensities = np.clip(intensities, 0.0, 1.0).astype(np.float32)

    return NoFFTSpectrogramResult(
        intensities=intensities,
        frequencies_hz=freqs,
        sample_rate=sample_rate,
        hop_size=hop_size,
        min_db=config.min_db,
        max_db=config.max_db,
    )


def compute_nofft_spectrogram_from_file(
    path: str,
    config: NoFFTSpectrogramConfig,
    target_frames: int | None = None,
    max_analysis_samples: int = 2_000_000,
) -> NoFFTSpectrogramResult:
    """Convenience loader + offline noFFT analyzer."""

    data, sample_rate = sf.read(path, dtype="float32")
    return compute_nofft_spectrogram(
        data,
        int(sample_rate),
        config,
        target_frames=target_frames,
        max_analysis_samples=max_analysis_samples,
    )
