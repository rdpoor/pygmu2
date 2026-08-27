"""
Tests for TralfamPE (FFT-domain random phase randomiser).

Only TralfamPE-specific behaviour is tested here.  MagFreqPE's shared
machinery (render boundaries, caching, channel count, inputs, extent
propagation, normalize_peak validation, infinite-source handling) is
covered in tests/test_mag_freq_pe.py and is not repeated.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np

import pygmu2 as pg
from pygmu2.mag_freq_pe import MagFreqPE
from pygmu2.tralfam_pe import TralfamPE

N = 128  # signal length for most tests


def make_source(n=N):
    """Deterministic noise-like mono signal with extent [0, n)."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(n).astype(np.float32)
    return pg.ArrayPE(data)


def magnitude_spectrum(signal_1d):
    """Return the FFT magnitude spectrum of a 1-D array."""
    return np.abs(np.fft.fft(signal_1d.astype(np.float64)))


# ── Subclass relationship ─────────────────────────────────────────────────────


class TestTralfamPEIsSubclass:

    def test_is_subclass_of_mag_freq_pe(self):
        pe = TralfamPE(make_source())
        assert isinstance(pe, MagFreqPE)

    def test_accessible_via_pg_namespace(self):
        assert hasattr(pg, "TralfamPE")
        assert pg.TralfamPE is TralfamPE


# ── repr ──────────────────────────────────────────────────────────────────────


class TestTralfamPERepr:

    def test_repr_minimal(self):
        r = repr(TralfamPE(make_source()))
        assert r.startswith("TralfamPE(")
        assert "seed" not in r
        assert "normalize_peak" not in r
        assert "padded_length" not in r

    def test_repr_with_seed(self):
        r = repr(TralfamPE(make_source(), seed=42))
        assert "seed=42" in r

    def test_repr_with_normalize_peak(self):
        r = repr(TralfamPE(make_source(), normalize_peak=0.5))
        assert "normalize_peak=0.5" in r

    def test_repr_with_padded_length(self):
        r = repr(TralfamPE(make_source(), padded_length=256))
        assert "padded_length=256" in r

    def test_repr_all_params(self):
        r = repr(
            TralfamPE(make_source(), seed=7, normalize_peak=0.3, padded_length=512)
        )
        assert "seed=7" in r
        assert "normalize_peak=0.3" in r
        assert "padded_length=512" in r


# ── Seed / reproducibility ────────────────────────────────────────────────────


class TestTralfamPESeed:

    def test_same_seed_gives_identical_output(self):
        src = make_source()
        out1 = TralfamPE(src, seed=1).render(0, N).data
        out2 = TralfamPE(src, seed=1).render(0, N).data
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_give_different_output(self):
        src = make_source()
        out1 = TralfamPE(src, seed=1).render(0, N).data
        out2 = TralfamPE(src, seed=2).render(0, N).data
        assert not np.array_equal(out1, out2)

    def test_no_seed_gives_different_output_each_construction(self):
        src = make_source()
        out1 = TralfamPE(src, seed=None).render(0, N).data
        out2 = TralfamPE(src, seed=None).render(0, N).data
        # Vanishingly unlikely to be equal with random phases
        assert not np.array_equal(out1, out2)


# ── DSP correctness ───────────────────────────────────────────────────────────


class TestTralfamPEDSP:

    def test_output_uses_source_magnitudes_and_random_phases(self):
        """
        Reconstruct the expected output from first principles and verify it
        matches TralfamPE exactly.

        Note: |FFT(output)| ≠ |FFT(source)| in general because taking the
        real part of IFFT(M*exp(iφ)) with non-conjugate-symmetric random
        phases mixes bins.  The correct invariant is at the intermediate
        stage: the magnitudes PASSED TO the mangler equal |FFT(source)|, and
        the random phases come from default_rng(seed).
        """
        SEED = 42
        src = make_source()

        # Mirror what MagFreqPE._mogrify() does internally
        source_frames = src.render(0, N).data  # (N, 1), float32
        magnitudes = np.abs(np.fft.fft(source_frames, axis=0))
        random_phases = np.random.default_rng(SEED).random((N, 1)) * 2.0 * np.pi
        expected = np.real(
            np.fft.ifft(magnitudes * np.exp(1j * random_phases), axis=0)
        ).astype(np.float32)

        output = TralfamPE(src, seed=SEED).render(0, N).data
        np.testing.assert_allclose(output, expected, atol=1e-5)

    def test_output_differs_from_input(self):
        """Random phases produce a different time-domain signal."""
        src = make_source()
        original = src.render(0, N).data
        output = TralfamPE(src, seed=0).render(0, N).data
        assert not np.allclose(output, original)

    def test_output_decorrelated_from_input(self):
        """
        With uniformly random phases the output should be nearly uncorrelated
        with the source in the time domain.
        """
        src = make_source()
        original = src.render(0, N).data[:, 0]
        output = TralfamPE(src, seed=99).render(0, N).data[:, 0]
        corr = np.corrcoef(original, output)[0, 1]
        assert abs(corr) < 0.5  # very loose; random phases decorrelate


# ── padded_length ─────────────────────────────────────────────────────────────


class TestTralfamPEPaddedLength:

    def test_padded_length_extends_extent(self):
        """Zero-padding to 2×N gives output extent [0, 2*N)."""
        src = make_source(N)
        pe = TralfamPE(src, seed=0, padded_length=2 * N)
        assert pe.extent() == pg.Extent(0, 2 * N)

    def test_padded_length_truncates_extent(self):
        """Truncation to N//2 gives output extent [0, N//2)."""
        src = make_source(N)
        pe = TralfamPE(src, seed=0, padded_length=N // 2)
        assert pe.extent() == pg.Extent(0, N // 2)

    def test_padded_length_equal_to_source_length(self):
        """padded_length == source length leaves extent unchanged."""
        src = make_source(N)
        pe = TralfamPE(src, seed=0, padded_length=N)
        assert pe.extent() == pg.Extent(0, N)

    def test_padded_length_output_has_correct_sample_count(self):
        """render() returns exactly padded_length samples."""
        padded = 3 * N // 2
        src = make_source(N)
        pe = TralfamPE(src, seed=0, padded_length=padded)
        out = pe.render(0, padded).data
        assert out.shape == (padded, 1)

    def test_padded_length_preserves_magnitude_spectrum_size(self):
        """
        The FFT operates on padded_length samples, so the magnitude spectrum
        has padded_length bins, not the original N.
        """
        padded = 2 * N
        src = make_source(N)
        out = TralfamPE(src, seed=0, padded_length=padded).render(0, padded).data[:, 0]
        assert len(np.fft.fft(out)) == padded

    def test_padded_length_none_uses_source_extent(self):
        """Default (no padding) output extent matches source extent."""
        src = make_source(N)
        pe = TralfamPE(src, seed=0)
        assert pe.extent() == src.extent()

    def test_padded_length_with_offset_source(self):
        """padded_length anchors at the source's start, not at 0."""
        make_source(N)
        src = pg.CropPE(pg.ConstantPE(0.0), start=100, duration=N)
        # Use ArrayPE-based source with offset via CropPE(clip=False)
        src = pg.CropPE(make_source(N), 100, N, clip=False)
        pe = TralfamPE(src, seed=0, padded_length=2 * N)
        assert pe.extent() == pg.Extent(100, 100 + 2 * N)

    def test_seed_reproducibility_with_padded_length(self):
        """seed + padded_length together are reproducible."""
        src = make_source(N)
        padded = 2 * N
        out1 = TralfamPE(src, seed=5, padded_length=padded).render(0, padded).data
        out2 = TralfamPE(src, seed=5, padded_length=padded).render(0, padded).data
        np.testing.assert_array_equal(out1, out2)
