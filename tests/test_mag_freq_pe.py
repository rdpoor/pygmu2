"""
Tests for MagFreqPE (FFT-domain magnitude and phase processor).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np
import pytest

import pygmu2 as pg
from pygmu2.mag_freq_pe import MagFreqPE


# ── Fixtures / helpers ────────────────────────────────────────────────────────

N = 64  # default signal length for most tests


def make_source(data):
    """1-D list/array → mono ArrayPE with extent [0, len(data))."""
    return pg.ArrayPE(np.array(data, dtype=np.float32))


def make_stereo_source(data):
    """2-D array (samples, 2) → stereo ArrayPE."""
    return pg.ArrayPE(np.array(data, dtype=np.float32))


def identity_mangler(magnitudes, phases):
    return magnitudes, phases


def zero_magnitude_mangler(magnitudes, phases):
    return np.zeros_like(magnitudes), phases


def negate_phase_mangler(magnitudes, phases):
    return magnitudes, -phases


def sine_signal(n=N, freq=4):
    """Pure sine wave with integer cycles so FFT has clean spectrum."""
    t = np.arange(n)
    return np.sin(2 * np.pi * freq * t / n).astype(np.float32)


# ── Construction ──────────────────────────────────────────────────────────────

class TestMagFreqPEConstruction:

    def test_basic_construction(self):
        pe = MagFreqPE(make_source(sine_signal()), identity_mangler)
        assert pe is not None

    def test_normalize_peak_zero_raises(self):
        with pytest.raises(ValueError, match="normalize_peak"):
            MagFreqPE(make_source(sine_signal()), identity_mangler, normalize_peak=0.0)

    def test_normalize_peak_negative_raises(self):
        with pytest.raises(ValueError, match="normalize_peak"):
            MagFreqPE(make_source(sine_signal()), identity_mangler, normalize_peak=-1.0)

    def test_normalize_peak_inf_raises(self):
        with pytest.raises(ValueError, match="normalize_peak"):
            MagFreqPE(make_source(sine_signal()), identity_mangler,
                      normalize_peak=float("inf"))

    def test_normalize_peak_nan_raises(self):
        with pytest.raises(ValueError, match="normalize_peak"):
            MagFreqPE(make_source(sine_signal()), identity_mangler,
                      normalize_peak=float("nan"))

    def test_normalize_peak_valid(self):
        pe = MagFreqPE(make_source(sine_signal()), identity_mangler,
                       normalize_peak=0.5)
        assert pe is not None


# ── PE interface ──────────────────────────────────────────────────────────────

class TestMagFreqPEInterface:

    def test_is_pure(self):
        pe = MagFreqPE(make_source(sine_signal()), identity_mangler)
        assert pe.is_pure() is True

    def test_channel_count_mono(self):
        pe = MagFreqPE(make_source(sine_signal()), identity_mangler)
        assert pe.channel_count() == 1

    def test_channel_count_stereo(self):
        data = np.ones((N, 2), dtype=np.float32)
        pe = MagFreqPE(make_stereo_source(data), identity_mangler)
        assert pe.channel_count() == 2

    def test_inputs_contains_source(self):
        src = make_source(sine_signal())
        pe = MagFreqPE(src, identity_mangler)
        assert src in pe.inputs()
        assert len(pe.inputs()) == 1

    def test_extent_matches_source(self):
        src = make_source(sine_signal(N))
        pe = MagFreqPE(src, identity_mangler)
        assert pe.extent() == pg.Extent(0, N)

    def test_extent_offset_source(self):
        inner = make_source(sine_signal(N))
        src = pg.CropPE(pg.ConstantPE(0.0), start=100, duration=N)
        pe = MagFreqPE(src, identity_mangler)
        assert pe.extent() == pg.Extent(100, 100 + N)

    def test_repr_without_normalize_peak(self):
        src = make_source(sine_signal())
        pe = MagFreqPE(src, identity_mangler)
        r = repr(pe)
        assert "MagFreqPE" in r
        assert "normalize_peak" not in r

    def test_repr_with_normalize_peak(self):
        src = make_source(sine_signal())
        pe = MagFreqPE(src, identity_mangler, normalize_peak=0.5)
        r = repr(pe)
        assert "MagFreqPE" in r
        assert "0.5" in r

    def test_accessible_via_pg_namespace(self):
        assert hasattr(pg, "MagFreqPE")
        assert pg.MagFreqPE is MagFreqPE


# ── DSP correctness ───────────────────────────────────────────────────────────

class TestMagFreqPEDSP:

    def test_identity_mangler_roundtrip(self):
        """FFT → identity → IFFT reproduces the original signal."""
        signal = sine_signal(N, freq=4)
        src = make_source(signal)
        pe = MagFreqPE(src, identity_mangler)
        out = pe.render(0, N).data[:, 0]
        np.testing.assert_allclose(out, signal, atol=1e-5)

    def test_identity_on_arbitrary_signal(self):
        """Round-trip works for non-trivial (noise-like) signals."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(N).astype(np.float32)
        src = make_source(signal)
        pe = MagFreqPE(src, identity_mangler)
        out = pe.render(0, N).data[:, 0]
        np.testing.assert_allclose(out, signal, atol=1e-5)

    def test_zero_magnitude_produces_silence(self):
        """Zeroing all magnitudes gives an all-zero output."""
        src = make_source(sine_signal(N))
        pe = MagFreqPE(src, zero_magnitude_mangler)
        out = pe.render(0, N).data[:, 0]
        np.testing.assert_array_equal(out, np.zeros(N, dtype=np.float32))

    def test_phase_negation_is_time_reversal(self):
        """
        Negating all phases conjugates the spectrum, which is equivalent to
        time-reversal with wrap-around: output[n] = input[(N - n) % N].
        """
        signal = sine_signal(N, freq=3)
        src = make_source(signal)
        pe = MagFreqPE(src, negate_phase_mangler)
        out = pe.render(0, N).data[:, 0]
        expected = np.roll(signal[::-1], 1)  # reverse then shift by 1 = [(N-n)%N]
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_normalize_peak_sets_peak_amplitude(self):
        """Output peak should equal normalize_peak (up to float32 precision)."""
        target = 0.5
        src = make_source(sine_signal(N))
        pe = MagFreqPE(src, identity_mangler, normalize_peak=target)
        out = pe.render(0, N).data[:, 0]
        actual_peak = np.max(np.abs(out))
        assert actual_peak == pytest.approx(target, abs=1e-5)

    def test_normalize_peak_none_leaves_amplitude_unchanged(self):
        """Without normalize_peak the identity mangler preserves amplitude."""
        signal = sine_signal(N) * 0.3
        src = make_source(signal)
        pe = MagFreqPE(src, identity_mangler, normalize_peak=None)
        out = pe.render(0, N).data[:, 0]
        np.testing.assert_allclose(out, signal, atol=1e-5)

    def test_normalize_peak_silent_source_stays_silent(self):
        """normalize_peak on an all-zero source must not divide by zero."""
        src = make_source(np.zeros(N, dtype=np.float32))
        pe = MagFreqPE(src, identity_mangler, normalize_peak=0.5)
        out = pe.render(0, N).data[:, 0]
        np.testing.assert_array_equal(out, np.zeros(N, dtype=np.float32))

    def test_mangler_receives_correct_shapes_mono(self):
        """Mangler arrays must have shape (n_samples, channels)."""
        received = {}

        def recording_mangler(magnitudes, phases):
            received["mag_shape"] = magnitudes.shape
            received["ph_shape"] = phases.shape
            return magnitudes, phases

        src = make_source(sine_signal(N))
        pe = MagFreqPE(src, recording_mangler)
        pe.render(0, N)

        assert received["mag_shape"] == (N, 1)
        assert received["ph_shape"] == (N, 1)

    def test_mangler_receives_correct_shapes_stereo(self):
        """Stereo source produces (n_samples, 2) mangler arrays."""
        received = {}

        def recording_mangler(magnitudes, phases):
            received["mag_shape"] = magnitudes.shape
            return magnitudes, phases

        data = np.tile(sine_signal(N).reshape(-1, 1), (1, 2))
        src = make_stereo_source(data)
        pe = MagFreqPE(src, recording_mangler)
        pe.render(0, N)

        assert received["mag_shape"] == (N, 2)

    def test_stereo_output_shape(self):
        """Two-channel source produces two-channel output."""
        data = np.tile(sine_signal(N).reshape(-1, 1), (1, 2))
        src = make_stereo_source(data)
        pe = MagFreqPE(src, identity_mangler)
        out = pe.render(0, N).data
        assert out.shape == (N, 2)


# ── Caching ───────────────────────────────────────────────────────────────────

class TestMagFreqPECaching:

    def test_mangler_called_only_once(self):
        """_mogrify() caches; the mangler must not be called again on re-render."""
        call_count = [0]

        def counting_mangler(magnitudes, phases):
            call_count[0] += 1
            return magnitudes, phases

        src = make_source(sine_signal(N))
        pe = MagFreqPE(src, counting_mangler)

        pe.render(0, N)
        pe.render(0, N)
        pe.render(0, 32)

        assert call_count[0] == 1

    def test_repeated_renders_agree(self):
        """Multiple renders of the same range return identical data."""
        src = make_source(sine_signal(N))
        pe = MagFreqPE(src, identity_mangler)
        out1 = pe.render(0, N).data.copy()
        out2 = pe.render(0, N).data.copy()
        np.testing.assert_array_equal(out1, out2)


# ── Render boundary behaviour ─────────────────────────────────────────────────

class TestMagFreqPERenderBoundaries:

    def _make_pe(self):
        """MagFreqPE over a known signal with extent [0, N)."""
        return MagFreqPE(make_source(sine_signal(N)), identity_mangler)

    def test_render_full_extent(self):
        """Rendering exactly [0, N) returns all N samples."""
        pe = self._make_pe()
        snip = pe.render(0, N)
        assert snip.data.shape == (N, 1)

    def test_render_inside_extent(self):
        """A sub-range inside the extent returns the correct slice."""
        signal = sine_signal(N)
        src = make_source(signal)
        pe = MagFreqPE(src, identity_mangler)
        out_full = pe.render(0, N).data[:, 0]
        out_slice = pe.render(10, 20).data[:, 0]
        np.testing.assert_allclose(out_slice, out_full[10:30], atol=1e-6)

    def test_render_before_extent_returns_zeros(self):
        """Requesting samples entirely before extent gives zeros."""
        pe = self._make_pe()
        out = pe.render(-20, 10).data[:, 0]
        np.testing.assert_array_equal(out, np.zeros(10, dtype=np.float32))

    def test_render_after_extent_returns_zeros(self):
        """Requesting samples entirely after extent gives zeros."""
        pe = self._make_pe()
        out = pe.render(N + 10, 10).data[:, 0]
        np.testing.assert_array_equal(out, np.zeros(10, dtype=np.float32))

    def test_render_overlapping_start(self):
        """Render straddling the left edge: leading samples are zero."""
        signal = sine_signal(N)
        src = make_source(signal)
        pe = MagFreqPE(src, identity_mangler)
        full = pe.render(0, N).data[:, 0]
        # Render [-10, 10): 10 zeros then signal[0:10]
        out = pe.render(-10, 20).data[:, 0]
        np.testing.assert_array_equal(out[:10], np.zeros(10, dtype=np.float32))
        np.testing.assert_allclose(out[10:20], full[0:10], atol=1e-6)

    def test_render_overlapping_end(self):
        """Render straddling the right edge: trailing samples are zero."""
        signal = sine_signal(N)
        src = make_source(signal)
        pe = MagFreqPE(src, identity_mangler)
        full = pe.render(0, N).data[:, 0]
        # Render [N-10, N+10): signal[N-10:N] then 10 zeros
        out = pe.render(N - 10, 20).data[:, 0]
        np.testing.assert_allclose(out[:10], full[N - 10:N], atol=1e-6)
        np.testing.assert_array_equal(out[10:], np.zeros(10, dtype=np.float32))

    def test_render_output_shape_matches_duration(self):
        """Output always has exactly (duration, channels) shape."""
        pe = self._make_pe()
        for start, dur in [(-5, 10), (0, N), (10, 5), (N, 8)]:
            out = pe.render(start, dur)
            assert out.data.shape == (dur, 1), f"shape mismatch for start={start}, dur={dur}"

    def test_render_zero_duration(self):
        """Requesting 0 samples returns an empty snippet."""
        pe = self._make_pe()
        out = pe.render(0, 0)
        assert out.data.shape[0] == 0


# ── Infinite-extent source ─────────────────────────────────────────────────────

class TestMagFreqPEInfiniteSource:

    def test_infinite_source_returns_zeros(self):
        """
        ConstantPE has infinite extent.  _render() short-circuits before
        _mogrify() and returns zeros (silent graceful degradation).
        """
        pe = MagFreqPE(pg.ConstantPE(0.5), identity_mangler)
        out = pe.render(0, N).data
        np.testing.assert_array_equal(out, np.zeros((N, 1), dtype=np.float32))

    def test_mogrify_raises_for_infinite_extent(self):
        """_mogrify() directly raises ValueError for infinite-extent sources."""
        pe = MagFreqPE(pg.ConstantPE(0.5), identity_mangler)
        with pytest.raises(ValueError, match="finite"):
            pe._mogrify()
