"""
Tests for ConvolvePE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
import pytest

from pygmu2 import (
    ArrayPE,
    ConvolvePE,
    ConstantPE,
    CropPE,
    Extent,
    NullRenderer,
)


class TestConvolvePEBasics:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=10_000)

    def test_filter_must_start_at_zero(self):
        src = ArrayPE([1, 2, 3, 4])
        filt = CropPE(ArrayPE([1, 0, 0]), Extent(1, 3))  # start != 0
        pe = ConvolvePE(src, filt)
        with pytest.raises(ValueError):
            pe.extent()

    def test_filter_must_be_finite(self):
        src = ArrayPE([1, 2, 3, 4])
        filt = ConstantPE(1.0)  # infinite extent (None,None)
        pe = ConvolvePE(src, filt)
        with pytest.raises(ValueError):
            pe.extent()


class TestConvolvePERender:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=10_000)

    def test_matches_numpy_convolve_mono(self):
        x = np.array([1, 2, 3, 4], dtype=np.float32)
        h = np.array([1, 0.5, -1], dtype=np.float32)

        src = ArrayPE(x)
        filt = ArrayPE(h)
        pe = ConvolvePE(src, filt, fft_size=16)
        self.renderer.set_source(pe)

        # Full convolution length
        y_expected = np.convolve(x, h, mode="full").astype(np.float32)
        y = pe.render(0, len(y_expected)).data[:, 0]
        np.testing.assert_allclose(y, y_expected, atol=1e-5, rtol=0.0)

    def test_dirac_impulse_is_identity(self):
        """
        Convolution with a unit impulse (Dirac) should reproduce the source.

        Here we use a 1-sample FIR [1.0], which satisfies the ConvolvePE filter
        contract (finite, starts at 0).
        """
        rng = np.random.default_rng(0)
        x = rng.normal(size=64).astype(np.float32)

        src = ArrayPE(x)
        filt = ArrayPE([1.0])
        pe = ConvolvePE(src, filt, fft_size=64)
        self.renderer.set_source(pe)

        y = pe.render(0, len(x)).data[:, 0]
        np.testing.assert_allclose(y, x, atol=1e-6, rtol=0.0)

    def test_filter_mono_applies_to_all_channels(self):
        x = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
        h = np.array([1, -1], dtype=np.float32)

        src = ArrayPE(x)
        filt = ArrayPE(h)
        pe = ConvolvePE(src, filt, fft_size=16)
        self.renderer.set_source(pe)

        y_expected_l = np.convolve(x[:, 0], h, mode="full")
        y_expected_r = np.convolve(x[:, 1], h, mode="full")
        y = pe.render(0, len(y_expected_l)).data

        np.testing.assert_allclose(y[:, 0], y_expected_l, atol=1e-5, rtol=0.0)
        np.testing.assert_allclose(y[:, 1], y_expected_r, atol=1e-5, rtol=0.0)

    def test_mono_src_multi_channel_filter_fans_out(self):
        """
        Mono source with multi-channel filter should fan out to multi-channel output.

        This is useful for HRTF-style processing where a mono signal is convolved
        with a stereo (or multi-channel) impulse response to produce a multi-
        channel result.
        """
        # Mono input
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        # Simple 2-channel FIR: left=[1, 0.5], right=[-1, 0.5]
        h = np.stack(
            [
                np.array([1.0, 0.5], dtype=np.float32),
                np.array([-1.0, 0.5], dtype=np.float32),
            ],
            axis=1,
        )  # shape (2, 2)

        src = ArrayPE(x)
        filt = ArrayPE(h)
        pe = ConvolvePE(src, filt, fft_size=16)
        self.renderer.set_source(pe)

        # Expected per-channel convolutions
        h_l = h[:, 0]
        h_r = h[:, 1]
        y_expected_l = np.convolve(x, h_l, mode="full")
        y_expected_r = np.convolve(x, h_r, mode="full")

        y = pe.render(0, len(y_expected_l)).data

        # Output should be stereo (2 channels)
        assert y.shape[1] == 2
        np.testing.assert_allclose(y[:, 0], y_expected_l, atol=1e-5, rtol=0.0)
        np.testing.assert_allclose(y[:, 1], y_expected_r, atol=1e-5, rtol=0.0)

    def test_chunked_render_matches_full(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=200).astype(np.float32)
        h = np.array([0.25, 0.5, 0.25], dtype=np.float32)

        src1 = ArrayPE(x)
        filt1 = ArrayPE(h)
        full = ConvolvePE(src1, filt1, fft_size=64)
        self.renderer.set_source(full)
        y_full = full.render(0, len(x) + len(h) - 1).data[:, 0]

        src2 = ArrayPE(x)
        filt2 = ArrayPE(h)
        chunked = ConvolvePE(src2, filt2, fft_size=64)
        self.renderer.set_source(chunked)

        y_parts = []
        start = 0
        total = len(x) + len(h) - 1
        for dur in (17, 23, 19, 41, 7, 93):
            if start >= total:
                break
            d = min(dur, total - start)
            y_parts.append(chunked.render(start, d).data[:, 0])
            start += d
        if start < total:
            y_parts.append(chunked.render(start, total - start).data[:, 0])
        y_chunk = np.concatenate(y_parts)[: len(y_full)]
        np.testing.assert_allclose(y_chunk, y_full, atol=1e-5, rtol=0.0)

