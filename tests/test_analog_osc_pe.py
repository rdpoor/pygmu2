"""
Tests for AnalogOscPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
import pytest

from pygmu2 import (
    AnalogOscPE,
    ArrayPE,
    ConstantPE,
    CropPE,
    Extent,
    NullRenderer,
)


class TestAnalogOscPEBasics:
    def test_create_defaults(self):
        osc = AnalogOscPE()
        assert osc.frequency == 440.0
        assert osc.duty_cycle == 0.5
        assert osc.waveform == "rectangle"
        assert osc.channel_count() == 1

    def test_is_pure_with_constants(self):
        osc = AnalogOscPE(frequency=220.0, duty_cycle=0.25, waveform="rectangle")
        assert osc.is_pure() is True
        assert osc.inputs() == []

    def test_infinite_extent_constant_params(self):
        osc = AnalogOscPE(frequency=220.0, duty_cycle=0.25, waveform="rectangle")
        extent = osc.extent()
        assert extent.start is None
        assert extent.end is None

    def test_waveform_validation(self):
        with pytest.raises(ValueError):
            AnalogOscPE(waveform="nope")

    def test_extent_with_disjoint_pe_inputs_is_empty(self):
        """
        Regression: disjoint input extents should yield an empty extent,
        not a crash.
        """
        freq = CropPE(ConstantPE(100.0), Extent(0, 10))
        duty = CropPE(ConstantPE(0.5), Extent(20, 30))  # disjoint from freq
        osc = AnalogOscPE(frequency=freq, duty_cycle=duty, waveform="rectangle")
        extent = osc.extent()
        assert extent.is_empty()


class TestAnalogOscPERender:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=10_000)

    def test_render_returns_snippet_shape(self):
        osc = AnalogOscPE(frequency=100.0, duty_cycle=0.25, waveform="rectangle")
        self.renderer.set_source(osc)
        snip = osc.render(0, 128)
        assert snip.start == 0
        assert snip.duration == 128
        assert snip.channels == 1

    def test_render_stereo_channels_are_identical(self):
        osc = AnalogOscPE(frequency=100.0, duty_cycle=0.25, waveform="rectangle", channels=2)
        self.renderer.set_source(osc)
        snip = osc.render(0, 256)
        assert snip.channels == 2
        np.testing.assert_array_equal(snip.data[:, 0], snip.data[:, 1])

    def test_rectangle_plateaus_away_from_edges(self):
        """
        Far from discontinuities, a bandlimited pulse should be ~+1 or ~-1.
        """
        osc = AnalogOscPE(frequency=100.0, duty_cycle=0.25, waveform="rectangle")
        self.renderer.set_source(osc)
        y = osc.render(0, 128).data[:, 0]

        # With sr=10k and f=100Hz => dt=0.01, BLEP window = 2dt = 0.02.
        # Choose phases comfortably away from edges.
        assert y[10] == pytest.approx(1.0, abs=1e-3)   # phase=0.10, high
        assert y[40] == pytest.approx(-1.0, abs=1e-3)  # phase=0.40, low

    def test_sawtooth_mode_triangle_at_half_duty(self):
        """
        In sawtooth-mode, duty=0.5 should produce a triangle wave.
        """
        # Use a low frequency so bandlimiting corrections are very localized.
        osc = AnalogOscPE(frequency=10.0, duty_cycle=0.5, waveform="sawtooth")
        self.renderer.set_source(osc)
        y = osc.render(0, 2000).data[:, 0]

        # Samples away from the corners should be close to ideal triangle values.
        # phase=0.25 -> 0.0, phase=0.75 -> 0.0
        assert y[250] == pytest.approx(0.0, abs=5e-3)
        assert y[750] == pytest.approx(0.0, abs=5e-3)

    def test_duty_endpoints_are_clamped_no_nans(self):
        osc = AnalogOscPE(frequency=100.0, duty_cycle=0.0, waveform="sawtooth")
        self.renderer.set_source(osc)
        y = osc.render(0, 256).data[:, 0]
        assert np.all(np.isfinite(y))

    def test_stateful_chunk_continuity_dynamic_frequency(self):
        """
        When driven by PE inputs, AnalogOscPE is stateful. Contiguous renders
        should match a single full render.
        """
        n = 200
        freq_values = np.linspace(80.0, 220.0, n, dtype=np.float32)

        # Full render
        osc_full = AnalogOscPE(frequency=ArrayPE(freq_values), duty_cycle=0.3, waveform="rectangle")
        self.renderer.set_source(osc_full)
        y_full = osc_full.render(0, n).data[:, 0]

        # Chunked render (fresh instance)
        osc_chunk = AnalogOscPE(frequency=ArrayPE(freq_values), duty_cycle=0.3, waveform="rectangle")
        self.renderer.set_source(osc_chunk)
        y1 = osc_chunk.render(0, 80).data[:, 0]
        y2 = osc_chunk.render(80, n - 80).data[:, 0]
        y_cat = np.concatenate([y1, y2])

        np.testing.assert_allclose(y_cat, y_full, atol=1e-6, rtol=0.0)

