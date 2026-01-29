"""
Tests for GainPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    GainPE,
    ConstantPE,
    RampPE,
    SinePE,
    IdentityPE,
    MixPE,
    CropPE,
    DelayPE,
    NullRenderer,
    Extent,
)


class TestGainPEBasics:
    """Test basic GainPE creation and properties."""

    def test_create_with_constant_gain(self):
        source = ConstantPE(1.0)
        gain = GainPE(source, gain=0.5)
        assert gain.source is source
        assert gain.gain == 0.5

    def test_create_with_default_gain(self):
        source = ConstantPE(1.0)
        gain = GainPE(source)
        assert gain.gain == 1.0

    def test_create_with_pe_gain(self):
        source = ConstantPE(1.0)
        gain_pe = RampPE(0.0, 1.0, duration=100)
        gain = GainPE(source, gain=gain_pe)
        assert gain.gain is gain_pe

    def test_inputs_constant_gain(self):
        source = ConstantPE(1.0)
        gain = GainPE(source, gain=0.5)
        assert gain.inputs() == [source]

    def test_inputs_pe_gain(self):
        source = ConstantPE(1.0)
        gain_pe = RampPE(0.0, 1.0, duration=100)
        gain = GainPE(source, gain=gain_pe)
        assert gain.inputs() == [source, gain_pe]

    def test_is_pure(self):
        source = ConstantPE(1.0)
        gain = GainPE(source, gain=0.5)
        assert gain.is_pure() is True

    def test_is_pure_with_pe_gain(self):
        source = ConstantPE(1.0)
        gain_pe = RampPE(0.0, 1.0, duration=100)
        gain = GainPE(source, gain=gain_pe)
        assert gain.is_pure() is True

    def test_channel_count_passthrough(self):
        source = ConstantPE(1.0, channels=2)
        gain = GainPE(source, gain=0.5)
        assert gain.channel_count() == 2

    def test_repr_constant(self):
        source = ConstantPE(1.0)
        gain = GainPE(source, gain=0.5)
        repr_str = repr(gain)
        assert "GainPE" in repr_str
        assert "ConstantPE" in repr_str
        assert "0.5" in repr_str

    def test_repr_pe_gain(self):
        source = ConstantPE(1.0)
        gain_pe = RampPE(0.0, 1.0, duration=100)
        gain = GainPE(source, gain=gain_pe)
        repr_str = repr(gain)
        assert "GainPE" in repr_str
        assert "RampPE" in repr_str


class TestGainPEExtent:
    """Test GainPE extent calculation."""

    def test_extent_constant_gain_infinite_source(self):
        source = ConstantPE(1.0)
        gain = GainPE(source, gain=0.5)

        extent = gain.extent()
        assert extent.start is None
        assert extent.end is None

    def test_extent_constant_gain_finite_source(self):
        source = RampPE(0.0, 1.0, duration=1000)
        gain = GainPE(source, gain=0.5)

        extent = gain.extent()
        assert extent.start == 0
        assert extent.end == 1000

    def test_extent_pe_gain_both_finite(self):
        source = RampPE(0.0, 1.0, duration=1000)  # Extent (0, 1000)
        gain_pe = RampPE(0.0, 1.0, duration=500)  # Extent (0, 500)
        gain = GainPE(source, gain=gain_pe)

        # Intersection is (0, 500)
        extent = gain.extent()
        assert extent.start == 0
        assert extent.end == 500

    def test_extent_pe_gain_no_overlap(self):
        source = RampPE(0.0, 1.0, duration=100)  # Extent (0, 100)
        gain_pe = CropPE(ConstantPE(0.5), Extent(200, 300))  # Extent (200, 300)
        gain = GainPE(source, gain=gain_pe)

        # No intersection -> empty extent at boundary
        extent = gain.extent()
        assert extent == Extent(200, 200)
        assert extent.is_empty() is True

    def test_extent_pe_gain_infinite(self):
        source = RampPE(0.0, 1.0, duration=1000)  # Extent (0, 1000)
        gain_pe = ConstantPE(0.5)  # Infinite extent
        gain = GainPE(source, gain=gain_pe)

        # Intersection is source extent
        extent = gain.extent()
        assert extent.start == 0
        assert extent.end == 1000


class TestGainPERenderConstant:
    """Test GainPE rendering with constant gain."""

    def test_render_unity_gain(self):
        source = ConstantPE(1.0)
        gain = GainPE(source, gain=1.0)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain)
        renderer.start()

        snippet = gain.render(0, 100)
        np.testing.assert_array_equal(
            snippet.data, np.full((100, 1), 1.0, dtype=np.float32)
        )

        renderer.stop()

    def test_render_half_gain(self):
        source = ConstantPE(1.0)
        gain = GainPE(source, gain=0.5)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain)
        renderer.start()

        snippet = gain.render(0, 100)
        np.testing.assert_array_equal(
            snippet.data, np.full((100, 1), 0.5, dtype=np.float32)
        )

        renderer.stop()

    def test_render_double_gain(self):
        source = ConstantPE(0.5)
        gain = GainPE(source, gain=2.0)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain)
        renderer.start()

        snippet = gain.render(0, 100)
        np.testing.assert_array_equal(
            snippet.data, np.full((100, 1), 1.0, dtype=np.float32)
        )

        renderer.stop()

    def test_render_negative_gain(self):
        source = ConstantPE(1.0)
        gain = GainPE(source, gain=-1.0)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain)
        renderer.start()

        snippet = gain.render(0, 100)
        np.testing.assert_array_equal(
            snippet.data, np.full((100, 1), -1.0, dtype=np.float32)
        )

        renderer.stop()

    def test_render_zero_gain(self):
        source = ConstantPE(1.0)
        gain = GainPE(source, gain=0.0)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain)
        renderer.start()

        snippet = gain.render(0, 100)
        np.testing.assert_array_equal(
            snippet.data, np.zeros((100, 1), dtype=np.float32)
        )

        renderer.stop()

    def test_render_stereo(self):
        source = ConstantPE(1.0, channels=2)
        gain = GainPE(source, gain=0.5)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain)
        renderer.start()

        snippet = gain.render(0, 100)
        assert snippet.channels == 2
        np.testing.assert_array_equal(
            snippet.data, np.full((100, 2), 0.5, dtype=np.float32)
        )

        renderer.stop()


class TestGainPERenderPE:
    """Test GainPE rendering with PE gain."""

    def test_render_ramp_gain(self):
        """Gain ramps from 0 to 1."""
        source = ConstantPE(1.0)
        gain_pe = RampPE(0.0, 1.0, duration=100)
        gain = GainPE(source, gain=gain_pe)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain)
        renderer.start()

        snippet = gain.render(0, 100)

        # Result should be ramp from 0 to 1 (source * gain)
        expected = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)

        renderer.stop()

    def test_render_pe_gain_on_ramp_source(self):
        """Both source and gain are ramps - result is product."""
        source = RampPE(0.0, 2.0, duration=100)
        gain_pe = RampPE(0.0, 0.5, duration=100)
        gain = GainPE(source, gain=gain_pe)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain)
        renderer.start()

        snippet = gain.render(0, 100)

        # Result is source * gain
        source_vals = np.linspace(0.0, 2.0, 100, dtype=np.float32)
        gain_vals = np.linspace(0.0, 0.5, 100, dtype=np.float32)
        expected = (source_vals * gain_vals).reshape(-1, 1)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)

        renderer.stop()

    def test_render_pe_gain_mono_on_stereo(self):
        """Mono gain PE applied to stereo source."""
        source = ConstantPE(1.0, channels=2)
        gain_pe = RampPE(0.0, 1.0, duration=100)
        gain = GainPE(source, gain=gain_pe)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain)
        renderer.start()

        snippet = gain.render(0, 100)
        assert snippet.channels == 2

        # Both channels should have same gain applied
        expected_mono = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(-1, 1)
        expected = np.tile(expected_mono, (1, 2))
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)

        renderer.stop()

    def test_render_constant_pe_gain(self):
        """ConstantPE as gain (should behave like constant)."""
        source = ConstantPE(1.0)
        gain_pe = ConstantPE(0.5)
        gain = GainPE(source, gain=gain_pe)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain)
        renderer.start()

        snippet = gain.render(0, 100)
        np.testing.assert_array_equal(
            snippet.data, np.full((100, 1), 0.5, dtype=np.float32)
        )

        renderer.stop()


class TestGainPEUseCases:
    """Test common GainPE use cases."""

    def test_fade_in(self):
        """Create a fade-in effect."""
        source = ConstantPE(1.0)
        fade = RampPE(0.0, 1.0, duration=1000)
        faded = GainPE(source, gain=fade)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(faded)
        renderer.start()

        # Start should be near zero
        snippet = faded.render(0, 10)
        assert snippet.data[0, 0] < 0.01

        # End should be near 1
        snippet = faded.render(990, 10)
        assert snippet.data[9, 0] > 0.99

        renderer.stop()

    def test_fade_out(self):
        """Create a fade-out effect."""
        source = ConstantPE(1.0)
        fade = RampPE(1.0, 0.0, duration=1000)
        faded = GainPE(source, gain=fade)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(faded)
        renderer.start()

        # Start should be near 1
        snippet = faded.render(0, 10)
        assert snippet.data[0, 0] > 0.99

        # End should be near zero
        snippet = faded.render(990, 10)
        assert snippet.data[9, 0] < 0.01

        renderer.stop()

    def test_tremolo(self):
        """Create a tremolo effect using LFO."""
        source = ConstantPE(1.0)
        # LFO: 0.7 ± 0.3 (oscillates between 0.4 and 1.0)
        lfo = SinePE(frequency=5.0, amplitude=0.3)
        base_gain = ConstantPE(0.7)
        tremolo_gain = MixPE(base_gain, lfo)
        tremolo = GainPE(source, gain=tremolo_gain)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(tremolo)
        renderer.start()

        # Render enough for several LFO cycles
        snippet = tremolo.render(0, 44100)

        # Check that values oscillate within expected range
        assert np.min(snippet.data) >= 0.35  # ~0.4 - some tolerance
        assert np.max(snippet.data) <= 1.05  # ~1.0 + some tolerance

        renderer.stop()

    def test_simple_echo(self):
        """Create a simple echo by mixing original with delayed+gained copy."""
        source = ConstantPE(1.0)
        cropped = CropPE(source, Extent(0, 100))

        delayed = DelayPE(cropped, delay=50)
        quieter = GainPE(delayed, gain=0.5)

        mixed = MixPE(cropped, quieter)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(mixed)
        renderer.start()

        # 0-49: original only (1.0)
        snippet = mixed.render(0, 50)
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 1), 1.0, dtype=np.float32)
        )

        # 50-99: original + echo (1.0 + 0.5 = 1.5)
        snippet = mixed.render(50, 50)
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 1), 1.5, dtype=np.float32)
        )

        # 100-149: echo only (0.5)
        snippet = mixed.render(100, 50)
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 1), 0.5, dtype=np.float32)
        )

        renderer.stop()


class TestGainPEIntegration:
    """Integration tests for GainPE."""

    def test_full_render_cycle(self):
        source = ConstantPE(1.0)
        gain = GainPE(source, gain=0.5)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain)

        with renderer:
            renderer.start()
            snippet = gain.render(0, 100)
            assert np.allclose(snippet.data, 0.5)

    def test_chain_gains(self):
        """Chain multiple gain stages."""
        source = ConstantPE(1.0)
        gain1 = GainPE(source, gain=0.5)
        gain2 = GainPE(gain1, gain=0.5)
        gain3 = GainPE(gain2, gain=0.5)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(gain3)
        renderer.start()

        snippet = gain3.render(0, 100)
        # 1.0 * 0.5 * 0.5 * 0.5 = 0.125
        np.testing.assert_array_almost_equal(
            snippet.data, np.full((100, 1), 0.125, dtype=np.float32), decimal=5
        )

        renderer.stop()
