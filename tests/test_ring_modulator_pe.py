"""
Tests for RingModulatorPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    RingModulatorPE,
    ConstantPE,
    PiecewisePE,
    CropPE,
    NullRenderer,
    Extent,
)


class TestRingModulatorPEBasics:
    """Test basic RingModulatorPE creation and properties."""

    def test_inputs_no_pe_params(self):
        carrier = ConstantPE(1.0)
        modulator = ConstantPE(0.5)
        rm = RingModulatorPE(carrier, modulator)
        assert rm.inputs() == [carrier, modulator]

    def test_inputs_with_pe_bias(self):
        carrier = ConstantPE(1.0)
        modulator = ConstantPE(0.5)
        bias_pe = ConstantPE(0.2)
        rm = RingModulatorPE(carrier, modulator, bias=bias_pe)
        assert rm.inputs() == [carrier, modulator, bias_pe]

    def test_inputs_with_pe_mix(self):
        carrier = ConstantPE(1.0)
        modulator = ConstantPE(0.5)
        mix_pe = ConstantPE(0.8)
        rm = RingModulatorPE(carrier, modulator, mix=mix_pe)
        assert rm.inputs() == [carrier, modulator, mix_pe]

    def test_inputs_with_all_pe_params(self):
        carrier = ConstantPE(1.0)
        modulator = ConstantPE(0.5)
        bias_pe = ConstantPE(0.2)
        mix_pe = ConstantPE(0.8)
        rm = RingModulatorPE(carrier, modulator, bias=bias_pe, mix=mix_pe)
        assert rm.inputs() == [carrier, modulator, bias_pe, mix_pe]

    def test_is_pure(self):
        carrier = ConstantPE(1.0)
        modulator = ConstantPE(0.5)
        rm = RingModulatorPE(carrier, modulator)
        assert rm.is_pure() is True

    def test_channel_count_from_carrier(self):
        carrier = ConstantPE(1.0, channels=2)
        modulator = ConstantPE(0.5)
        rm = RingModulatorPE(carrier, modulator)
        assert rm.channel_count() == 2


class TestRingModulatorPEExtent:
    """Test RingModulatorPE extent calculation."""

    def test_extent_both_infinite(self):
        carrier = ConstantPE(1.0)
        modulator = ConstantPE(0.5)
        rm = RingModulatorPE(carrier, modulator)
        extent = rm.extent()
        assert extent.start is None
        assert extent.end is None

    def test_extent_intersection_of_carrier_and_modulator(self):
        carrier = PiecewisePE([(0, 1.0), (1000, 1.0)])   # extent (0, 1000)
        modulator = PiecewisePE([(0, 0.5), (500, 0.5)])  # extent (0, 500)
        rm = RingModulatorPE(carrier, modulator)
        extent = rm.extent()
        assert extent.start == 0
        assert extent.end == 500

    def test_extent_with_pe_bias(self):
        carrier = PiecewisePE([(0, 1.0), (1000, 1.0)])
        modulator = ConstantPE(0.5)
        bias_pe = PiecewisePE([(0, 0.0), (200, 1.0)])    # extent (0, 200)
        rm = RingModulatorPE(carrier, modulator, bias=bias_pe)
        extent = rm.extent()
        assert extent.end == 200

    def test_extent_with_pe_mix(self):
        carrier = PiecewisePE([(0, 1.0), (1000, 1.0)])
        modulator = ConstantPE(0.5)
        mix_pe = PiecewisePE([(0, 0.0), (300, 1.0)])     # extent (0, 300)
        rm = RingModulatorPE(carrier, modulator, mix=mix_pe)
        extent = rm.extent()
        assert extent.end == 300


class TestRingModulatorPERender:
    """Test RingModulatorPE rendering."""

    def _setup(self, rm):
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(rm)
        renderer.start()
        return renderer

    def test_pure_ring_mod(self):
        """bias=0, mix=1: output equals carrier × modulator exactly."""
        carrier = ConstantPE(0.8)
        modulator = ConstantPE(0.5)
        rm = RingModulatorPE(carrier, modulator, bias=0.0, mix=1.0)
        renderer = self._setup(rm)
        snippet = rm.render(0, 100)
        # 0.8 × 0.5 = 0.4
        expected = np.full((100, 1), 0.4, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=6)
        renderer.stop()

    def test_amplitude_modulation(self):
        """bias=1, mix=1: output equals carrier × (modulator + 1)."""
        carrier = ConstantPE(0.5)
        modulator = ConstantPE(0.5)
        rm = RingModulatorPE(carrier, modulator, bias=1.0, mix=1.0)
        renderer = self._setup(rm)
        snippet = rm.render(0, 100)
        # 0.5 × (0.5 + 1.0) = 0.75
        expected = np.full((100, 1), 0.75, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=6)
        renderer.stop()

    def test_dry_passthrough(self):
        """mix=0: passes carrier unchanged."""
        carrier = ConstantPE(0.6)
        modulator = ConstantPE(0.5)
        rm = RingModulatorPE(carrier, modulator, bias=0.0, mix=0.0)
        renderer = self._setup(rm)
        snippet = rm.render(0, 100)
        expected = np.full((100, 1), 0.6, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=6)
        renderer.stop()

    def test_mono_modulator_stereo_carrier(self):
        """Mono modulator is broadcast to match stereo carrier."""
        carrier = ConstantPE(1.0, channels=2)
        modulator = ConstantPE(0.5)   # mono
        rm = RingModulatorPE(carrier, modulator, bias=0.0, mix=1.0)
        renderer = self._setup(rm)
        snippet = rm.render(0, 100)
        assert snippet.channels == 2
        # 1.0 × 0.5 = 0.5 on both channels
        expected = np.full((100, 2), 0.5, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=6)
        renderer.stop()

    def test_pe_bias(self):
        """bias as PE input."""
        carrier = ConstantPE(1.0)
        modulator = ConstantPE(0.0)
        # wet = 1.0 × (0 + bias) = bias; output = mix × wet = bias
        bias_pe = ConstantPE(0.3)
        rm = RingModulatorPE(carrier, modulator, bias=bias_pe, mix=1.0)
        renderer = self._setup(rm)
        snippet = rm.render(0, 100)
        expected = np.full((100, 1), 0.3, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=6)
        renderer.stop()

    def test_pe_mix(self):
        """mix as PE input."""
        carrier = ConstantPE(1.0)
        modulator = ConstantPE(-1.0)
        # wet = 1.0 × (-1.0 + 0) = -1.0
        # output = (1 - 0.5) × 1.0 + 0.5 × (-1.0) = 0.5 - 0.5 = 0.0
        mix_pe = ConstantPE(0.5)
        rm = RingModulatorPE(carrier, modulator, bias=0.0, mix=mix_pe)
        renderer = self._setup(rm)
        snippet = rm.render(0, 100)
        expected = np.zeros((100, 1), dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=6)
        renderer.stop()

    def test_output_dtype_float32(self):
        """Output must be float32."""
        carrier = ConstantPE(1.0)
        modulator = ConstantPE(0.5)
        rm = RingModulatorPE(carrier, modulator)
        renderer = self._setup(rm)
        snippet = rm.render(0, 100)
        assert snippet.data.dtype == np.float32
        renderer.stop()
