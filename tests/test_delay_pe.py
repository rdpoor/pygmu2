"""
Tests for DelayPE (unified: integer, float, and PE delays).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    DelayPE,
    InterpolationMode,
    ConstantPE,
    PiecewisePE,
    SinePE,
    MixPE,
    IdentityPE,
    NullRenderer,
    Extent,
    CropPE,
)


# =============================================================================
# Integer Delay Tests (Fast Path)
# =============================================================================

class TestDelayPEIntegerBasics:
    """Test basic DelayPE creation with integer delay."""
    
    def test_create_delay_pe(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100)
        assert delay.source is source
        assert delay.delay == 100
    
    def test_create_with_zero_delay(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=0)
        assert delay.delay == 0
    
    def test_create_with_negative_delay(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=-100)
        assert delay.delay == -100
    
    def test_inputs_integer(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100)
        # Integer delay: only source in inputs
        assert delay.inputs() == [source]
    
    def test_is_pure(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100)
        assert delay.is_pure() is True
    
    def test_channel_count_passthrough(self):
        source = ConstantPE(1.0, channels=2)
        delay = DelayPE(source, delay=100)
        assert delay.channel_count() == 2
    
    def test_repr_integer(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100)
        repr_str = repr(delay)
        assert "DelayPE" in repr_str
        assert "ConstantPE" in repr_str
        assert "100" in repr_str
        # Should NOT include interpolation for integer delay
        assert "interpolation" not in repr_str
    
    def test_whole_number_float_treated_as_int(self):
        """A float that is a whole number should be treated as int."""
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100.0)
        assert delay.delay == 100
        assert isinstance(delay.delay, int)


class TestDelayPEIntegerExtent:
    """Test DelayPE extent calculation with integer delay."""
    
    def test_extent_finite_positive_delay(self):
        source = PiecewisePE([(0, 0.0), (1000, 1.0)])
        delay = DelayPE(source, delay=500)
        
        extent = delay.extent()
        assert extent.start == 500
        assert extent.end == 1500
    
    def test_extent_finite_zero_delay(self):
        source = PiecewisePE([(0, 0.0), (1000, 1.0)])
        delay = DelayPE(source, delay=0)
        
        extent = delay.extent()
        assert extent.start == 0
        assert extent.end == 1000
    
    def test_extent_finite_negative_delay(self):
        source = PiecewisePE([(0, 0.0), (1000, 1.0)])
        delay = DelayPE(source, delay=-200)
        
        extent = delay.extent()
        assert extent.start == -200
        assert extent.end == 800
    
    def test_extent_infinite_source(self):
        source = ConstantPE(1.0)  # Infinite extent
        delay = DelayPE(source, delay=500)
        
        extent = delay.extent()
        assert extent.start is None
        assert extent.end is None


class TestDelayPEIntegerRender:
    """Test DelayPE rendering with integer delay."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_render_constant_with_delay(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100)
        
        self.renderer.set_source(delay)
        
        snippet = delay.render(0, 50)
        assert snippet.start == 0
        assert snippet.duration == 50
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 1), 1.0, dtype=np.float32)
        )
    
    def test_render_ramp_delayed(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        delay = DelayPE(source, delay=50)
        
        self.renderer.set_source(delay)
        
        # Before delayed start - DelayPE looks at source time -50, which is before
        # PiecewisePE's extent (0-100), so PiecewisePE returns zeros (its ExtendMode.ZERO behavior)
        snippet = delay.render(0, 50)
        np.testing.assert_array_equal(
            snippet.data, np.zeros((50, 1), dtype=np.float32)
        )
        
        # At delayed start: source ramp [0,100) gives values 0..0.99; we read 0..49 → 0, 0.01, ..., 0.49
        snippet = delay.render(50, 50)
        expected = np.linspace(0.0, 0.49, 50, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
    
    def test_render_zero_delay(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        delay = DelayPE(source, delay=0)
        
        self.renderer.set_source(delay)
        
        snippet = delay.render(0, 50)
        source_snippet = source.render(0, 50)
        np.testing.assert_array_equal(snippet.data, source_snippet.data)
    
    def test_render_negative_delay(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        delay = DelayPE(source, delay=-25)
        
        self.renderer.set_source(delay)
        
        # At position 0, we get source position 25
        snippet = delay.render(0, 50)
        source_snippet = source.render(25, 50)
        np.testing.assert_array_equal(snippet.data, source_snippet.data)
    
    def test_render_stereo(self):
        source = ConstantPE(0.5, channels=2)
        delay = DelayPE(source, delay=100)
        
        self.renderer.set_source(delay)
        
        snippet = delay.render(0, 50)
        assert snippet.channels == 2
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 2), 0.5, dtype=np.float32)
        )


# =============================================================================
# Float Delay Tests (Constant Fractional Delay)
# =============================================================================

class TestDelayPEFloatBasics:
    """Test DelayPE with constant fractional (float) delay."""
    
    def test_create_float_delay(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100.5)
        assert delay.delay == 100.5
        assert delay.interpolation == InterpolationMode.LINEAR
    
    def test_create_float_delay_cubic(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100.5, interpolation=InterpolationMode.CUBIC)
        assert delay.interpolation == InterpolationMode.CUBIC
    
    def test_inputs_float(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100.5)
        # Float delay: only source in inputs
        assert delay.inputs() == [source]
    
    def test_repr_float(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100.5)
        repr_str = repr(delay)
        assert "100.5" in repr_str
        assert "interpolation" in repr_str


class TestDelayPEFloatRender:
    """Test DelayPE rendering with fractional delay."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_fractional_delay_interpolation(self):
        """Test that fractional delay uses interpolation."""
        source = IdentityPE()  # Outputs sample index
        delay = DelayPE(source, delay=5.5)
        
        self.renderer.set_source(delay)
        
        # At t=100, delay=5.5 -> index=94.5
        # Linear interp: 0.5*94 + 0.5*95 = 94.5
        snippet = delay.render(100, 1)
        
        assert abs(snippet.data[0, 0] - 94.5) < 0.01
    
    def test_fractional_delay_vs_integer(self):
        """Fractional delay should differ from nearest integer delay."""
        source = IdentityPE()
        
        int_delay = DelayPE(source, delay=10)
        float_delay = DelayPE(source, delay=10.3)
        
        self.renderer.set_source(int_delay)
        int_snippet = int_delay.render(100, 1)
        
        self.renderer.set_source(float_delay)
        float_snippet = float_delay.render(100, 1)
        
        # Integer delay at t=100: source[90] = 90
        assert abs(int_snippet.data[0, 0] - 90.0) < 0.01
        
        # Float delay at t=100: source[89.7], interpolated
        # 0.7*89 + 0.3*90 = 62.3 + 27 = 89.3... wait that's wrong
        # floor(89.7) = 89, frac = 0.7
        # Linear: (1-0.7)*89 + 0.7*90 = 0.3*89 + 0.7*90 = 26.7 + 63 = 89.7
        assert abs(float_snippet.data[0, 0] - 89.7) < 0.01
    
    def test_fractional_delay_cubic(self):
        """Test cubic interpolation with fractional delay."""
        source = IdentityPE()
        delay = DelayPE(source, delay=10.5, interpolation=InterpolationMode.CUBIC)
        
        self.renderer.set_source(delay)
        
        # For linear data, cubic should match linear closely
        snippet = delay.render(100, 1)
        assert abs(snippet.data[0, 0] - 89.5) < 0.1


# =============================================================================
# PE Delay Tests (Variable Delay)
# =============================================================================

class TestDelayPEVariableBasics:
    """Test DelayPE with PE (variable) delay."""
    
    def test_create_pe_delay(self):
        source = IdentityPE()
        delay_pe = ConstantPE(10.0)
        
        delay = DelayPE(source, delay=delay_pe)
        
        assert delay.source is source
        assert delay.delay is delay_pe
        assert delay.interpolation == InterpolationMode.LINEAR
    
    def test_create_pe_delay_cubic(self):
        source = IdentityPE()
        delay_pe = ConstantPE(10.0)
        
        delay = DelayPE(source, delay=delay_pe, interpolation=InterpolationMode.CUBIC)
        
        assert delay.interpolation == InterpolationMode.CUBIC
    
    def test_inputs_pe(self):
        source = IdentityPE()
        delay_pe = ConstantPE(10.0)
        
        delay = DelayPE(source, delay=delay_pe)
        
        inputs = delay.inputs()
        assert len(inputs) == 2
        assert source in inputs
        assert delay_pe in inputs
    
    def test_extent_from_delay_pe(self):
        source = PiecewisePE([(0, 0.0), (1000, 1.0)])  # Extent: (0, 1000)
        delay_pe = PiecewisePE([(0, 0.0), (500, 100.0)])  # Extent: (0, 500)
        
        delay = DelayPE(source, delay=delay_pe)
        extent = delay.extent()
        
        # Extent should match delay PE, not source
        assert extent.start == 0
        assert extent.end == 500
    
    def test_repr_pe(self):
        source = IdentityPE()
        delay_pe = ConstantPE(10.0)
        
        delay = DelayPE(source, delay=delay_pe)
        repr_str = repr(delay)
        
        assert "DelayPE" in repr_str
        assert "IdentityPE" in repr_str
        assert "ConstantPE" in repr_str
        assert "interpolation" in repr_str


class TestDelayPEVariableSignConvention:
    """Test that variable delay uses same sign convention as integer delay."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_positive_delay_looks_into_past(self):
        """Positive delay should look into the past."""
        source = IdentityPE()
        delay_pe = ConstantPE(10.0)
        
        delay = DelayPE(source, delay=delay_pe)
        self.renderer.set_source(delay)
        
        # At t=100, with delay=10, we should get source[90] = 90
        snippet = delay.render(100, 1)
        
        assert abs(snippet.data[0, 0] - 90.0) < 0.01
    
    def test_matches_integer_delay_behavior(self):
        """Variable delay with constant PE should match integer delay."""
        source = IdentityPE()
        
        # Integer delay
        int_delay = DelayPE(source, delay=50)
        
        # PE delay with constant value
        pe_delay = DelayPE(source, delay=ConstantPE(50.0))
        
        self.renderer.set_source(int_delay)
        int_snippet = int_delay.render(100, 10)
        
        self.renderer.set_source(pe_delay)
        pe_snippet = pe_delay.render(100, 10)
        
        np.testing.assert_array_almost_equal(
            int_snippet.data,
            pe_snippet.data,
            decimal=4
        )
    
    def test_zero_delay_passthrough(self):
        """Zero delay should pass through unchanged."""
        source = IdentityPE()
        delay = DelayPE(source, delay=ConstantPE(0.0))
        
        self.renderer.set_source(delay)
        
        snippet = delay.render(50, 10)
        expected = np.arange(50, 60, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=4)


class TestDelayPEVariableRender:
    """Test variable delay rendering."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_varying_delay(self):
        """Test with time-varying delay."""
        source = IdentityPE()
        # Delay ramps from 0 to 10 over 10 samples so at t=k, delay=k → source index 0 for all
        delay_pe = PiecewisePE([(0, 0.0), (10, 10.0)])
        
        delay = DelayPE(source, delay=delay_pe)
        self.renderer.set_source(delay)
        
        # At t=k, delay=k, so index=0 for all samples
        snippet = delay.render(0, 10)
        
        expected = np.zeros((10, 1), dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=3)
    
    def test_fractional_pe_delay(self):
        """Test PE delay with fractional values."""
        source = IdentityPE()
        delay_pe = ConstantPE(5.5)
        
        delay = DelayPE(source, delay=delay_pe)
        self.renderer.set_source(delay)
        
        # At t=100, delay=5.5 -> index=94.5
        snippet = delay.render(100, 1)
        
        assert abs(snippet.data[0, 0] - 94.5) < 0.01


class TestDelayPEVariableOutOfBounds:
    """Test out-of-bounds handling with variable delay."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_out_of_bounds_past(self):
        """Accessing before source extent returns zero."""
        source = CropPE(IdentityPE(), 100, (200) - (100))
        delay_pe = ConstantPE(200.0)
        
        delay = DelayPE(source, delay=delay_pe)
        self.renderer.set_source(delay)
        
        # At t=150, delay=200 -> index=-50 (out of bounds)
        snippet = delay.render(150, 10)
        
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((10, 1), dtype=np.float32),
            decimal=5
        )
    
    def test_out_of_bounds_future(self):
        """Accessing past source extent returns zero."""
        source = CropPE(IdentityPE(), 0, (100) - (0))
        delay_pe = ConstantPE(-50.0)
        
        delay = DelayPE(source, delay=delay_pe)
        self.renderer.set_source(delay)
        
        # At t=80, delay=-50 -> index=130 (out of bounds)
        snippet = delay.render(80, 10)
        
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((10, 1), dtype=np.float32),
            decimal=5
        )


# =============================================================================
# Chaining and Integration Tests
# =============================================================================

class TestDelayPEChaining:
    """Test DelayPE in chains with other PEs."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_double_delay(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        delay1 = DelayPE(source, delay=50)
        delay2 = DelayPE(delay1, delay=50)
        
        # Total delay should be 100
        extent = delay2.extent()
        assert extent.start == 100
        assert extent.end == 200
    
    def test_delay_with_mix(self):
        source = ConstantPE(1.0)
        delayed = DelayPE(source, delay=50)
        mixed = MixPE(source, delayed)

        self.renderer.set_source(mixed)

        # Both contribute, so sum is 2.0
        snippet = mixed.render(0, 50)
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 1), 2.0, dtype=np.float32)
        )
    
    def test_delay_preserves_purity_chain(self):
        source = ConstantPE(1.0)
        delay1 = DelayPE(source, delay=10)
        delay2 = DelayPE(delay1, delay=20)
        
        assert delay2.is_pure() is True


class TestDelayPEIntegration:
    """Integration tests for DelayPE with the renderer."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_full_render_cycle(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        delay = DelayPE(source, delay=100)
        
        self.renderer.set_source(delay)
        
        with self.renderer:
            self.renderer.start()
            
            assert delay.extent().start == 100
            assert delay.extent().end == 200
            
            snippet = delay.render(100, 100)
            # Source ramp [0,100) gives 0..0.99; we read 0..99
            expected = np.linspace(0.0, 0.99, 100, dtype=np.float32).reshape(-1, 1)
            np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
    
    def test_large_delay(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=1_000_000)
        
        self.renderer.set_source(delay)
        
        snippet = delay.render(1_000_000, 100)
        np.testing.assert_array_equal(
            snippet.data, np.full((100, 1), 1.0, dtype=np.float32)
        )


class TestDelayPEStereo:
    """Test stereo handling across all delay modes."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_stereo_integer_delay(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)], channels=2)
        delay = DelayPE(source, delay=50)
        
        self.renderer.set_source(delay)
        
        snippet = delay.render(50, 10)
        assert snippet.channels == 2
    
    def test_stereo_float_delay(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)], channels=2)
        delay = DelayPE(source, delay=50.5)
        
        self.renderer.set_source(delay)
        
        snippet = delay.render(50, 10)
        assert snippet.channels == 2
    
    def test_stereo_pe_delay(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)], channels=2)
        delay_pe = ConstantPE(50.0)
        delay = DelayPE(source, delay=delay_pe)
        
        self.renderer.set_source(delay)
        
        snippet = delay.render(50, 10)
        assert snippet.channels == 2
