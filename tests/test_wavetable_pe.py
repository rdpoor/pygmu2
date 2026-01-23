"""
Tests for WavetablePE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    WavetablePE,
    InterpolationMode,
    OutOfBoundsMode,
    ConstantPE,
    RampPE,
    IdentityPE,
    NullRenderer,
    Extent,
)


class TestWavetablePEBasics:
    """Test basic WavetablePE creation and properties."""
    
    def test_create_wavetable_pe(self):
        wavetable = RampPE(0.0, 1.0, duration=100)
        indexer = ConstantPE(50.0)
        
        wt_pe = WavetablePE(wavetable, indexer)
        
        assert wt_pe.wavetable is wavetable
        assert wt_pe.indexer is indexer
        assert wt_pe.interpolation == InterpolationMode.LINEAR
        assert wt_pe.out_of_bounds == OutOfBoundsMode.ZERO
    
    def test_create_with_options(self):
        wavetable = RampPE(0.0, 1.0, duration=100)
        indexer = ConstantPE(50.0)
        
        wt_pe = WavetablePE(
            wavetable,
            indexer,
            interpolation=InterpolationMode.CUBIC,
            out_of_bounds=OutOfBoundsMode.WRAP,
        )
        
        assert wt_pe.interpolation == InterpolationMode.CUBIC
        assert wt_pe.out_of_bounds == OutOfBoundsMode.WRAP
    
    def test_inputs(self):
        wavetable = RampPE(0.0, 1.0, duration=100)
        indexer = ConstantPE(50.0)
        
        wt_pe = WavetablePE(wavetable, indexer)
        
        inputs = wt_pe.inputs()
        assert len(inputs) == 2
        assert wavetable in inputs
        assert indexer in inputs
    
    def test_is_pure(self):
        wavetable = RampPE(0.0, 1.0, duration=100)
        indexer = ConstantPE(50.0)
        
        wt_pe = WavetablePE(wavetable, indexer)
        assert wt_pe.is_pure() is True
    
    def test_channel_count_from_wavetable(self):
        wavetable = RampPE(0.0, 1.0, duration=100, channels=2)
        indexer = ConstantPE(50.0)
        
        wt_pe = WavetablePE(wavetable, indexer)
        assert wt_pe.channel_count() == 2
    
    def test_extent_from_indexer(self):
        wavetable = RampPE(0.0, 1.0, duration=100)  # Extent: (0, 100)
        indexer = RampPE(0.0, 99.0, duration=200)   # Extent: (0, 200)
        
        wt_pe = WavetablePE(wavetable, indexer)
        extent = wt_pe.extent()
        
        # Extent should match indexer, not wavetable
        assert extent.start == 0
        assert extent.end == 200
    
    def test_repr(self):
        wavetable = RampPE(0.0, 1.0, duration=100)
        indexer = ConstantPE(50.0)
        
        wt_pe = WavetablePE(wavetable, indexer)
        repr_str = repr(wt_pe)
        
        assert "WavetablePE" in repr_str
        assert "RampPE" in repr_str
        assert "ConstantPE" in repr_str
        assert "linear" in repr_str
        assert "zero" in repr_str


class TestWavetablePELinearInterpolation:
    """Test linear interpolation."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_integer_indices(self):
        """Test lookup at integer indices (no interpolation needed)."""
        # Wavetable: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        wavetable = IdentityPE()  # Outputs sample index as value
        indexer = ConstantPE(5.0)  # Always look up index 5
        
        wt_pe = WavetablePE(wavetable, indexer)
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        # All values should be 5.0
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((10, 1), 5.0, dtype=np.float32),
            decimal=5
        )
    
    def test_fractional_indices(self):
        """Test interpolation at fractional indices."""
        # Wavetable: [0, 1, 2, 3, ...]
        wavetable = IdentityPE()
        indexer = ConstantPE(2.5)  # Halfway between 2 and 3
        
        wt_pe = WavetablePE(wavetable, indexer)
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        # Linear interp: 0.5*2 + 0.5*3 = 2.5
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((10, 1), 2.5, dtype=np.float32),
            decimal=5
        )
    
    def test_varying_indices(self):
        """Test with time-varying indexer."""
        # Wavetable: [0, 1, 2, ..., 99]
        wavetable = IdentityPE()
        # Indexer: ramp from 0 to 9 over 10 samples
        indexer = RampPE(0.0, 9.0, duration=10)
        
        wt_pe = WavetablePE(wavetable, indexer)
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        # Output should be [0, 1, 2, ..., 9]
        expected = np.arange(10, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=4)
    
    def test_stereo_wavetable(self):
        """Test with stereo wavetable."""
        wavetable = RampPE(0.0, 1.0, duration=100, channels=2)
        indexer = ConstantPE(50.0)
        
        wt_pe = WavetablePE(wavetable, indexer)
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        assert snippet.channels == 2
        # Both channels should be approximately 0.5
        assert abs(snippet.data[0, 0] - 0.5) < 0.02
        assert abs(snippet.data[0, 1] - 0.5) < 0.02


class TestWavetablePECubicInterpolation:
    """Test cubic interpolation."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_cubic_integer_indices(self):
        """Cubic interpolation at integer indices should match the value."""
        wavetable = IdentityPE()
        indexer = ConstantPE(5.0)
        
        wt_pe = WavetablePE(
            wavetable, indexer,
            interpolation=InterpolationMode.CUBIC
        )
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        # At integer index, cubic should return the exact value
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((10, 1), 5.0, dtype=np.float32),
            decimal=4
        )
    
    def test_cubic_smoother_than_linear(self):
        """
        Cubic interpolation should produce smoother results.
        For a linear ramp, both should match. For more complex data,
        cubic would differ.
        """
        # For a simple linear ramp, both should give similar results
        wavetable = IdentityPE()
        indexer = ConstantPE(2.5)
        
        linear_pe = WavetablePE(
            wavetable, indexer,
            interpolation=InterpolationMode.LINEAR
        )
        cubic_pe = WavetablePE(
            wavetable, indexer,
            interpolation=InterpolationMode.CUBIC
        )
        
        self.renderer.set_source(linear_pe)
        linear_snippet = linear_pe.render(0, 1)
        
        self.renderer.set_source(cubic_pe)
        cubic_snippet = cubic_pe.render(0, 1)
        
        # For linear data, both should be close
        np.testing.assert_array_almost_equal(
            linear_snippet.data,
            cubic_snippet.data,
            decimal=3
        )


class TestWavetablePEOutOfBounds:
    """Test out-of-bounds handling."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_zero_mode_out_of_bounds(self):
        """ZERO mode: output 0 for out-of-bounds indices."""
        # Wavetable extent: [0, 100)
        wavetable = RampPE(1.0, 2.0, duration=100)  # Values 1.0 to 2.0
        indexer = ConstantPE(150.0)  # Way out of bounds
        
        wt_pe = WavetablePE(
            wavetable, indexer,
            out_of_bounds=OutOfBoundsMode.ZERO
        )
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        # All values should be 0 (out of bounds)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((10, 1), dtype=np.float32),
            decimal=5
        )
    
    def test_zero_mode_negative_index(self):
        """ZERO mode: output 0 for negative indices."""
        wavetable = RampPE(1.0, 2.0, duration=100)
        indexer = ConstantPE(-10.0)  # Negative index
        
        wt_pe = WavetablePE(
            wavetable, indexer,
            out_of_bounds=OutOfBoundsMode.ZERO
        )
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((10, 1), dtype=np.float32),
            decimal=5
        )
    
    def test_clamp_mode_high(self):
        """CLAMP mode: clamp to max valid index."""
        # Wavetable: values from 0.0 to 1.0 over 100 samples
        wavetable = RampPE(0.0, 1.0, duration=100)
        indexer = ConstantPE(150.0)  # Out of bounds high
        
        wt_pe = WavetablePE(
            wavetable, indexer,
            out_of_bounds=OutOfBoundsMode.CLAMP
        )
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        # Should clamp to index 99, which has value ~1.0
        assert snippet.data[0, 0] > 0.98
    
    def test_clamp_mode_low(self):
        """CLAMP mode: clamp to min valid index."""
        wavetable = RampPE(0.0, 1.0, duration=100)
        indexer = ConstantPE(-50.0)  # Out of bounds low
        
        wt_pe = WavetablePE(
            wavetable, indexer,
            out_of_bounds=OutOfBoundsMode.CLAMP
        )
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        # Should clamp to index 0, which has value 0.0
        assert abs(snippet.data[0, 0]) < 0.02
    
    def test_wrap_mode(self):
        """WRAP mode: wrap indices around wavetable length."""
        # Wavetable: 10 samples with values [0, 1, 2, ..., 9]
        wavetable = IdentityPE()
        # Create a wavetable PE that limits to 10 samples
        from pygmu2 import CropPE
        cropped_wavetable = CropPE(wavetable, Extent(0, 10))
        
        indexer = ConstantPE(12.0)  # Should wrap to 12 % 10 = 2
        
        wt_pe = WavetablePE(
            cropped_wavetable, indexer,
            out_of_bounds=OutOfBoundsMode.WRAP
        )
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        # Index 12 wraps to 2, value should be ~2.0
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((10, 1), 2.0, dtype=np.float32),
            decimal=4
        )
    
    def test_wrap_mode_negative(self):
        """WRAP mode: handle negative indices."""
        from pygmu2 import CropPE
        wavetable = IdentityPE()
        cropped_wavetable = CropPE(wavetable, Extent(0, 10))
        
        indexer = ConstantPE(-2.0)  # Should wrap to (-2 % 10) = 8
        
        wt_pe = WavetablePE(
            cropped_wavetable, indexer,
            out_of_bounds=OutOfBoundsMode.WRAP
        )
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        # Index -2 wraps to 8
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((10, 1), 8.0, dtype=np.float32),
            decimal=4
        )


class TestWavetablePEArbitraryExtent:
    """Test wavetables with non-zero starting extent."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_wavetable_offset_start(self):
        """Test wavetable that doesn't start at 0."""
        from pygmu2 import CropPE, DelayPE
        
        # Create wavetable starting at sample 100
        base = IdentityPE()  # Values = sample index
        delayed = DelayPE(base, delay=100)  # Now extent is (100, None)
        wavetable = CropPE(delayed, Extent(100, 200))  # Extent: (100, 200)
        
        # Index into the middle: 150
        indexer = ConstantPE(150.0)
        
        wt_pe = WavetablePE(wavetable, indexer)
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 10)
        
        # Index 150 in delayed IdentityPE gives value 50 (150 - 100 delay)
        # Wait, DelayPE shifts the extent but the values are still the original
        # At sample index 150, DelayPE reads from source at 150-100=50, giving value 50
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((10, 1), 50.0, dtype=np.float32),
            decimal=4
        )


class TestWavetablePEEdgeCases:
    """Test edge cases."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_single_sample(self):
        """Test rendering a single sample."""
        wavetable = IdentityPE()
        indexer = ConstantPE(5.0)
        
        wt_pe = WavetablePE(wavetable, indexer)
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 1)
        
        assert snippet.duration == 1
        assert abs(snippet.data[0, 0] - 5.0) < 0.01
    
    def test_boundary_interpolation(self):
        """Test interpolation right at boundary of wavetable."""
        from pygmu2 import CropPE
        
        wavetable = CropPE(IdentityPE(), Extent(0, 10))  # [0, 1, ..., 9]
        indexer = ConstantPE(8.5)  # Between 8 and 9
        
        wt_pe = WavetablePE(wavetable, indexer)
        self.renderer.set_source(wt_pe)
        
        snippet = wt_pe.render(0, 1)
        
        # Linear interp: 0.5*8 + 0.5*9 = 8.5
        assert abs(snippet.data[0, 0] - 8.5) < 0.1
    
    def test_indexer_with_infinite_extent(self):
        """Test with indexer that has infinite extent (like ConstantPE)."""
        wavetable = RampPE(0.0, 1.0, duration=100)
        indexer = ConstantPE(50.0)  # Infinite extent
        
        wt_pe = WavetablePE(wavetable, indexer)
        
        # WavetablePE extent should be infinite (from indexer)
        extent = wt_pe.extent()
        assert extent.start is None
        assert extent.end is None
