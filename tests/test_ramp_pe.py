"""
Tests for RampPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import RampPE, NullRenderer, Extent, ExtendMode, CropPE


class TestRampPEBasics:
    """Test basic RampPE creation and properties."""
    
    def test_create_ramp_pe(self):
        ramp = RampPE(0.0, 1.0, duration=100)
        assert ramp.start_value == 0.0
        assert ramp.end_value == 1.0
        assert ramp.ramp_duration == 100
        assert ramp.channel_count() == 1
    
    def test_create_with_channels(self):
        ramp = RampPE(0.0, 1.0, duration=100, channels=2)
        assert ramp.channel_count() == 2
    
    def test_finite_extent(self):
        ramp = RampPE(0.0, 1.0, duration=100)
        extent = ramp.extent()
        assert extent.start == 0
        assert extent.end == 100
    
    def test_infinite_extent_with_hold(self):
        """RampPE with ExtendMode.HOLD_BOTH has infinite extent."""
        ramp = RampPE(0.0, 1.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        extent = ramp.extent()
        assert extent.start is None
        assert extent.end is None
    
    def test_extend_mode_property(self):
        ramp = RampPE(0.0, 1.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        assert ramp.extend_mode == ExtendMode.HOLD_BOTH
        
        ramp2 = RampPE(0.0, 1.0, duration=100)
        assert ramp2.extend_mode == ExtendMode.ZERO
    
    def test_is_pure(self):
        ramp = RampPE(0.0, 1.0, duration=100)
        assert ramp.is_pure() is True
    
    def test_no_inputs(self):
        ramp = RampPE(0.0, 1.0, duration=100)
        assert ramp.inputs() == []
    
    def test_repr(self):
        ramp = RampPE(0.0, 1.0, duration=100)
        repr_str = repr(ramp)
        assert "RampPE" in repr_str
        assert "0.0" in repr_str
        assert "1.0" in repr_str
        assert "100" in repr_str


class TestRampPERender:
    """Test RampPE rendering."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_render_returns_snippet(self):
        ramp = RampPE(0.0, 1.0, duration=100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(0, 100)
        assert snippet.start == 0
        assert snippet.duration == 100
        assert snippet.channels == 1
    
    def test_render_full_ramp(self):
        ramp = RampPE(0.0, 1.0, duration=100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(0, 100)
        
        # First value should be 0
        assert abs(snippet.data[0, 0] - 0.0) < 1e-6
        # Last value should be 1
        assert abs(snippet.data[-1, 0] - 1.0) < 1e-6
        # Middle value should be ~0.5
        assert abs(snippet.data[49, 0] - 0.5) < 0.02
    
    def test_render_descending_ramp(self):
        ramp = RampPE(1.0, 0.0, duration=100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(0, 100)
        
        # First value should be 1
        assert abs(snippet.data[0, 0] - 1.0) < 1e-6
        # Last value should be 0
        assert abs(snippet.data[-1, 0] - 0.0) < 1e-6
    
    def test_render_stereo(self):
        ramp = RampPE(0.0, 1.0, duration=100, channels=2)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(0, 100)
        assert snippet.channels == 2
        # Both channels should be identical
        np.testing.assert_array_equal(snippet.data[:, 0], snippet.data[:, 1])
    
    def test_render_partial_start(self):
        """Render only the beginning of the ramp."""
        ramp = RampPE(0.0, 1.0, duration=100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(0, 50)
        
        # First value should be 0
        assert abs(snippet.data[0, 0] - 0.0) < 1e-6
        # Should end around 0.49
        assert abs(snippet.data[-1, 0] - 0.49) < 0.02
    
    def test_render_partial_middle(self):
        """Render the middle portion of the ramp."""
        ramp = RampPE(0.0, 1.0, duration=100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(25, 50)
        
        # Should start around 0.25
        assert abs(snippet.data[0, 0] - 0.25) < 0.02
        # Should end around 0.74
        assert abs(snippet.data[-1, 0] - 0.74) < 0.02
    
    def test_render_partial_end(self):
        """Render only the end of the ramp."""
        ramp = RampPE(0.0, 1.0, duration=100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(50, 50)
        
        # Should end at 1.0
        assert abs(snippet.data[-1, 0] - 1.0) < 1e-6
    
    def test_render_before_extent(self):
        """Render before the ramp starts."""
        ramp = RampPE(0.0, 1.0, duration=100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(-50, 50)
        
        # All values should be zero (before ramp)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((50, 1), dtype=np.float32)
        )
    
    def test_render_after_extent(self):
        """Render after the ramp ends."""
        ramp = RampPE(0.0, 1.0, duration=100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(100, 50)
        
        # All values should be zero (after ramp)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((50, 1), dtype=np.float32)
        )
    
    def test_render_spanning_start(self):
        """Render spanning the start of the ramp."""
        ramp = RampPE(0.0, 1.0, duration=100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(-25, 50)
        
        # First 25 samples should be zero
        np.testing.assert_array_almost_equal(
            snippet.data[:25, 0],
            np.zeros(25, dtype=np.float32)
        )
        # Sample 25 should be 0 (start of ramp)
        assert abs(snippet.data[25, 0] - 0.0) < 1e-6
    
    def test_render_spanning_end(self):
        """Render spanning the end of the ramp."""
        ramp = RampPE(0.0, 1.0, duration=100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(75, 50)
        
        # First 25 samples should be ramp values (75-99)
        # Sample 24 should be ~1.0 (end of ramp at index 99)
        assert abs(snippet.data[24, 0] - 1.0) < 1e-6
        # Remaining samples should be zero
        np.testing.assert_array_almost_equal(
            snippet.data[25:, 0],
            np.zeros(25, dtype=np.float32)
        )
    
    def test_render_large_values(self):
        """Test ramp with values outside [-1, 1] for frequency sweeps."""
        ramp = RampPE(220.0, 880.0, duration=44100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(0, 100)
        
        # First value should be 220
        assert abs(snippet.data[0, 0] - 220.0) < 0.1
    
    def test_render_constant_ramp(self):
        """Test ramp where start equals end (constant)."""
        ramp = RampPE(0.5, 0.5, duration=100)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(0, 100)
        
        # All values should be 0.5
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((100, 1), 0.5, dtype=np.float32)
        )
    
    def test_extend_mode_before_ramp(self):
        """With extend_mode=ExtendMode.HOLD_BOTH, values before ramp hold start_value."""
        ramp = RampPE(0.0, 1.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(-50, 50)
        
        # All values should be start_value (0.0) before ramp
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((50, 1), dtype=np.float32)
        )
    
    def test_extend_mode_after_ramp(self):
        """With extend_mode=ExtendMode.HOLD_BOTH, values after ramp hold end_value."""
        ramp = RampPE(0.0, 1.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(100, 50)
        
        # All values should be end_value (1.0) after ramp
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.ones((50, 1), dtype=np.float32)
        )
    
    def test_extend_mode_spanning_start(self):
        """Render spanning start with extend_mode=ExtendMode.HOLD_BOTH."""
        ramp = RampPE(0.5, 1.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(-25, 50)
        
        # First 25 samples should hold start_value (0.5)
        np.testing.assert_array_almost_equal(
            snippet.data[:25, 0],
            np.full(25, 0.5, dtype=np.float32)
        )
        # Sample 25 should be 0.5 (start of ramp)
        assert abs(snippet.data[25, 0] - 0.5) < 1e-6
    
    def test_extend_mode_spanning_end(self):
        """Render spanning end with extend_mode=ExtendMode.HOLD_BOTH."""
        ramp = RampPE(0.0, 1.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        self.renderer.set_source(ramp)
        
        snippet = ramp.render(75, 50)
        
        # First 25 samples should be ramp values (75-99)
        # Sample 24 should be ~1.0 (end of ramp at index 99)
        assert abs(snippet.data[24, 0] - 1.0) < 1e-6
        # Remaining samples should hold end_value (1.0)
        np.testing.assert_array_almost_equal(
            snippet.data[25:, 0],
            np.ones(25, dtype=np.float32)
        )
    
    def test_extend_mode_stereo(self):
        """extend_mode works with multi-channel ramps."""
        ramp = RampPE(0.0, 1.0, duration=100, channels=2, extend_mode=ExtendMode.HOLD_BOTH)
        self.renderer.set_source(ramp)
        
        # Before ramp: hold start_value
        snippet_before = ramp.render(-10, 10)
        np.testing.assert_array_almost_equal(
            snippet_before.data,
            np.zeros((10, 2), dtype=np.float32)
        )
        
        # After ramp: hold end_value
        snippet_after = ramp.render(100, 10)
        np.testing.assert_array_almost_equal(
            snippet_after.data,
            np.ones((10, 2), dtype=np.float32)
        )


class TestRampPERegression:
    """Regression tests for RampPE with extend_mode and cropping."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_ramp_with_extend_mode_has_infinite_extent(self):
        """
        Regression test: RampPE with extend_mode=ExtendMode.HOLD_BOTH should have infinite extent.
        """
        ramp = RampPE(0.0, 1.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        extent = ramp.extent()
        
        assert extent.start is None
        assert extent.end is None
    
    def test_crop_ramp_with_extend_mode_limits_output(self):
        """
        Regression test: CropPE should properly limit RampPE with infinite extent.
        
        RampPE with extend_mode=ExtendMode.HOLD_BOTH has infinite extent. This test verifies
        that CropPE correctly limits it to a finite window.
        """
        ramp = RampPE(10.0, 20.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        cropped = CropPE(ramp, Extent(0, 150))  # Crop to 150 samples
        
        self.renderer.set_source(cropped)
        
        # Render within crop window
        snippet1 = cropped.render(50, 50)
        # Should get ramp values (samples 50-99 are in ramp, 100-149 hold end_value)
        assert snippet1.data[0, 0] == pytest.approx(15.0, abs=0.1)  # Middle of ramp
        assert snippet1.data[49, 0] == pytest.approx(20.0, abs=0.1)  # End value
        
        # Render after crop window - should be zeros
        snippet2 = cropped.render(200, 50)
        np.testing.assert_array_almost_equal(
            snippet2.data,
            np.zeros((50, 1), dtype=np.float32),
            decimal=5
        )
    
    def test_crop_ramp_with_extend_mode_before_ramp(self):
        """
        Regression test: CropPE should handle RampPE with extend_mode when cropped
        before the ramp starts.
        """
        ramp = RampPE(5.0, 10.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        # Crop to start before ramp (at sample -50, extending to sample 50)
        cropped = CropPE(ramp, Extent(-50, 50))
        
        self.renderer.set_source(cropped)
        
        # Render the cropped region
        snippet = cropped.render(-50, 100)
        
        # First 50 samples should hold start_value (5.0)
        np.testing.assert_array_almost_equal(
            snippet.data[:50, 0],
            np.full(50, 5.0, dtype=np.float32),
            decimal=5
        )
        
        # Next 50 samples should be ramp values (0-49 of ramp)
        # Sample 50 should be approximately 5.0 (start of ramp)
        assert snippet.data[50, 0] == pytest.approx(5.0, abs=0.1)
    
