"""
Tests for DelayPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    DelayPE,
    ConstantPE,
    RampPE,
    SinePE,
    MixPE,
    NullRenderer,
    Extent,
)


class TestDelayPEBasics:
    """Test basic DelayPE creation and properties."""
    
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
    
    def test_inputs(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100)
        assert delay.inputs() == [source]
    
    def test_is_pure(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100)
        assert delay.is_pure() is True
    
    def test_channel_count_passthrough(self):
        source = ConstantPE(1.0, channels=2)
        delay = DelayPE(source, delay=100)
        assert delay.channel_count() == 2
    
    def test_repr(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100)
        repr_str = repr(delay)
        assert "DelayPE" in repr_str
        assert "ConstantPE" in repr_str
        assert "100" in repr_str


class TestDelayPEExtent:
    """Test DelayPE extent calculation."""
    
    def test_extent_finite_positive_delay(self):
        source = RampPE(0.0, 1.0, duration=1000)
        delay = DelayPE(source, delay=500)
        
        # Source extent is (0, 1000), delayed should be (500, 1500)
        extent = delay.extent()
        assert extent.start == 500
        assert extent.end == 1500
    
    def test_extent_finite_zero_delay(self):
        source = RampPE(0.0, 1.0, duration=1000)
        delay = DelayPE(source, delay=0)
        
        extent = delay.extent()
        assert extent.start == 0
        assert extent.end == 1000
    
    def test_extent_finite_negative_delay(self):
        source = RampPE(0.0, 1.0, duration=1000)
        delay = DelayPE(source, delay=-200)
        
        # Source extent is (0, 1000), delayed should be (-200, 800)
        extent = delay.extent()
        assert extent.start == -200
        assert extent.end == 800
    
    def test_extent_infinite_source(self):
        source = ConstantPE(1.0)  # Infinite extent
        delay = DelayPE(source, delay=500)
        
        # Infinite extent stays infinite
        extent = delay.extent()
        assert extent.start is None
        assert extent.end is None
    
    def test_extent_semi_infinite_start(self):
        # A hypothetical PE with (None, 1000) extent
        # We'll use ConstantPE and check the general behavior
        source = SinePE()  # Infinite extent
        delay = DelayPE(source, delay=100)
        
        extent = delay.extent()
        assert extent.start is None
        assert extent.end is None


class TestDelayPERender:
    """Test DelayPE rendering."""
    
    def test_render_constant_with_delay(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=100)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delay)
        renderer.start()
        
        # Render at any position - constant value
        snippet = delay.render(0, 50)
        assert snippet.start == 0
        assert snippet.duration == 50
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 1), 1.0, dtype=np.float32)
        )
        
        renderer.stop()
    
    def test_render_ramp_delayed(self):
        # Ramp from 0 to 1 over 100 samples
        source = RampPE(0.0, 1.0, duration=100)
        delay = DelayPE(source, delay=50)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delay)
        renderer.start()
        
        # Render before the delayed start (samples 0-49) - should be zeros
        snippet = delay.render(0, 50)
        np.testing.assert_array_equal(
            snippet.data, np.zeros((50, 1), dtype=np.float32)
        )
        
        # Render at the delayed start (samples 50-99) - should be ramp 0-49
        snippet = delay.render(50, 50)
        expected = np.linspace(0.0, 0.4949494949, 50, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        
        renderer.stop()
    
    def test_render_ramp_spanning_delay_boundary(self):
        # Ramp from 0 to 1 over 100 samples
        source = RampPE(0.0, 1.0, duration=100)
        delay = DelayPE(source, delay=50)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delay)
        renderer.start()
        
        # Render spanning the delay boundary (25-74)
        snippet = delay.render(25, 50)
        
        # First 25 samples (indices 25-49 in output = -25 to -1 in source) are zeros
        # Next 25 samples (indices 50-74 in output = 0 to 24 in source) are from ramp
        expected_zeros = np.zeros((25, 1), dtype=np.float32)
        expected_ramp = np.linspace(0.0, 0.24242424, 25, dtype=np.float32).reshape(-1, 1)
        expected = np.vstack([expected_zeros, expected_ramp])
        
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        
        renderer.stop()
    
    def test_render_zero_delay(self):
        source = RampPE(0.0, 1.0, duration=100)
        delay = DelayPE(source, delay=0)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delay)
        renderer.start()
        
        # Should be identical to source
        snippet = delay.render(0, 50)
        source_snippet = source.render(0, 50)
        np.testing.assert_array_equal(snippet.data, source_snippet.data)
        
        renderer.stop()
    
    def test_render_negative_delay(self):
        # Ramp from 0 to 1 over 100 samples
        source = RampPE(0.0, 1.0, duration=100)
        delay = DelayPE(source, delay=-25)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delay)
        renderer.start()
        
        # At position 0, we get source position 25
        snippet = delay.render(0, 50)
        source_snippet = source.render(25, 50)
        np.testing.assert_array_equal(snippet.data, source_snippet.data)
        
        renderer.stop()
    
    def test_render_stereo(self):
        source = ConstantPE(0.5, channels=2)
        delay = DelayPE(source, delay=100)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delay)
        renderer.start()
        
        snippet = delay.render(0, 50)
        assert snippet.channels == 2
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 2), 0.5, dtype=np.float32)
        )
        
        renderer.stop()


class TestDelayPEChaining:
    """Test DelayPE in chains with other PEs."""
    
    def test_double_delay(self):
        source = RampPE(0.0, 1.0, duration=100)
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
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(mixed)
        renderer.start()
        
        # Both contribute, so sum is 2.0
        snippet = mixed.render(0, 50)
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 1), 2.0, dtype=np.float32)
        )
        
        renderer.stop()
    
    def test_delay_preserves_purity_chain(self):
        source = ConstantPE(1.0)
        delay1 = DelayPE(source, delay=10)
        delay2 = DelayPE(delay1, delay=20)
        
        assert delay2.is_pure() is True


class TestDelayPEIntegration:
    """Integration tests for DelayPE with the renderer."""
    
    def test_full_render_cycle(self):
        source = RampPE(0.0, 1.0, duration=100)
        delay = DelayPE(source, delay=100)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delay)
        
        with renderer:
            renderer.start()
            
            # Verify extent
            assert delay.extent().start == 100
            assert delay.extent().end == 200
            
            # Render full extent
            snippet = delay.render(100, 100)
            expected = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(-1, 1)
            np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
    
    def test_large_delay(self):
        source = ConstantPE(1.0)
        delay = DelayPE(source, delay=1_000_000)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delay)
        renderer.start()
        
        # Should still work with large delay
        snippet = delay.render(1_000_000, 100)
        np.testing.assert_array_equal(
            snippet.data, np.full((100, 1), 1.0, dtype=np.float32)
        )
        
        renderer.stop()
