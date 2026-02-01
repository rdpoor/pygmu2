"""
Tests for MixPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import MixPE, ConstantPE, SinePE, PiecewisePE, NullRenderer, Extent


class TestMixPEBasics:
    """Test basic MixPE creation and properties."""
    
    def test_create_mix_pe(self):
        source1 = ConstantPE(0.5)
        source2 = ConstantPE(0.3)
        mix = MixPE(source1, source2)
        assert len(mix.inputs()) == 2
    
    def test_create_with_three_inputs(self):
        source1 = ConstantPE(0.3)
        source2 = ConstantPE(0.3)
        source3 = ConstantPE(0.3)
        mix = MixPE(source1, source2, source3)
        assert len(mix.inputs()) == 3
    
    def test_requires_at_least_two_inputs(self):
        source = ConstantPE(0.5)
        with pytest.raises(ValueError, match="at least 2 inputs"):
            MixPE(source)
    
    def test_requires_at_least_two_inputs_empty(self):
        with pytest.raises(ValueError, match="at least 2 inputs"):
            MixPE()
    
    def test_is_pure(self):
        source1 = ConstantPE(0.5)
        source2 = ConstantPE(0.3)
        mix = MixPE(source1, source2)
        assert mix.is_pure() is True
    
    def test_inputs_returns_all_inputs(self):
        source1 = ConstantPE(0.5)
        source2 = ConstantPE(0.3)
        source3 = ConstantPE(0.2)
        mix = MixPE(source1, source2, source3)
        
        inputs = mix.inputs()
        assert source1 in inputs
        assert source2 in inputs
        assert source3 in inputs
    
    def test_repr(self):
        source1 = ConstantPE(0.5)
        source2 = SinePE()
        mix = MixPE(source1, source2)
        
        repr_str = repr(mix)
        assert "MixPE" in repr_str
        assert "ConstantPE" in repr_str
        assert "SinePE" in repr_str


class TestMixPERender:
    """Test MixPE rendering."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_render_adds_values(self):
        """Test that mix adds values together."""
        source1 = ConstantPE(0.3)
        source2 = ConstantPE(0.4)
        mix = MixPE(source1, source2)
        
        self.renderer.set_source(mix)
        snippet = mix.render(0, 100)
        
        # 0.3 + 0.4 = 0.7
        expected = np.full((100, 1), 0.7, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_render_three_inputs(self):
        """Test mixing three inputs."""
        source1 = ConstantPE(0.2)
        source2 = ConstantPE(0.3)
        source3 = ConstantPE(0.4)
        mix = MixPE(source1, source2, source3)
        
        self.renderer.set_source(mix)
        snippet = mix.render(0, 100)
        
        # 0.2 + 0.3 + 0.4 = 0.9
        expected = np.full((100, 1), 0.9, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_render_stereo(self):
        """Test mixing stereo inputs."""
        source1 = ConstantPE(0.3, channels=2)
        source2 = ConstantPE(0.4, channels=2)
        mix = MixPE(source1, source2)
        
        self.renderer.set_source(mix)
        snippet = mix.render(0, 100)
        
        assert snippet.channels == 2
        expected = np.full((100, 2), 0.7, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_render_with_negative_values(self):
        """Test that negative values cancel positive."""
        source1 = ConstantPE(0.5)
        source2 = ConstantPE(-0.3)
        mix = MixPE(source1, source2)
        
        self.renderer.set_source(mix)
        snippet = mix.render(0, 100)
        
        # 0.5 + (-0.3) = 0.2
        expected = np.full((100, 1), 0.2, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_render_sine_waves(self):
        """Test mixing sine waves (time-varying signals)."""
        sine1 = SinePE(frequency=440.0, amplitude=0.5)
        sine2 = SinePE(frequency=550.0, amplitude=0.5)
        mix = MixPE(sine1, sine2)
        
        self.renderer.set_source(mix)
        snippet = mix.render(0, 1000)
        
        # Result should be sum of two sines
        # Max possible is 1.0 (when both align at peaks)
        assert np.max(snippet.data) <= 1.0 + 1e-6
        # Min possible is -1.0
        assert np.min(snippet.data) >= -1.0 - 1e-6
    
    def test_render_can_exceed_one(self):
        """Test that mixing can produce values > 1.0 (no automatic limiting)."""
        source1 = ConstantPE(0.8)
        source2 = ConstantPE(0.8)
        mix = MixPE(source1, source2)
        
        self.renderer.set_source(mix)
        snippet = mix.render(0, 100)
        
        # 0.8 + 0.8 = 1.6 (exceeds 1.0, no clipping)
        expected = np.full((100, 1), 1.6, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)


class TestMixPEExtent:
    """Test MixPE extent handling."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_extent_union_infinite(self):
        """Infinite extents result in infinite union."""
        source1 = ConstantPE(0.5)  # Infinite extent
        source2 = ConstantPE(0.3)  # Infinite extent
        mix = MixPE(source1, source2)
        
        extent = mix.extent()
        assert extent.start is None
        assert extent.end is None
    
    def test_extent_union_finite(self):
        """Finite extents union to cover full range."""
        ramp1 = PiecewisePE([(0, 0.0), (100, 1.0)])  # 0 to 100
        ramp2 = PiecewisePE([(0, 0.0), (200, 1.0)])  # 0 to 200
        mix = MixPE(ramp1, ramp2)
        
        extent = mix.extent()
        assert extent.start == 0
        assert extent.end == 200  # Union covers 0-200
    
    def test_extent_union_mixed(self):
        """Mix of infinite and finite gives infinite."""
        ramp = PiecewisePE([(0, 0.0), (100, 1.0)])  # Finite: 0 to 100
        constant = ConstantPE(0.5)  # Infinite
        mix = MixPE(ramp, constant)
        
        extent = mix.extent()
        assert extent.start is None  # Infinite from constant
        assert extent.end is None


class TestMixPEChannels:
    """Test MixPE channel handling."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_channel_count_from_inputs(self):
        """MixPE gets channel count from first input."""
        source1 = ConstantPE(0.5, channels=2)
        source2 = ConstantPE(0.3, channels=2)
        mix = MixPE(source1, source2)
        
        # channel_count returns first input's channel count
        assert mix.channel_count() == 2
        
        # Renderer also reports same channel count
        self.renderer.set_source(mix)
        assert self.renderer.channel_count == 2
    
    def test_resolve_channel_count_matches(self):
        """resolve_channel_count succeeds when all match."""
        source1 = ConstantPE(0.5, channels=2)
        source2 = ConstantPE(0.3, channels=2)
        mix = MixPE(source1, source2)
        
        result = mix.resolve_channel_count([2, 2])
        assert result == 2
    
    def test_resolve_channel_count_mismatch(self):
        """resolve_channel_count fails on mismatch."""
        source1 = ConstantPE(0.5, channels=1)
        source2 = ConstantPE(0.3, channels=2)
        mix = MixPE(source1, source2)
        
        with pytest.raises(ValueError, match="channel mismatch"):
            mix.resolve_channel_count([1, 2])


class TestMixPEIntegration:
    """Integration tests for MixPE with real graphs."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_nested_mix(self):
        """Test nesting MixPEs."""
        a = ConstantPE(0.1)
        b = ConstantPE(0.2)
        c = ConstantPE(0.3)
        d = ConstantPE(0.4)
        
        mix1 = MixPE(a, b)  # 0.3
        mix2 = MixPE(c, d)  # 0.7
        final = MixPE(mix1, mix2)  # 1.0
        
        self.renderer.set_source(final)
        snippet = final.render(0, 100)
        
        expected = np.full((100, 1), 1.0, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_mix_with_modulated_sine(self):
        """Test mixing with modulated sources."""
        # Two sines at different frequencies
        sine1 = SinePE(frequency=440.0, amplitude=0.3)
        sine2 = SinePE(frequency=880.0, amplitude=0.3)
        
        # Add a DC offset
        dc = ConstantPE(0.1)
        
        mix = MixPE(sine1, sine2, dc)
        
        self.renderer.set_source(mix)
        self.renderer.start()
        
        snippet = mix.render(0, 44100)  # One second
        
        # Check DC offset is present (mean should be around 0.1)
        mean = np.mean(snippet.data)
        assert abs(mean - 0.1) < 0.01
        
        self.renderer.stop()
