"""
Tests for ConstantPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import ConstantPE, NullRenderer, Extent


class TestConstantPEBasics:
    """Test basic ConstantPE creation and properties."""
    
    def test_create_constant_pe(self):
        const = ConstantPE(0.5)
        assert const.value == 0.5
        assert const.channel_count() == 1
    
    def test_create_with_channels(self):
        const = ConstantPE(0.5, channels=2)
        assert const.value == 0.5
        assert const.channel_count() == 2
    
    def test_infinite_extent(self):
        const = ConstantPE(1.0)
        extent = const.extent()
        assert extent.start is None
        assert extent.end is None
    
    def test_is_pure(self):
        const = ConstantPE(1.0)
        assert const.is_pure() is True
    
    def test_no_inputs(self):
        const = ConstantPE(1.0)
        assert const.inputs() == []
    
    def test_repr(self):
        const = ConstantPE(0.5, channels=2)
        repr_str = repr(const)
        assert "ConstantPE" in repr_str
        assert "0.5" in repr_str
        assert "2" in repr_str


class TestConstantPERender:
    """Test ConstantPE rendering."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_render_returns_snippet(self):
        const = ConstantPE(0.5)
        self.renderer.set_source(const)
        
        snippet = const.render(0, 100)
        assert snippet.start == 0
        assert snippet.duration == 100
        assert snippet.channels == 1
    
    def test_render_correct_value(self):
        const = ConstantPE(0.75)
        self.renderer.set_source(const)
        
        snippet = const.render(0, 100)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((100, 1), 0.75, dtype=np.float32)
        )
    
    def test_render_stereo(self):
        const = ConstantPE(0.5, channels=2)
        self.renderer.set_source(const)
        
        snippet = const.render(0, 100)
        assert snippet.channels == 2
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((100, 2), 0.5, dtype=np.float32)
        )
    
    def test_render_negative_value(self):
        const = ConstantPE(-0.5)
        self.renderer.set_source(const)
        
        snippet = const.render(0, 100)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((100, 1), -0.5, dtype=np.float32)
        )
    
    def test_render_zero(self):
        const = ConstantPE(0.0)
        self.renderer.set_source(const)
        
        snippet = const.render(0, 100)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((100, 1), dtype=np.float32)
        )
    
    def test_render_negative_start(self):
        const = ConstantPE(0.5)
        self.renderer.set_source(const)
        
        snippet = const.render(-100, 200)
        assert snippet.start == -100
        assert snippet.duration == 200
        # All values should still be 0.5
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((200, 1), 0.5, dtype=np.float32)
        )
    
    def test_render_large_value(self):
        """Test with values outside [-1, 1] range."""
        const = ConstantPE(440.0)  # e.g., for frequency modulation
        self.renderer.set_source(const)
        
        snippet = const.render(0, 100)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((100, 1), 440.0, dtype=np.float32)
        )
