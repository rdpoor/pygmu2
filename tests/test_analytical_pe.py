"""
Tests for analytical PEs: IdentityPE and DiracPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    IdentityPE,
    DiracPE,
    DelayPE,
    NullRenderer,
    Extent,
)


class TestIdentityPEBasics:
    """Test basic IdentityPE creation and properties."""
    
    def test_create_identity_pe(self):
        identity = IdentityPE()
        assert identity.channel_count() == 1
    
    def test_create_with_channels(self):
        identity = IdentityPE(channels=2)
        assert identity.channel_count() == 2
    
    def test_is_pure(self):
        identity = IdentityPE()
        assert identity.is_pure() is True
    
    def test_no_inputs(self):
        identity = IdentityPE()
        assert identity.inputs() == []
    
    def test_extent_infinite(self):
        identity = IdentityPE()
        extent = identity.extent()
        assert extent.start is None
        assert extent.end is None
    
    def test_repr(self):
        identity = IdentityPE()
        repr_str = repr(identity)
        assert "IdentityPE" in repr_str


class TestIdentityPERender:
    """Test IdentityPE rendering."""
    
    def test_render_from_zero(self):
        identity = IdentityPE()
        snippet = identity.render(0, 5)
        
        expected = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
        assert snippet.start == 0
        assert snippet.duration == 5
    
    def test_render_from_positive(self):
        identity = IdentityPE()
        snippet = identity.render(100, 5)
        
        expected = np.array([[100.0], [101.0], [102.0], [103.0], [104.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
    
    def test_render_from_negative(self):
        identity = IdentityPE()
        snippet = identity.render(-3, 6)
        
        expected = np.array([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
    
    def test_render_stereo(self):
        identity = IdentityPE(channels=2)
        snippet = identity.render(0, 3)
        
        expected = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
        assert snippet.channels == 2
    
    def test_render_large_indices(self):
        identity = IdentityPE()
        snippet = identity.render(1_000_000, 3)
        
        expected = np.array([[1_000_000.0], [1_000_001.0], [1_000_002.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)


class TestIdentityPEWithDelay:
    """Test IdentityPE combined with DelayPE."""
    
    def test_identity_with_delay(self):
        identity = IdentityPE()
        delayed = DelayPE(identity, delay=100)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delayed)
        renderer.start()
        
        # At output position 100, we get source position 0
        snippet = delayed.render(100, 5)
        expected = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
        
        renderer.stop()
    
    def test_identity_verifies_delay_amount(self):
        """IdentityPE is useful for verifying delay values."""
        identity = IdentityPE()
        delayed = DelayPE(identity, delay=42)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delayed)
        renderer.start()
        
        # At output position 42, we get source position 0 (value 0)
        snippet = delayed.render(42, 1)
        assert snippet.data[0, 0] == 0.0
        
        # At output position 0, we get source position -42 (value -42)
        snippet = delayed.render(0, 1)
        assert snippet.data[0, 0] == -42.0
        
        renderer.stop()


class TestDiracPEBasics:
    """Test basic DiracPE creation and properties."""
    
    def test_create_dirac_pe(self):
        dirac = DiracPE()
        assert dirac.channel_count() == 1
    
    def test_create_with_channels(self):
        dirac = DiracPE(channels=2)
        assert dirac.channel_count() == 2
    
    def test_is_pure(self):
        dirac = DiracPE()
        assert dirac.is_pure() is True
    
    def test_no_inputs(self):
        dirac = DiracPE()
        assert dirac.inputs() == []
    
    def test_extent_infinite(self):
        dirac = DiracPE()
        extent = dirac.extent()
        assert extent.start is None
        assert extent.end is None
    
    def test_repr(self):
        dirac = DiracPE()
        repr_str = repr(dirac)
        assert "DiracPE" in repr_str


class TestDiracPERender:
    """Test DiracPE rendering."""
    
    def test_render_including_zero(self):
        dirac = DiracPE()
        snippet = dirac.render(-2, 5)
        
        # Indices: -2, -1, 0, 1, 2
        expected = np.array([[0.0], [0.0], [1.0], [0.0], [0.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
    
    def test_render_from_zero(self):
        dirac = DiracPE()
        snippet = dirac.render(0, 5)
        
        expected = np.array([[1.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
    
    def test_render_after_zero(self):
        dirac = DiracPE()
        snippet = dirac.render(1, 5)
        
        expected = np.zeros((5, 1), dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
    
    def test_render_before_zero(self):
        dirac = DiracPE()
        snippet = dirac.render(-10, 5)
        
        expected = np.zeros((5, 1), dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
    
    def test_render_ending_at_zero(self):
        dirac = DiracPE()
        snippet = dirac.render(-4, 5)
        
        # Indices: -4, -3, -2, -1, 0
        expected = np.array([[0.0], [0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
    
    def test_render_stereo(self):
        dirac = DiracPE(channels=2)
        snippet = dirac.render(0, 3)
        
        expected = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
    
    def test_render_single_sample_at_zero(self):
        dirac = DiracPE()
        snippet = dirac.render(0, 1)
        
        expected = np.array([[1.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
    
    def test_render_single_sample_not_at_zero(self):
        dirac = DiracPE()
        snippet = dirac.render(5, 1)
        
        expected = np.array([[0.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)


class TestDiracPEWithDelay:
    """Test DiracPE combined with DelayPE."""
    
    def test_delayed_impulse(self):
        dirac = DiracPE()
        delayed = DelayPE(dirac, delay=100)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delayed)
        renderer.start()
        
        # Impulse should now be at sample 100
        snippet = delayed.render(98, 5)
        # Indices: 98, 99, 100, 101, 102
        expected = np.array([[0.0], [0.0], [1.0], [0.0], [0.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
        
        renderer.stop()
    
    def test_delayed_impulse_not_at_original(self):
        dirac = DiracPE()
        delayed = DelayPE(dirac, delay=100)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delayed)
        renderer.start()
        
        # Original position 0 should now be zero
        snippet = delayed.render(-2, 5)
        expected = np.zeros((5, 1), dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
        
        renderer.stop()


class TestAnalyticalPEIntegration:
    """Integration tests for analytical PEs."""
    
    def test_identity_with_renderer(self):
        identity = IdentityPE()
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(identity)
        
        with renderer:
            renderer.start()
            snippet = identity.render(0, 10)
            assert snippet.data[0, 0] == 0.0
            assert snippet.data[9, 0] == 9.0
    
    def test_dirac_with_renderer(self):
        dirac = DiracPE()
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(dirac)
        
        with renderer:
            renderer.start()
            # Sum of all samples should be 1.0 (single impulse)
            snippet = dirac.render(-100, 200)
            assert np.sum(snippet.data) == 1.0
