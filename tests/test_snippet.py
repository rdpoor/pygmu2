"""
Tests for Snippet class.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import Snippet


class TestSnippetBasics:
    """Test basic Snippet functionality."""
    
    def test_create_mono_snippet(self):
        """Test creating a mono snippet from 2D array."""
        data = np.array([[0.1], [0.2], [0.3]])
        snip = Snippet(100, data)
        assert snip.start == 100
        assert snip.end == 103
        assert snip.duration == 3
        assert snip.channels == 1
    
    def test_create_stereo_snippet(self):
        """Test creating a stereo snippet."""
        data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        snip = Snippet(0, data)
        assert snip.duration == 3
        assert snip.channels == 2
    
    def test_create_from_1d_array(self):
        """Test that 1D array is converted to (samples, 1)."""
        data = np.array([0.1, 0.2, 0.3])
        snip = Snippet(0, data)
        assert snip.channels == 1
        assert snip.data.shape == (3, 1)
    
    def test_data_access(self):
        """Test accessing the underlying data."""
        data = np.array([[0.1, 0.2], [0.3, 0.4]])
        snip = Snippet(0, data)
        assert np.allclose(snip.data, data)
    
    def test_invalid_3d_array_raises(self):
        """Test that 3D array raises ValueError."""
        data = np.zeros((2, 2, 2))
        with pytest.raises(ValueError):
            Snippet(0, data)
    
    def test_empty_array_raises(self):
        """Test that empty array raises ValueError."""
        data = np.array([]).reshape(0, 1)
        with pytest.raises(ValueError):
            Snippet(0, data)


class TestSnippetFromZeros:
    """Test Snippet.from_zeros() factory method."""
    
    def test_from_zeros_mono(self):
        """Test creating mono silence."""
        snip = Snippet.from_zeros(start=100, duration=1000, channels=1)
        assert snip.start == 100
        assert snip.duration == 1000
        assert snip.channels == 1
        assert np.all(snip.data == 0)
    
    def test_from_zeros_stereo(self):
        """Test creating stereo silence."""
        snip = Snippet.from_zeros(start=0, duration=500, channels=2)
        assert snip.channels == 2
        assert snip.data.shape == (500, 2)
        assert np.all(snip.data == 0)
    
    def test_from_zeros_default_channels(self):
        """Test that default channels is 1."""
        snip = Snippet.from_zeros(start=0, duration=100)
        assert snip.channels == 1


class TestSnippetEquality:
    """Test Snippet equality comparison."""
    
    def test_equal_snippets(self):
        """Test that identical snippets are equal."""
        data = np.array([[0.1], [0.2]])
        snip1 = Snippet(100, data.copy())
        snip2 = Snippet(100, data.copy())
        assert snip1 == snip2
    
    def test_different_start_not_equal(self):
        """Test that snippets with different start are not equal."""
        data = np.array([[0.1], [0.2]])
        snip1 = Snippet(100, data.copy())
        snip2 = Snippet(200, data.copy())
        assert snip1 != snip2
    
    def test_different_data_not_equal(self):
        """Test that snippets with different data are not equal."""
        snip1 = Snippet(0, np.array([[0.1], [0.2]]))
        snip2 = Snippet(0, np.array([[0.3], [0.4]]))
        assert snip1 != snip2
    
    def test_repr(self):
        """Test string representation."""
        snip = Snippet(100, np.zeros((50, 2)))
        r = repr(snip)
        assert "100" in r
        assert "50" in r
        assert "2" in r
