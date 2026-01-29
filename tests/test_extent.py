"""
Tests for Extent class.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
from pygmu2 import Extent


class TestExtentBasics:
    """Test basic Extent functionality."""
    
    def test_create_finite_extent(self):
        """Test creating a finite extent."""
        ext = Extent(100, 200)
        assert ext.start == 100
        assert ext.end == 200
        assert ext.duration == 100
    
    def test_create_infinite_start(self):
        """Test creating extent with infinite past."""
        ext = Extent(None, 200)
        assert ext.start is None
        assert ext.end == 200
        assert ext.duration is None
    
    def test_create_infinite_end(self):
        """Test creating extent with infinite future."""
        ext = Extent(100, None)
        assert ext.start == 100
        assert ext.end is None
        assert ext.duration is None
    
    def test_create_fully_infinite(self):
        """Test creating fully infinite extent."""
        ext = Extent(None, None)
        assert ext.start is None
        assert ext.end is None
        assert ext.duration is None
    
    def test_invalid_extent_raises(self):
        """Test that start > end raises ValueError (empty extents are allowed)."""
        with pytest.raises(ValueError):
            Extent(200, 100)
        # Empty extents (start == end) are allowed
        assert Extent(100, 100).is_empty() is True
    
    def test_repr(self):
        """Test string representation."""
        assert "100" in repr(Extent(100, 200))
        assert "-∞" in repr(Extent(None, 200))
        assert "+∞" in repr(Extent(100, None))


class TestExtentContains:
    """Test Extent.contains() method."""
    
    def test_contains_within(self):
        """Test contains for index within bounds."""
        ext = Extent(100, 200)
        assert ext.contains(100) is True
        assert ext.contains(150) is True
        assert ext.contains(199) is True
    
    def test_contains_outside(self):
        """Test contains for index outside bounds."""
        ext = Extent(100, 200)
        assert ext.contains(99) is False
        assert ext.contains(200) is False
        assert ext.contains(300) is False
    
    def test_contains_infinite_start(self):
        """Test contains with infinite start."""
        ext = Extent(None, 200)
        assert ext.contains(-1000000) is True
        assert ext.contains(199) is True
        assert ext.contains(200) is False
    
    def test_contains_infinite_end(self):
        """Test contains with infinite end."""
        ext = Extent(100, None)
        assert ext.contains(99) is False
        assert ext.contains(100) is True
        assert ext.contains(1000000) is True


class TestExtentSpans:
    """Test Extent.spans() method."""
    
    def test_spans_fully_contained(self):
        """Test spans when range is fully contained."""
        ext = Extent(100, 200)
        assert ext.spans(100, 100) is True
        assert ext.spans(110, 50) is True
        assert ext.spans(150, 50) is True
    
    def test_spans_exceeds_start(self):
        """Test spans when range starts before extent."""
        ext = Extent(100, 200)
        assert ext.spans(90, 50) is False
    
    def test_spans_exceeds_end(self):
        """Test spans when range ends after extent."""
        ext = Extent(100, 200)
        assert ext.spans(150, 100) is False
    
    def test_spans_zero_duration(self):
        """Test spans with zero duration always returns True."""
        ext = Extent(100, 200)
        assert ext.spans(50, 0) is True
        assert ext.spans(250, 0) is True


class TestExtentIntersects:
    """Test Extent.intersects() method."""
    
    def test_intersects_overlapping(self):
        """Test intersects with overlapping extents."""
        ext1 = Extent(100, 200)
        ext2 = Extent(150, 250)
        assert ext1.intersects(ext2) is True
        assert ext2.intersects(ext1) is True
    
    def test_intersects_non_overlapping(self):
        """Test intersects with non-overlapping extents."""
        ext1 = Extent(100, 200)
        ext2 = Extent(200, 300)
        assert ext1.intersects(ext2) is False
        assert ext2.intersects(ext1) is False

    def test_intersects_with_empty(self):
        """Empty extents never intersect anything."""
        empty = Extent(10, 10)
        ext = Extent(0, 20)
        assert empty.intersects(ext) is False
        assert ext.intersects(empty) is False
    
    def test_intersects_contained(self):
        """Test intersects when one contains the other."""
        ext1 = Extent(100, 300)
        ext2 = Extent(150, 200)
        assert ext1.intersects(ext2) is True
        assert ext2.intersects(ext1) is True
    
    def test_intersects_infinite(self):
        """Test intersects with infinite extents."""
        ext_finite = Extent(100, 200)
        ext_inf_start = Extent(None, 150)
        ext_inf_end = Extent(150, None)
        
        assert ext_finite.intersects(ext_inf_start) is True
        assert ext_finite.intersects(ext_inf_end) is True


class TestExtentIntersection:
    """Test Extent.intersection() method."""
    
    def test_intersection_overlapping(self):
        """Test intersection of overlapping extents."""
        ext1 = Extent(100, 200)
        ext2 = Extent(150, 250)
        result = ext1.intersection(ext2)
        assert result == Extent(150, 200)
    
    def test_intersection_non_overlapping(self):
        """Test intersection of non-overlapping extents."""
        ext1 = Extent(100, 200)
        ext2 = Extent(200, 300)
        result = ext1.intersection(ext2)
        assert result.is_empty() is True
        assert result == Extent(200, 200)
    
    def test_intersection_with_infinite(self):
        """Test intersection with infinite extent."""
        ext_finite = Extent(100, 200)
        ext_inf = Extent(None, None)
        result = ext_finite.intersection(ext_inf)
        assert result == Extent(100, 200)


class TestExtentUnion:
    """Test Extent.union() method."""
    
    def test_union_overlapping(self):
        """Test union of overlapping extents."""
        ext1 = Extent(100, 200)
        ext2 = Extent(150, 250)
        result = ext1.union(ext2)
        assert result == Extent(100, 250)
    
    def test_union_non_overlapping(self):
        """Test union of non-overlapping extents."""
        ext1 = Extent(100, 200)
        ext2 = Extent(300, 400)
        result = ext1.union(ext2)
        assert result == Extent(100, 400)

    def test_union_with_empty(self):
        """Empty extents add nothing to a union."""
        empty = Extent(10, 10)
        ext = Extent(100, 200)
        assert ext.union(empty) == ext
        assert empty.union(ext) == ext
    
    def test_union_with_infinite_start(self):
        """Test union where one has infinite start."""
        ext1 = Extent(100, 200)
        ext2 = Extent(None, 150)
        result = ext1.union(ext2)
        assert result.start is None
        assert result.end == 200
    
    def test_union_with_infinite_end(self):
        """Test union where one has infinite end."""
        ext1 = Extent(100, 200)
        ext2 = Extent(150, None)
        result = ext1.union(ext2)
        assert result.start == 100
        assert result.end is None
