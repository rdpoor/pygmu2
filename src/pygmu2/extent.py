"""
Extent class for defining temporal bounds of ProcessingElements.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations
from enum import Enum
from typing import Optional


class ExtendMode(Enum):
    """Behavior for samples outside a ProcessingElement's extent."""
    ZERO = "zero"           # Output zeros (default)
    HOLD_FIRST = "hold_first"  # Hold first value before extent
    HOLD_LAST = "hold_last"    # Hold last value after extent
    HOLD_BOTH = "hold_both"    # Hold first before, last after


class Extent:
    """
    Defines the start and end sample indices available from a ProcessingElement.
    
    Supports infinite bounds: None means "undefined" (infinite in that direction).
    - start=None: started infinitely in the past
    - end=None: ends indefinitely in the future
    """
    
    def __init__(self, start: Optional[int] = None, end: Optional[int] = None):
        """
        Create an Extent.
        
        Args:
            start: Starting sample index, or None for infinite past
            end: Ending sample index (exclusive), or None for infinite future
        
        Raises:
            ValueError: If start > end (when both are defined)
        """
        # Empty extents (start == end) are allowed and represent "no samples".
        if start is not None and end is not None and start > end:
            raise ValueError(f"start ({start}) must be less than or equal to end ({end})")
        self._start = start
        self._end = end
    
    @property
    def start(self) -> Optional[int]:
        """Starting sample index, or None if undefined (infinite past)."""
        return self._start
    
    @property
    def end(self) -> Optional[int]:
        """Ending sample index (exclusive), or None if undefined (infinite future)."""
        return self._end
    
    @property
    def duration(self) -> Optional[int]:
        """
        Frame count of this extent, or None if either bound is undefined.
        """
        if self._start is None or self._end is None:
            return None
        return self._end - self._start
    
    def is_empty(self) -> bool:
        """Returns True if extent has zero duration (start == end with finite bounds)."""
        return self._start is not None and self._end is not None and self._start == self._end
    
    def contains(self, sample_index: int) -> bool:
        """
        Returns True if the sample index is within this extent.
        
        Args:
            sample_index: The sample index to check
        """
        if self._start is not None and sample_index < self._start:
            return False
        if self._end is not None and sample_index >= self._end:
            return False
        return True
    
    def spans(self, start: int, duration: int) -> bool:
        """
        Returns True if this extent entirely contains the range [start, start+duration).
        
        Args:
            start: Start of the range to check
            duration: Duration of the range to check
        """
        if duration <= 0:
            return True
        end = start + duration
        if self._start is not None and start < self._start:
            return False
        if self._end is not None and end > self._end:
            return False
        return True
    
    def intersects(self, other: Extent) -> bool:
        """
        Returns True if this extent overlaps with another extent.
        
        Args:
            other: The other Extent to check
        """
        # Empty extents never overlap anything (including other empty extents).
        if self.is_empty() or other.is_empty():
            return False

        # Check if self ends before other starts
        if self._end is not None and other._start is not None:
            if self._end <= other._start:
                return False
        # Check if other ends before self starts
        if other._end is not None and self._start is not None:
            if other._end <= self._start:
                return False
        return True
    
    def intersection(self, other: Extent) -> Extent:
        """
        Returns the intersection of this extent with another.
        
        If the extents do not overlap, returns an empty extent (start == end)
        at the intersection boundary.
        
        Args:
            other: The other Extent to intersect with
        """
        # If either is empty, intersection is empty.
        if self.is_empty():
            return Extent(self._start, self._start)
        if other.is_empty():
            return Extent(other._start, other._start)
        
        # Compute intersection start (max of starts, treating None as -infinity)
        if self._start is None:
            new_start = other._start
        elif other._start is None:
            new_start = self._start
        else:
            new_start = max(self._start, other._start)
        
        # Compute intersection end (min of ends, treating None as +infinity)
        if self._end is None:
            new_end = other._end
        elif other._end is None:
            new_end = self._end
        else:
            new_end = min(self._end, other._end)

        # Disjoint extents: represent as an empty extent at the boundary.
        # For half-open intervals, this boundary is max(start1, start2) when finite.
        if new_start is not None and new_end is not None and new_start > new_end:
            return Extent(new_start, new_start)

        return Extent(new_start, new_end)
    
    def union(self, other: Extent) -> Extent:
        """
        Returns the union of this extent with another (smallest extent containing both).
        
        Args:
            other: The other Extent to union with
        """
        # Empty extents add nothing.
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        # Compute union start (min of starts, treating None as -infinity)
        if self._start is None or other._start is None:
            new_start = None
        else:
            new_start = min(self._start, other._start)
        
        # Compute union end (max of ends, treating None as +infinity)
        if self._end is None or other._end is None:
            new_end = None
        else:
            new_end = max(self._end, other._end)
        
        return Extent(new_start, new_end)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Extent):
            return NotImplemented
        return self._start == other._start and self._end == other._end
    
    def __repr__(self) -> str:
        start_str = str(self._start) if self._start is not None else "-∞"
        end_str = str(self._end) if self._end is not None else "+∞"
        return f"Extent({start_str}, {end_str})"

    def __bool__(self) -> bool:
        """
        Treat empty extents as falsy.

        This preserves common idioms like:
            extent = extent.intersection(param_extent) or extent
        where a non-overlapping (empty) intersection should fall back.
        """
        return not self.is_empty()
