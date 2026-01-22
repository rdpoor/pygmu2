"""
MixPE - mixes (adds) multiple PE outputs together.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
from typing import Optional

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class MixPE(ProcessingElement):
    """
    A ProcessingElement that mixes (adds) multiple PE outputs together.
    
    All inputs must be ProcessingElements. For adding constants to signals,
    use AddPE (to be implemented) or ConstantPE as an input.
    
    Channel handling:
    - All inputs must have the same channel count
    - Output channel count matches input channel count
    
    Extent:
    - The union of all input extents (covers the full range of all inputs)
    
    Args:
        *inputs: Two or more ProcessingElements to mix together
    
    Raises:
        ValueError: If fewer than 2 inputs provided
    
    Example:
        # Mix two sine waves
        sine1 = SinePE(frequency=440.0, amplitude=0.5)
        sine2 = SinePE(frequency=550.0, amplitude=0.5)
        mixed = MixPE(sine1, sine2)
        
        # Mix three sources
        mixed = MixPE(source1, source2, source3)
    """
    
    def __init__(self, *inputs: ProcessingElement):
        if len(inputs) < 2:
            raise ValueError("MixPE requires at least 2 inputs")
        
        self._inputs = list(inputs)
    
    def inputs(self) -> list[ProcessingElement]:
        """Return the list of input PEs."""
        return self._inputs
    
    def is_pure(self) -> bool:
        """
        MixPE is pure - it performs a stateless operation (addition).
        
        Note: Graph validation will still check that non-pure inputs
        aren't used in multiple places.
        """
        return True
    
    def render(self, start: int, duration: int) -> Snippet:
        """
        Mix all inputs by adding their samples together.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate
        
        Returns:
            Snippet containing the sum of all input samples
        """
        # Render all inputs
        snippets = [inp.render(start, duration) for inp in self._inputs]
        
        # Sum all data arrays
        result = snippets[0].data.copy()
        for snippet in snippets[1:]:
            result += snippet.data
        
        return Snippet(start, result)
    
    def _compute_extent(self) -> Extent:
        """
        Compute the union of all input extents.
        
        The mix produces output wherever any input has output.
        """
        result = self._inputs[0].extent()
        for inp in self._inputs[1:]:
            result = result.union(inp.extent())
        return result
    
    def channel_count(self) -> Optional[int]:
        """
        Return the channel count (same as inputs).
        
        Returns None to indicate it passes through input channels.
        Validation ensures all inputs have compatible channel counts.
        """
        return None
    
    def required_input_channels(self) -> Optional[int]:
        """
        All inputs must have the same channel count.
        
        Returns None - validation is done by checking that all inputs
        resolve to the same channel count.
        """
        return None
    
    def resolve_channel_count(self, input_channel_counts: list[int]) -> int:
        """
        Resolve output channel count from inputs.
        
        All inputs must have the same channel count.
        
        Raises:
            ValueError: If inputs have different channel counts
        """
        if not input_channel_counts:
            raise ValueError("MixPE has no inputs")
        
        first = input_channel_counts[0]
        for i, count in enumerate(input_channel_counts[1:], start=2):
            if count != first:
                raise ValueError(
                    f"MixPE input channel mismatch: input 1 has {first} channels, "
                    f"input {i} has {count} channels"
                )
        
        return first
    
    def __repr__(self) -> str:
        input_names = [inp.__class__.__name__ for inp in self._inputs]
        return f"MixPE({', '.join(input_names)})"
