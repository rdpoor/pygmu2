"""
MixPE - mixes (adds) multiple PE outputs together.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np
from typing import Optional

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.logger import get_logger

logger = get_logger(__name__)


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
        sine1_stream = SinePE(frequency=440.0, amplitude=0.5)
        sine2_stream = SinePE(frequency=550.0, amplitude=0.5)
        mixed_stream = MixPE(sine1_stream, sine2_stream)
        
        # Mix three sources
        mixed_stream = MixPE(source1_stream, source2_stream, source3_stream)
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
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Mix all inputs by adding their samples together.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet containing the sum of all input samples
        """
        # Render all inputs
        snippets = [inp.render(start, duration) for inp in self._inputs]
        
        # Debug: log what each input is contributing
        if len(self._inputs) > 1 and duration > 0:
            for i, snippet in enumerate(snippets):
                first_val = snippet.data[0, 0] if snippet.data.shape[0] > 0 else 0
                mid_idx = duration // 2
                mid_val = snippet.data[mid_idx, 0] if mid_idx < snippet.data.shape[0] else 0
                last_val = snippet.data[-1, 0] if snippet.data.shape[0] > 0 else 0
                logger.debug(
                    f"MixPE: Input {i} ({self._inputs[i].__class__.__name__}) at start={start}: "
                    f"first={first_val:.2f}, mid={mid_val:.2f}, last={last_val:.2f}"
                )
        
        # Sum all data arrays
        result = snippets[0].data.copy()
        for snippet in snippets[1:]:
            result += snippet.data
        
        # Debug: log the result
        if len(self._inputs) > 1 and duration > 0:
            first_result = result[0, 0] if result.shape[0] > 0 else 0
            mid_idx = duration // 2
            mid_result = result[mid_idx, 0] if mid_idx < result.shape[0] else 0
            last_result = result[-1, 0] if result.shape[0] > 0 else 0
            logger.debug(
                f"MixPE: Result at start={start}: "
                f"first={first_result:.2f}, mid={mid_result:.2f}, last={last_result:.2f}"
            )
        
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
        
        Queries the first input's channel count. Validation ensures
        all inputs have compatible channel counts.
        """
        if self._inputs:
            return self._inputs[0].channel_count()
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
