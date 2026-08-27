"""
MixPE - mixes (adds) multiple PE outputs together.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

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
        *inputs: One or more ProcessingElements to mix together. You may also
            pass a single list/tuple of inputs.  When exactly one input is
            given, render() delegates directly to that input with no copying.

    Raises:
        ValueError: If no inputs are provided

    Example:
        # Mix two sine waves
        sine1_stream = SinePE(frequency=440.0, amplitude=0.5)
        sine2_stream = SinePE(frequency=550.0, amplitude=0.5)
        mixed_stream = MixPE(sine1_stream, sine2_stream)

        # Mix three sources
        mixed_stream = MixPE(source1_stream, source2_stream, source3_stream)

        # Pass-through a single source
        mixed_stream = MixPE(source1_stream)
    """

    def __init__(self, *inputs: ProcessingElement):
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            inputs = tuple(inputs[0])

        if len(inputs) < 1:
            raise ValueError("MixPE requires at least 1 input")

        self._inputs = list(inputs)

    def inputs(self) -> list[ProcessingElement]:
        """Return the list of input PEs."""
        return self._inputs

    def _render(self, start: int, duration: int) -> Snippet:
        """
        Mix all inputs by adding their samples together.

        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)

        Returns:
            Snippet containing the sum of all input samples
        """
        # Singleton fast-path: delegate directly, no allocation or copying.
        if len(self._inputs) == 1:
            return self._inputs[0].render(start, duration)

        # Render only inputs whose extents intersect the requested range
        req_extent = Extent(start, start + duration)
        rendered = []
        for inp in self._inputs:
            if inp.extent().intersects(req_extent):
                rendered.append((inp, inp.render(start, duration)))

        if not rendered:
            channels = self.channel_count() or 1
            return Snippet.from_zeros(start, duration, channels)

        # Sum all data arrays
        result = rendered[0][1].data.copy()
        for _, snippet in rendered[1:]:
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

    def channel_count(self) -> int | None:
        """
        Return the channel count (same as inputs).

        Queries the first input's channel count. Validation ensures
        all inputs have compatible channel counts.
        """
        if self._inputs:
            return self._inputs[0].channel_count()
        return None

    def __repr__(self) -> str:
        input_names = [inp.__class__.__name__ for inp in self._inputs]
        return f"MixPE({', '.join(input_names)})"
