"""
NullRenderer - renders as fast as possible with no output.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from pygmu2.renderer import Renderer
from pygmu2.snippet import Snippet


class NullRenderer(Renderer):
    """
    A renderer that discards output, rendering as fast as possible.
    
    Use cases:
    - Benchmarking PE graph performance
    - Testing
    - Driving side-effect PEs (e.g., file writers) without other output
    
    Example:
        >>> renderer = NullRenderer(sample_rate=44100)
        >>> renderer.set_source(my_graph)
        >>> renderer.start()
        >>> renderer.render(0, 44100 * 10)  # Render 10 seconds
        >>> renderer.stop()
    """
    
    def _output(self, snippet: Snippet) -> None:
        """Discard the snippet (no output)."""
        pass
