"""
ProcessingElement and SourcePE abstract base classes.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.config import handle_error


class ProcessingElement(ABC):
    """
    Abstract base class for all audio processing elements.
    
    A ProcessingElement generates audio samples on demand via render().
    Processing elements form a directed acyclic graph (DAG), where:
    - Sources (SourcePE subclasses) have no inputs
    - Processors have one or more input ProcessingElements
    
    The render() method always returns a Snippet of the requested size,
    with samples outside the element's extent() zero-filled.
    
    Before rendering, the graph must be configured via configure() which
    is called automatically by Renderer.set_source(). This injects the
    sample rate into all PEs in the graph.
    """
    
    # Sample rate is injected by Renderer.set_source() via configure()
    _sample_rate: Optional[int] = None
    
    # Cached extent (computed lazily on first access)
    _cached_extent: Optional[Extent] = None
    
    @property
    def sample_rate(self) -> int:
        """
        The sample rate in Hz.
        
        Available after the graph is configured via Renderer.set_source().
        
        Raises:
            RuntimeError: If accessed before configuration (always fatal)
        """
        if self._sample_rate is None:
            handle_error(
                f"{self.__class__.__name__}.sample_rate accessed before configuration. "
                f"Call Renderer.set_source() first.",
                fatal=True
            )
        return self._sample_rate
    
    def configure(self, sample_rate: int) -> None:
        """
        Configure this PE and all its inputs with the sample rate.
        
        Called automatically by Renderer.set_source(). Propagates
        configuration through the entire graph.
        
        Args:
            sample_rate: The sample rate in Hz
        """
        self._sample_rate = sample_rate
        for input_pe in self.inputs():
            input_pe.configure(sample_rate)
    
    def render(self, start: int, duration: int) -> Snippet:
        """
        Generate audio samples for the given range.
        
        This method ALWAYS returns a Snippet of exactly `duration` samples
        starting at `start`. Samples outside self.extent() are zero-filled.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate
        
        Returns:
            Snippet containing the requested audio data
        """
        if duration <= 0:
            import numpy as np
            # Determine channel count (concrete value needed for shape)
            channels = self.channel_count()
            if channels is None:
                # If dynamic, try to get from first input or default to 1
                inputs = self.inputs()
                if inputs:
                    channels = inputs[0].channel_count()
                if channels is None:
                    channels = 1
            
            return Snippet.from_zeros(start, 0, channels)
            
        return self._render(start, duration)

    @abstractmethod
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Actual rendering logic, implemented by subclasses.
        
        Called by render() when duration > 0.
        
        Args:
            start: Starting sample index
            duration: Number of samples (> 0)
            
        Returns:
            Snippet containing the audio data
        """
        pass
    
    def extent(self) -> Extent:
        """
        Return the temporal bounds of this processing element.
        
        The extent defines where this element has actual data.
        Requests outside the extent will return zeros.
        
        Computed lazily and cached. Override _compute_extent() to
        customize (not this method).
        
        Returns:
            Extent defining start and end bounds
        """
        if self._cached_extent is None:
            self._cached_extent = self._compute_extent()
        return self._cached_extent
    
    def _compute_extent(self) -> Extent:
        """
        Compute the temporal extent of this PE.
        
        Default: infinite extent (None, None).
        
        Override for:
        - Finite sources (e.g., WavFileReaderPE)
        - PEs that compute extent from inputs (e.g., MixPE -> union)
        
        Returns:
            Extent defining start and end bounds
        """
        return Extent(None, None)
    
    @abstractmethod
    def inputs(self) -> list[ProcessingElement]:
        """
        Return the list of input ProcessingElements.
        
        Returns:
            List of input PEs (empty for sources)
        """
        pass
    
    def is_pure(self) -> bool:
        """
        Returns True if this PE is stateless (pure/idempotent).
        
        A pure PE:
        - Has no mutable internal state that changes between render() calls
        - Always returns the same output for the same (start, duration) inputs
        - Can safely have multiple sinks (outputs connected to multiple PEs)
        
        Non-pure PEs (e.g., ReverbPE with internal delay buffers) must have
        exactly one sink to ensure correct render order.
        
        Default: False (safe default for stateful PEs)
        """
        return False
    
    def channel_count(self) -> Optional[int]:
        """
        Number of output channels this PE produces.
        
        Returns:
            int: Fixed channel count
            None: Same as primary input (pass-through)
        
        Sources (PEs with no inputs) must return int, not None.
        """
        return None  # Default: pass-through
    
    def required_input_channels(self) -> Optional[int]:
        """
        Number of channels required from input(s).
        
        Returns:
            int: Requires exactly this many input channels
            None: Accepts any channel count
        """
        return None  # Default: accept any
    
    def resolve_channel_count(self, input_channel_counts: list[int]) -> int:
        """
        Resolve output channel count when channel_count() returns None.
        
        Called by the graph validator when this PE has multiple inputs
        with potentially different channel counts.
        
        Default: Return the first input's channel count.
        
        Args:
            input_channel_counts: Channel counts of all inputs
        
        Returns:
            The output channel count for this PE
        """
        if input_channel_counts:
            return input_channel_counts[0]
        raise ValueError(
            f"{self.__class__.__name__} has no inputs but channel_count() is None"
        )
    
    def on_start(self) -> None:
        """
        Called once before first render, after configure().
        
        Override to allocate resources, open files, initialize state, etc.
        Called by Renderer.start() in bottom-up order (inputs first).
        
        Default implementation does nothing.
        """
        pass
    
    def on_stop(self) -> None:
        """
        Called once after final render.
        
        Override to release resources, close files, finalize output, etc.
        Called by Renderer.stop() in top-down order (outputs first).
        
        Default implementation does nothing.
        """
        pass
    
    def reset_state(self) -> None:
        """
        Reset this PE's internal state.
        
        Calls _reset_state() if the subclass implements it. Pure PEs typically
        don't implement _reset_state() (no-op). Non-pure PEs can override
        _reset_state() to reset their state (e.g., oscillator phase, filter
        memory, envelope state).
        
        Useful for:
        - Resetting oscillators on gate/trigger events (analog-like behavior)
        - Resetting state when scrubbing/jogging to different positions
        - Re-initializing stateful PEs during rendering
        
        Default implementation does nothing (calls _reset_state() if it exists).
        """
        if hasattr(self, '_reset_state'):
            self._reset_state()


class SourcePE(ProcessingElement):
    """
    Abstract base class for source ProcessingElements (no inputs).
    
    Sources generate audio from external data (files, synthesis, etc.)
    rather than processing input from other PEs.
    
    Sources are typically pure (stateless) and must declare their
    output channel count explicitly.
    """
    
    def inputs(self) -> list[ProcessingElement]:
        """Sources have no inputs."""
        return []
    
    def is_pure(self) -> bool:
        """
        Sources are typically pure (stateless).
        
        Override and return False for sources with internal state
        (e.g., noise generator without fixed seed).
        """
        return True
    
    def required_input_channels(self) -> Optional[int]:
        """Not applicable for sources."""
        return None
    
    @abstractmethod
    def channel_count(self) -> int:
        """
        Sources MUST declare their output channel count.
        
        Returns:
            Number of output channels (must be concrete int, not None)
        """
        pass
