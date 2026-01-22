"""
Renderer abstract base class for audio output.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from pygmu2.config import handle_error
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.processing_element import ProcessingElement, SourcePE
from pygmu2.logger import get_logger

logger = get_logger(__name__)


class Renderer(ABC):
    """
    Abstract base class for rendering audio output.
    
    A Renderer consumes audio from a ProcessingElement and outputs it
    to a destination (DAC, file, etc.). The Renderer is the authority
    on sample rate and validates the processing graph before rendering.
    
    Lifecycle:
        1. set_source() - Configure and validate the graph
        2. start() - Call on_start() on all PEs, allocate resources
        3. render() - Process audio (can be called multiple times)
        4. stop() - Call on_stop() on all PEs, release resources
    
    Subclasses implement _output() for specific output formats.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the Renderer.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 44100)
        """
        self._sample_rate = sample_rate
        self._source: Optional[ProcessingElement] = None
        self._channel_count: Optional[int] = None
        self._started: bool = False
    
    @property
    def sample_rate(self) -> int:
        """The sample rate in Hz."""
        return self._sample_rate
    
    @property
    def source(self) -> Optional[ProcessingElement]:
        """The source ProcessingElement, or None if not set."""
        return self._source
    
    @property
    def channel_count(self) -> Optional[int]:
        """
        The output channel count of the source graph.
        
        Available after set_source() validates the graph.
        """
        return self._channel_count
    
    @property
    def started(self) -> bool:
        """True if the renderer has been started."""
        return self._started
    
    def set_source(self, source: ProcessingElement) -> None:
        """
        Set the source ProcessingElement, configure, and validate the graph.
        
        This method:
        1. Configures all PEs in the graph with the sample rate
        2. Validates the graph (purity, channel compatibility)
        
        Does NOT call on_start() - call start() explicitly.
        
        Args:
            source: The root ProcessingElement to render from
        
        Raises:
            RuntimeError: If called while started (in STRICT mode)
            ValueError: If graph validation fails (non-pure multi-sink,
                        channel mismatch, etc.)
        """
        if self._started:
            if handle_error("Cannot set source while started. Call stop() first."):
                return  # Lenient mode: warn and return
        
        # Configure all PEs with sample rate
        source.configure(self._sample_rate)
        
        # Validate the graph
        self._channel_count = self._validate_graph(source)
        self._source = source
        logger.info(
            f"Source set: {source.__class__.__name__}, "
            f"sample_rate={self._sample_rate}, "
            f"channel_count={self._channel_count}"
        )
    
    def start(self) -> None:
        """
        Start the renderer. Calls on_start() on all PEs in the graph.
        
        Must call set_source() first. PEs are started bottom-up
        (inputs before outputs).
        
        Raises:
            RuntimeError: If no source set (always fatal) or already started
                          (in STRICT mode)
        """
        if self._source is None:
            handle_error("No source set. Call set_source() first.", fatal=True)
            return  # Never reached, but satisfies type checker
        if self._started:
            if handle_error("Already started. Call stop() first."):
                return  # Lenient mode: warn and return
        
        self._start_graph(self._source)
        self._started = True
        logger.info("Renderer started")
    
    def stop(self) -> None:
        """
        Stop the renderer. Calls on_stop() on all PEs in the graph.
        
        PEs are stopped top-down (outputs before inputs).
        Safe to call multiple times (idempotent).
        """
        if not self._started:
            return  # Idempotent
        
        if self._source is not None:
            self._stop_graph(self._source)
        self._started = False
        logger.info("Renderer stopped")
    
    def render(self, start: int, duration: int) -> None:
        """
        Request a Snippet from the source and output it.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render
        
        Raises:
            RuntimeError: If no source set or not started (always fatal)
        """
        if self._source is None:
            handle_error("No source set. Call set_source() first.", fatal=True)
            return
        if not self._started:
            handle_error("Not started. Call start() first.", fatal=True)
            return
        
        snippet = self._source.render(start, duration)
        self._output(snippet)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures stop() is called."""
        self.stop()
        return False
    
    @abstractmethod
    def _output(self, snippet: Snippet) -> None:
        """
        Output the snippet to the destination.
        
        Subclasses implement this for specific output formats
        (DAC playback, file writing, etc.).
        
        Args:
            snippet: The audio data to output
        """
        pass
    
    def _validate_graph(
        self,
        pe: ProcessingElement,
        seen: Optional[dict[int, int]] = None,
    ) -> int:
        """
        Recursively validate the processing graph.
        
        Checks:
        - Non-pure PEs have only one sink (not reused)
        - Channel counts are compatible
        
        Args:
            pe: The ProcessingElement to validate
            seen: Dictionary mapping PE id to output channel count (for cycle/reuse detection)
        
        Returns:
            The output channel count of this PE
        
        Raises:
            ValueError: If validation fails
        """
        if seen is None:
            seen = {}
        
        pe_id = id(pe)
        
        # Check for PE reuse (multi-sink)
        if pe_id in seen:
            if not pe.is_pure():
                raise ValueError(
                    f"{pe.__class__.__name__} is not pure but has multiple sinks. "
                    f"Stateful PEs can only connect to one downstream PE."
                )
            logger.debug(f"Reusing pure PE: {pe.__class__.__name__}")
            return seen[pe_id]  # Return cached channel count
        
        # Recursively validate inputs first
        input_channel_counts: list[int] = []
        for input_pe in pe.inputs():
            channels = self._validate_graph(input_pe, seen)
            input_channel_counts.append(channels)
        
        # Validate this PE accepts its input channel counts
        required = pe.required_input_channels()
        if required is not None:
            for i, actual in enumerate(input_channel_counts):
                if actual != required:
                    input_pe = pe.inputs()[i]
                    raise ValueError(
                        f"{pe.__class__.__name__} requires {required} channel(s), "
                        f"but {input_pe.__class__.__name__} outputs {actual}"
                    )
        
        # Compute output channels
        output = pe.channel_count()
        if output is None:
            if not input_channel_counts:
                raise ValueError(
                    f"{pe.__class__.__name__} has no inputs but "
                    f"channel_count() is None"
                )
            if hasattr(pe, 'resolve_channel_count'):
                output = pe.resolve_channel_count(input_channel_counts)
            else:
                output = input_channel_counts[0]  # Default: match first input
        
        # Cache and return
        seen[pe_id] = output
        logger.debug(
            f"Validated {pe.__class__.__name__}: "
            f"inputs={input_channel_counts}, output={output}"
        )
        return output
    
    def _start_graph(
        self,
        pe: ProcessingElement,
        started: Optional[set[int]] = None,
    ) -> None:
        """
        Recursively start all PEs in the graph (bottom-up).
        
        Calls on_start() on each PE exactly once, inputs first.
        
        Args:
            pe: The ProcessingElement to start
            started: Set of PE ids already started (for diamond graphs)
        """
        if started is None:
            started = set()
        
        pe_id = id(pe)
        if pe_id in started:
            return  # Already started
        started.add(pe_id)
        
        # Start inputs first (bottom-up)
        for input_pe in pe.inputs():
            self._start_graph(input_pe, started)
        
        pe.on_start()
        logger.debug(f"Started {pe.__class__.__name__}")
    
    def _stop_graph(
        self,
        pe: ProcessingElement,
        stopped: Optional[set[int]] = None,
    ) -> None:
        """
        Recursively stop all PEs in the graph (top-down).
        
        Calls on_stop() on each PE exactly once, outputs first.
        
        Args:
            pe: The ProcessingElement to stop
            stopped: Set of PE ids already stopped (for diamond graphs)
        """
        if stopped is None:
            stopped = set()
        
        pe_id = id(pe)
        if pe_id in stopped:
            return  # Already stopped
        stopped.add(pe_id)
        
        pe.on_stop()
        logger.debug(f"Stopped {pe.__class__.__name__}")
        
        # Stop inputs after (top-down)
        for input_pe in pe.inputs():
            self._stop_graph(input_pe, stopped)