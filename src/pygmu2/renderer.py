"""
Renderer abstract base class for audio output.

A Renderer consumes audio from a ProcessingElement graph and outputs it
to a destination (DAC, file, etc.). Per DESIGN_PHILOSOPHY.md PD-2 there
is no up-front graph validation: set_source() assigns, start()/stop()
walk the graph for lifecycle, and anything wrong surfaces as an error
at render time. For per-PE profiling use pygmu2.diagnostics.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from pygmu2.config import get_sample_rate
from pygmu2.snippet import Snippet
from pygmu2.processing_element import ProcessingElement
from pygmu2.logger import get_logger

logger = get_logger(__name__)


class Renderer(ABC):
    """
    Abstract base class for rendering audio output.

    Lifecycle:
        1. set_source() - Attach the root ProcessingElement
        2. start() - Call on_start() on all PEs (bottom-up)
        3. render() - Process audio (can be called multiple times)
        4. stop() - Call on_stop() on all PEs (top-down)

    Subclasses implement _output() for specific output formats.
    """

    def __init__(self, sample_rate: int | None = None):
        """
        Initialize the Renderer.

        If sample_rate is None, reads from the global set_sample_rate().

        Raises:
            RuntimeError: If no sample_rate is provided and global is not set.
        """
        if sample_rate is None:
            sample_rate = get_sample_rate()
        if sample_rate is None:
            raise RuntimeError(
                "Sample rate not set. Call pygmu2.set_sample_rate() or pass sample_rate."
            )
        self._sample_rate = sample_rate
        self._source: ProcessingElement | None = None
        self._started: bool = False

    @property
    def sample_rate(self) -> int:
        """The sample rate in Hz."""
        return self._sample_rate

    @property
    def source(self) -> ProcessingElement | None:
        """The source ProcessingElement, or None if not set."""
        return self._source

    @property
    def started(self) -> bool:
        """True if the renderer has been started."""
        return self._started

    def set_source(self, source: ProcessingElement) -> None:
        """
        Set the source ProcessingElement.

        Raises:
            RuntimeError: If called while started.
        """
        if self._started:
            raise RuntimeError("Cannot set source while started. Call stop() first.")
        self._source = source
        logger.info(
            f"Source set: {source.__class__.__name__}, sample_rate={self._sample_rate}"
        )

    def start(self) -> None:
        """
        Start the renderer. Calls on_start() on all PEs in the graph,
        bottom-up (inputs before outputs).

        Raises:
            RuntimeError: If no source set or already started.
        """
        if self._source is None:
            raise RuntimeError("No source set. Call set_source() first.")
        if self._started:
            raise RuntimeError("Already started. Call stop() first.")

        self._start_graph(self._source)
        self._started = True
        logger.info("Renderer started")

    def stop(self) -> None:
        """
        Stop the renderer. Calls on_stop() on all PEs in the graph,
        top-down (outputs before inputs). Idempotent.
        """
        if not self._started:
            return
        if self._source is not None:
            self._stop_graph(self._source)
        self._started = False
        logger.info("Renderer stopped")

    def render(self, start: int, duration: int) -> None:
        """
        Request a Snippet from the source and output it.

        Raises:
            RuntimeError: If no source set or not started.
            ValueError: If duration < 1.
        """
        if self._source is None:
            raise RuntimeError("No source set. Call set_source() first.")
        if not self._started:
            raise RuntimeError("Not started. Call start() first.")
        if duration < 1:
            raise ValueError(
                "Renderer.render() requires duration >= 1 to prevent infinite loops."
            )
        self._output(self._source.render(start, duration))

    def __enter__(self) -> "Renderer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit - ensures stop() is called (never swallows)."""
        self.stop()

    @abstractmethod
    def _output(self, snippet: Snippet) -> None:
        """
        Output the snippet to the destination (DAC, file, etc.).
        Implemented by subclasses.
        """

    def _start_graph(
        self,
        pe: ProcessingElement,
        started: set[int] | None = None,
    ) -> None:
        """Call on_start() on each reachable PE exactly once, inputs first."""
        if started is None:
            started = set()
        pe_id = id(pe)
        if pe_id in started:
            return
        started.add(pe_id)
        for input_pe in pe.inputs():
            self._start_graph(input_pe, started)
        pe.on_start()

    def _stop_graph(
        self,
        pe: ProcessingElement,
        stopped: set[int] | None = None,
    ) -> None:
        """Call on_stop() on each reachable PE exactly once, outputs first."""
        if stopped is None:
            stopped = set()
        pe_id = id(pe)
        if pe_id in stopped:
            return
        stopped.add(pe_id)
        pe.on_stop()
        for input_pe in pe.inputs():
            self._stop_graph(input_pe, stopped)
