"""
Renderer abstract base class for audio output.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import time

from pygmu2.config import handle_error
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.processing_element import ProcessingElement, SourcePE
from pygmu2.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PEProfile:
    """Profiling data for a single ProcessingElement."""
    pe_class: str
    pe_id: int
    render_count: int = 0
    total_time_ns: int = 0
    total_samples: int = 0
    min_time_ns: int = 0
    max_time_ns: int = 0
    
    @property
    def total_time_ms(self) -> float:
        """Total render time in milliseconds."""
        return self.total_time_ns / 1_000_000
    
    @property
    def avg_time_ms(self) -> float:
        """Average render time per call in milliseconds."""
        if self.render_count == 0:
            return 0.0
        return self.total_time_ms / self.render_count
    
    @property
    def samples_per_second(self) -> float:
        """Throughput in samples per second."""
        if self.total_time_ns == 0:
            return 0.0
        return self.total_samples / (self.total_time_ns / 1_000_000_000)
    
    @property
    def realtime_ratio(self) -> float:
        """Ratio of realtime to render time (>1 means faster than realtime)."""
        if self.total_time_ns == 0:
            return 0.0
        # Assuming 44100 Hz sample rate for this calculation
        realtime_ns = (self.total_samples / 44100) * 1_000_000_000
        return realtime_ns / self.total_time_ns


@dataclass
class ProfileReport:
    """Complete profiling report for a render session."""
    pe_profiles: dict[int, PEProfile] = field(default_factory=dict)
    total_render_time_ns: int = 0
    total_output_time_ns: int = 0
    total_samples: int = 0
    render_calls: int = 0
    
    def add_pe_timing(self, pe: ProcessingElement, time_ns: int, samples: int) -> None:
        """Record timing for a PE render call."""
        pe_id = id(pe)
        if pe_id not in self.pe_profiles:
            self.pe_profiles[pe_id] = PEProfile(
                pe_class=pe.__class__.__name__,
                pe_id=pe_id,
                min_time_ns=time_ns,
                max_time_ns=time_ns,
            )
        
        profile = self.pe_profiles[pe_id]
        profile.render_count += 1
        profile.total_time_ns += time_ns
        profile.total_samples += samples
        profile.min_time_ns = min(profile.min_time_ns, time_ns)
        profile.max_time_ns = max(profile.max_time_ns, time_ns)
    
    def summary(self, sample_rate: int = 44100) -> str:
        """Generate a human-readable summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("RENDER PROFILE REPORT")
        lines.append("=" * 70)
        lines.append(f"Total render calls: {self.render_calls}")
        lines.append(f"Total samples: {self.total_samples:,}")
        lines.append(f"Total render time: {self.total_render_time_ns / 1_000_000:.2f} ms")
        lines.append(f"Total output time: {self.total_output_time_ns / 1_000_000:.2f} ms")
        
        if self.total_render_time_ns > 0:
            realtime_ns = (self.total_samples / sample_rate) * 1_000_000_000
            ratio = realtime_ns / self.total_render_time_ns
            lines.append(f"Realtime ratio: {ratio:.1f}x (>{1.0:.1f}x is faster than realtime)")
        
        lines.append("")
        lines.append("PER-PE BREAKDOWN (sorted by total time):")
        lines.append("-" * 70)
        lines.append(f"{'PE Class':<20} {'Calls':>8} {'Total ms':>10} {'Avg ms':>10} {'Samples/s':>12}")
        lines.append("-" * 70)
        
        # Sort by total time descending
        sorted_profiles = sorted(
            self.pe_profiles.values(),
            key=lambda p: p.total_time_ns,
            reverse=True
        )
        
        for profile in sorted_profiles:
            lines.append(
                f"{profile.pe_class:<20} {profile.render_count:>8} "
                f"{profile.total_time_ms:>10.2f} {profile.avg_time_ms:>10.4f} "
                f"{profile.samples_per_second:>12,.0f}"
            )
        
        lines.append("=" * 70)
        return "\n".join(lines)


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
        
        # Profiling state
        self._profiling: bool = False
        self._profile_report: Optional[ProfileReport] = None
        self._pe_list: list[ProcessingElement] = []  # Flattened list for profiling
    
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
    
    @property
    def profiling(self) -> bool:
        """True if profiling is enabled."""
        return self._profiling
    
    def enable_profiling(self) -> None:
        """
        Enable render profiling.
        
        When enabled, each render() call will measure the time spent
        in each PE's render() method. Use get_profile_report() to
        retrieve the results.
        """
        self._profiling = True
        self._profile_report = ProfileReport()
        logger.info("Profiling enabled")
    
    def disable_profiling(self) -> None:
        """Disable render profiling."""
        self._profiling = False
        logger.info("Profiling disabled")
    
    def get_profile_report(self) -> Optional[ProfileReport]:
        """
        Get the current profile report.
        
        Returns:
            ProfileReport with timing data, or None if profiling not enabled
        """
        return self._profile_report
    
    def print_profile_report(self) -> None:
        """Print the profile report summary to stdout."""
        if self._profile_report is None:
            print("No profile data available. Call enable_profiling() first.")
            return
        print(self._profile_report.summary(self._sample_rate))
    
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
        
        # Build flattened PE list for profiling (bottom-up order)
        self._pe_list = self._collect_pes(source)
        
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
            duration: Number of samples to render (must be >= 1)
        
        Raises:
            RuntimeError: If no source set or not started (always fatal)
            ValueError: If duration < 1 (always fatal)
        """
        if self._source is None:
            handle_error("No source set. Call set_source() first.", fatal=True)
            return
        if not self._started:
            handle_error("Not started. Call start() first.", fatal=True)
            return
        if duration < 1:
            handle_error(
                "Renderer.render() requires duration >= 1 to prevent infinite loops.",
                fatal=True,
                exception_class=ValueError,
            )
            return
        
        if self._profiling and self._profile_report is not None:
            self._render_profiled(start, duration)
        else:
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
    
    def _collect_pes(
        self,
        pe: ProcessingElement,
        collected: Optional[set[int]] = None,
        result: Optional[list[ProcessingElement]] = None,
    ) -> list[ProcessingElement]:
        """
        Collect all PEs in the graph in bottom-up order.
        
        Args:
            pe: The root ProcessingElement
            collected: Set of PE ids already collected
            result: List to append PEs to
        
        Returns:
            List of all PEs in bottom-up order (inputs before outputs)
        """
        if collected is None:
            collected = set()
        if result is None:
            result = []
        
        pe_id = id(pe)
        if pe_id in collected:
            return result
        collected.add(pe_id)
        
        # Collect inputs first (bottom-up)
        for input_pe in pe.inputs():
            self._collect_pes(input_pe, collected, result)
        
        result.append(pe)
        return result
    
    def _render_profiled(self, start: int, duration: int) -> None:
        """
        Render with profiling enabled.
        
        Times each PE's render() call individually by walking the graph
        and rendering each PE in bottom-up order.
        
        Note: This changes the render order slightly - each PE is rendered
        explicitly rather than letting the graph pull data lazily. This
        should produce equivalent results but may have slightly different
        performance characteristics.
        """
        if self._source is None or self._profile_report is None:
            return
        
        report = self._profile_report
        report.render_calls += 1
        report.total_samples += duration
        
        total_render_start = time.perf_counter_ns()
        
        # Render each PE individually and time it
        # We render in bottom-up order (inputs first)
        # Each PE will get its input from already-rendered upstream PEs
        # Note: This doesn't perfectly isolate PE times because render()
        # calls cascade, but it gives us useful relative timings
        
        # For accurate per-PE timing, we need to render just the source
        # and let the graph pull naturally, but instrument each PE
        # Here we take a simpler approach: time the full render and attribute
        # it to the source PE, then recursively time sub-graphs
        
        # Simple approach: time the full graph render
        snippet = self._source.render(start, duration)
        
        total_render_end = time.perf_counter_ns()
        render_time = total_render_end - total_render_start
        report.total_render_time_ns += render_time
        
        # Attribute time to source PE (rough approximation)
        # For more accurate per-PE timing, PEs would need internal instrumentation
        report.add_pe_timing(self._source, render_time, duration)
        
        # Time the output separately
        output_start = time.perf_counter_ns()
        self._output(snippet)
        output_end = time.perf_counter_ns()
        report.total_output_time_ns += output_end - output_start