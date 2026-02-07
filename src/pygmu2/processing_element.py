"""
ProcessingElement and SourcePE abstract base classes.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations
import time
from abc import ABC, abstractmethod
from typing import Optional, Union

from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.config import (
    get_sample_rate,
    handle_error,
)
from pygmu2.diagnostics import (
    is_enabled,
    pull_count_enabled,
    record_pull,
    record_timing,
    timing_enabled,
)


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

    # For impure PEs: end of last render request; used to enforce contiguous requests
    _last_rendered_end: Optional[int] = None

    def __new__(cls, *args, **kwargs):
        """
        Enforce global sample rate requirement before any PE is constructed.

        This runs even when subclasses override __init__ (no super().__init__ needed).
        """
        sample_rate = get_sample_rate()
        if sample_rate is None:
            raise RuntimeError(
                "Global sample_rate is required but not set. "
                "Call pygmu2.set_sample_rate(rate) before constructing PEs."
            )
        obj = super().__new__(cls)
        obj._sample_rate = sample_rate
        return obj

    @property
    def sample_rate(self) -> Optional[int]:
        """
        The sample rate in Hz, if known.

        Returns None if the rate has not been configured and cannot be inferred
        from inputs.
        """
        if self._sample_rate is not None:
            return self._sample_rate

        inferred = None
        for input_pe in self.inputs():
            rate = input_pe.sample_rate
            if rate is None:
                continue
            if inferred is None:
                inferred = rate
            elif inferred != rate:
                handle_error(
                    f"{self.__class__.__name__}.sample_rate inferred conflicting input rates: "
                    f"{inferred} vs {rate}. Using {inferred}.",
                    fatal=False,
                )
                break

        return inferred
    
    def render(self, start: int, duration: int) -> Snippet:
        """
        Generate audio samples for the given range.
        
        This method ALWAYS returns a Snippet of exactly `duration` samples
        starting at `start`. Samples outside self.extent() are zero-filled.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (must be >= 0)
        
        Returns:
            Snippet containing the requested audio data

        Notes:
            Implementations must treat input Snippet buffers as immutable.
            Do not modify `snippet.data` from any input PE in-place.
        """
        if duration < 0:
            raise ValueError(f"duration must be >= 0, got {duration}")

        if is_enabled() and pull_count_enabled():
            record_pull(self)

        if duration == 0:
            # Determine channel count (concrete value needed for shape).
            channels = self.channel_count()
            if channels is None:
                # A 0-length snippet is semantically empty; don't overthink it.
                # Default to mono when channel count is dynamic/unknown.
                channels = 1

            return Snippet.from_zeros(start, 0, int(channels))

        # Impure PEs require contiguous requests (state precludes arbitrary render times)
        if not self.is_pure():
            if self._last_rendered_end is not None and start != self._last_rendered_end:
                raise ValueError(
                    f"{self.__class__.__name__} is not pure; render requests must be contiguous. "
                    f"Expected start={self._last_rendered_end}, got start={start}."
                )

        if is_enabled() and timing_enabled():
            t0 = time.perf_counter_ns()
            result = self._render(start, duration)
            record_timing(self, time.perf_counter_ns() - t0)
        else:
            result = self._render(start, duration)

        if not self.is_pure():
            self._last_rendered_end = start + duration

        return result

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
        Returns True if this PE is pure (arbitrary render times, multi-sink OK).
        
        pure == True: render() may be called with arbitrary (start, duration)
        in any order; same (start, duration) always yields the same output.
        Multiple consumers (sinks) are allowed.
        
        pure == False: the PE has state. After the first call, each
        render(start, duration) must have start equal to the end of the
        previous request (contiguous, no gaps, no out-of-order). The
        framework enforces this. Exactly one consumer (sink) is allowed.
        
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
        
        Called by Renderer.start() in bottom-up order (inputs first).
        Performs framework work (e.g. reset contiguous-request watermark for
        impure PEs), then calls _on_start() if the subclass implements it.
        Subclasses should override _on_start() (not this method).
        """
        if not self.is_pure():
            self._last_rendered_end = None
        if hasattr(self, '_on_start'):
            self._on_start()

    def on_stop(self) -> None:
        """
        Called once after final render.
        
        Called by Renderer.stop() in top-down order (outputs first).
        Calls _on_stop() if the subclass implements it.
        Subclasses should override _on_stop() (not this method).
        """
        if hasattr(self, '_on_stop'):
            self._on_stop()
    
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
        
        Default implementation calls _reset_state() if it exists.
        For impure PEs, also resets the contiguous-request watermark so the
        next render() may use any start (new stream).
        """
        if not self.is_pure():
            self._last_rendered_end = None
        if hasattr(self, '_reset_state'):
            self._reset_state()

    def _time_to_samples(
        self,
        *,
        samples: Optional[int] = None,
        seconds: Optional[float] = None,
        name: str = "time",
    ) -> int:
        """
        Resolve a time parameter specified in either samples or seconds.

        Conventions:
        - At most one of `samples` or `seconds` may be provided.
        - If neither is provided, resolve to 0 samples (useful for optional times).
        - Values must be non-negative.
        - Seconds are converted using the configured sample rate and rounded to
          the nearest sample.
        """
        if samples is None and seconds is None:
            return 0

        if samples is not None and seconds is not None:
            raise ValueError(
                f"{name}: specify either {name}_samples or {name}_seconds, not both "
                f"(got {name}_samples={samples}, {name}_seconds={seconds})"
            )

        if samples is not None:
            s = int(samples)
            if s < 0:
                raise ValueError(f"{name}_samples must be non-negative (got {samples})")
            return s

        from pygmu2.conversions import seconds_to_samples

        sec = float(seconds)
        if sec < 0.0:
            raise ValueError(f"{name}_seconds must be non-negative (got {seconds})")
        return int(round(float(seconds_to_samples(sec, self.sample_rate))))

    def _scalar_or_pe_values(
        self,
        param: Union[float, int, "ProcessingElement"],
        start: int,
        duration: int,
        *,
        dtype: "object" = None,
        channel: int = 0,
        allow_multichannel: bool = False,
        channels: Optional[int] = None,
    ):
        """
        Protected helper for "scalar-or-PE" parameters.

        Many processing elements accept either a scalar value or a ProcessingElement.
        This method handles this common case, returning a 1D array of constant values
        (for a scalar parameter) or rendered data from the ProcessingElement (for a 
        ProcessingElement parameter).

        Conventions:
        - **Default is 1D control**: returns a 1D array of shape (duration,).
          If `param` is a PE with multiple channels, channel 0 is used by default.
        - **Optional multi-channel**: set allow_multichannel=True to return a 2D
          array of shape (duration, channels). For scalar params, you must pass
          `channels` (or it defaults to 1).

        Args:
            param: scalar (float/int) or a ProcessingElement
            start: start sample index to render (if param is a PE)
            duration: number of samples
            dtype: numpy dtype (default: np.float64)
            channel: which channel to select when returning 1D from a multi-channel PE
            allow_multichannel: if True, return the full (duration, channels) array
            channels: required when allow_multichannel=True and param is scalar

        Returns:
            np.ndarray: shape (duration,) by default, or (duration, channels) if
            allow_multichannel=True.
        """
        import numpy as np

        if dtype is None:
            dtype = np.float64

        if duration <= 0:
            if allow_multichannel:
                ch = channels if channels is not None else 1
                return np.zeros((0, ch), dtype=dtype)
            return np.zeros((0,), dtype=dtype)

        if isinstance(param, ProcessingElement):
            data = param.render(start, duration).data
            if allow_multichannel:
                return data.astype(dtype, copy=False)

            # 1D control: use one channel (default 0)
            if data.ndim != 2 or data.shape[1] < 1:
                raise ValueError(f"param PE returned invalid shape {getattr(data, 'shape', None)}")
            if channel < 0 or channel >= data.shape[1]:
                raise ValueError(f"channel {channel} out of range for param with {data.shape[1]} channels")
            return data[:, channel].astype(dtype, copy=False)

        # Scalar value
        value = float(param)
        if allow_multichannel:
            ch = channels if channels is not None else 1
            return np.full((duration, ch), value, dtype=dtype)
        return np.full((duration,), value, dtype=dtype)

class SourcePE(ProcessingElement):
    """
    Abstract base class for source ProcessingElements (no inputs).
    
    Sources generate audio from external data (files, synthesis, etc.)
    rather than processing input from other PEs.
    
    Sources are typically pure (arbitrary render times, multi-sink OK) and must declare their
    output channel count explicitly.
    """
    
    def inputs(self) -> list[ProcessingElement]:
        """Sources have no inputs."""
        return []
    
    def is_pure(self) -> bool:
        """
        Sources are typically pure (arbitrary render times, multi-sink OK).
        
        Override and return False for sources with state that require
        contiguous render requests.
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
