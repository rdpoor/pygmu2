"""
ProcessingElement abstract base class.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations
import time
from abc import ABC, abstractmethod

from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.config import (
    get_sample_rate,
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

    The global sample rate must be set via set_sample_rate() before any
    ProcessingElement is constructed — enforced by __new__().
    """

    # Sample rate is captured from the global set_sample_rate() at construction time
    _sample_rate: int | None = None

    # Cached extent (computed lazily on first access)
    _cached_extent: Extent | None = None

    # True if this PE holds render state (phase accumulator, filter memory,
    # delay line, ...). Stateful PEs must be rendered contiguously: each
    # render(start, duration) must have start equal to the end of the
    # previous request — enforced in render(). An explicit reset_state()
    # is the sanctioned way to seek. Stateless PEs may be rendered at
    # arbitrary positions in any order and shared by multiple sinks.
    #
    # Set `stateful = True` as a class attribute (or instance attribute in
    # __init__ when statefulness depends on constructor arguments). The
    # declaration is falsified by CI: the contract suite renders every PE
    # in shuffled order and requires it to either match the contiguous
    # reference (stateless) or raise (stateful).
    stateful: bool = False

    # Next expected start for stateful PEs (None = no render yet)
    _expected_start: int | None = None

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
    def sample_rate(self) -> int | None:
        """The sample rate in Hz, set at construction time from the global value."""
        return self._sample_rate

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

        # Stateful PEs must be pulled contiguously; a gap or seek would
        # silently produce wrong audio (e.g. a phase jump). Seeking is
        # explicit: call reset_state() first.
        if self.stateful:
            if self._expected_start is not None and start != self._expected_start:
                raise RuntimeError(
                    f"{self!r}: non-contiguous render: expected start "
                    f"{self._expected_start}, got {start}. "
                    f"Call reset_state() to seek."
                )
            self._expected_start = start + duration

        if is_enabled() and timing_enabled():
            t0 = time.perf_counter_ns()
            result = self._render(start, duration)
            record_timing(self, time.perf_counter_ns() - t0)
        else:
            result = self._render(start, duration)

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

    def channel_count(self) -> int | None:
        """
        Number of output channels this PE produces.

        Returns:
            int: Fixed channel count
            None: Same as primary input (pass-through)

        Sources (PEs with no inputs) must return int, not None.
        """
        return None  # Default: pass-through

    def on_start(self) -> None:
        """
        Called once before first render, after configure().

        Called by Renderer.start() in bottom-up order (inputs first).
        Calls _on_start() if the subclass implements it.
        Subclasses should override _on_start() (not this method).
        """
        self._expected_start = None
        if hasattr(self, "_on_start"):
            self._on_start()

    def on_stop(self) -> None:
        """
        Called once after final render.

        Called by Renderer.stop() in top-down order (outputs first).
        Calls _on_stop() if the subclass implements it.
        Subclasses should override _on_stop() (not this method).
        """
        self._expected_start = None
        if hasattr(self, "_on_stop"):
            self._on_stop()

    def reset_state(self) -> None:
        """
        Reset this PE's internal state.

        Calls _reset_state() if the subclass implements it. Stateless PEs
        typically don't implement _reset_state() (no-op). Stateful PEs can
        override _reset_state() to reset their state (e.g., oscillator
        phase, filter memory, envelope state).

        This is also the sanctioned way to seek: it clears the contiguity
        expectation, so the next render() may start anywhere.

        Useful for:
        - Resetting oscillators on gate/trigger events (analog-like behavior)
        - Resetting state when scrubbing/jogging to different positions
        - Re-initializing stateful PEs during rendering

        Default implementation calls _reset_state() if it exists.
        """
        self._expected_start = None
        if hasattr(self, "_reset_state"):
            self._reset_state()

    def _scalar_or_pe_values(
        self,
        param: float | int | "ProcessingElement",
        start: int,
        duration: int,
        *,
        dtype: "object" = None,
        channel: int = 0,
        allow_multichannel: bool = False,
        channels: int | None = None,
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
                raise ValueError(
                    f"param PE returned invalid shape {getattr(data, 'shape', None)}"
                )
            if channel < 0 or channel >= data.shape[1]:
                raise ValueError(
                    f"channel {channel} out of range for param with {data.shape[1]} channels"
                )
            return data[:, channel].astype(dtype, copy=False)

        # Scalar value
        value = float(param)
        if allow_multichannel:
            ch = channels if channels is not None else 1
            return np.full((duration, ch), value, dtype=dtype)
        return np.full((duration,), value, dtype=dtype)
