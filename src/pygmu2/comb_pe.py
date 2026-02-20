"""
CombPE - Feedback comb filter with frequency control.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import numpy as np

# Try to import numba for JIT compilation (optional optimization)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator when numba isn't available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def _comb_process_numba(
    x: np.ndarray,
    freq_values: np.ndarray,
    fb_values: np.ndarray,
    buffer: np.ndarray,
    write_pos: int,
    smoothed_freq: float,
    sample_rate: float,
    min_frequency: float,
    smoothing_samples: int,
    max_feedback: float,
) -> tuple:
    """
    Numba-accelerated comb filter processing.

    Implements a feedback comb filter where delay length is set to one period
    of the target frequency: delay_samples = sample_rate / frequency.

    The difference equation is:
        y[n] = x[n] + feedback * y[n - delay]

    Frequency changes are smoothed using a one-pole lowpass filter to prevent
    clicks and zipper noise when the frequency parameter changes rapidly.

    Returns (output, buffer, write_pos, smoothed_freq).
    """
    duration = x.shape[0]
    channels = x.shape[1]
    y = np.zeros((duration, channels), dtype=np.float64)
    buffer_len = buffer.shape[0]

    # Smoothing coefficient for one-pole lowpass on frequency changes
    smooth_alpha = 1.0 / smoothing_samples

    for n in range(duration):
        # --- Frequency smoothing ---
        # Clamp raw frequency to minimum to prevent excessively long delays
        raw_freq = freq_values[n]
        if raw_freq < min_frequency:
            raw_freq = min_frequency

        # Initialize or update smoothed frequency with one-pole lowpass
        if smoothed_freq < 0.0:
            smoothed_freq = raw_freq
        else:
            smoothed_freq += (raw_freq - smoothed_freq) * smooth_alpha

        # --- Calculate delay length from frequency ---
        # delay_samples = sample_rate / frequency gives one period of delay
        freq = smoothed_freq
        if freq < 1.0:
            freq = 1.0
        delay_samples = int(np.round(sample_rate / freq))

        # Clamp delay to valid buffer range
        if delay_samples < 1:
            delay_samples = 1
        if delay_samples >= buffer_len:
            delay_samples = buffer_len - 1

        # --- Calculate read position (circular buffer) ---
        read_pos = write_pos - delay_samples
        if read_pos < 0:
            read_pos += buffer_len

        # --- Clamp feedback to prevent instability ---
        fb = fb_values[n]
        if not np.isfinite(fb):
            fb = 0.0
        if fb > max_feedback:
            fb = max_feedback
        if fb < -max_feedback:
            fb = -max_feedback

        # --- Apply comb filter equation: y[n] = x[n] + fb * y[n-delay] ---
        for c in range(channels):
            delayed = buffer[read_pos, c]
            out_sample = x[n, c] + fb * delayed
            buffer[write_pos, c] = out_sample
            y[n, c] = out_sample

        # Advance circular buffer write position
        write_pos += 1
        if write_pos >= buffer_len:
            write_pos = 0

    return y, buffer, write_pos, smoothed_freq

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class CombPE(ProcessingElement):
    """
    Feedback comb filter tuned by a target frequency.

    The delay length is set to one period of the target frequency
    (delay_samples = sample_rate / frequency). The output is:
        y[n] = x[n] + feedback * y[n - delay]

    Args:
        source: Input audio PE
        frequency: Target frequency in Hz (float or PE)
        feedback: Feedback amount (-0.995..0.995, float or PE)
        min_frequency: Minimum frequency in Hz (limits max delay)
        smoothing_samples: Smoothing window for frequency changes
    """

    _MAX_FEEDBACK = 0.995

    def __init__(
        self,
        source: ProcessingElement,
        frequency: float | ProcessingElement,
        feedback: float | ProcessingElement = 0.0,
        min_frequency: float = 20.0,
        smoothing_samples: int = 2400,
    ):
        self._source = source
        self._frequency = frequency
        self._feedback = feedback
        self._min_frequency = max(1.0, float(min_frequency))
        self._smoothing_samples = max(1, int(smoothing_samples))

        self._freq_is_pe = isinstance(frequency, ProcessingElement)
        self._fb_is_pe = isinstance(feedback, ProcessingElement)

        self._buffer: np.ndarray | None = None
        self._write_pos: int = 0
        self._smoothed_freq: float = -1.0

    @property
    def source(self) -> ProcessingElement:
        """The input audio PE."""
        return self._source

    @property
    def frequency(self) -> float | ProcessingElement:
        """Target frequency in Hz."""
        return self._frequency

    @property
    def feedback(self) -> float | ProcessingElement:
        """Feedback amount (-0.995..0.995)."""
        return self._feedback

    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs (source and any parameter PEs)."""
        inputs = [self._source]
        if self._freq_is_pe:
            inputs.append(self._frequency)
        if self._fb_is_pe:
            inputs.append(self._feedback)
        return inputs

    def is_pure(self) -> bool:
        """CombPE maintains state and is not pure."""
        return False

    def channel_count(self) -> int | None:
        """Pass through channel count from source."""
        return self._source.channel_count()

    def _compute_extent(self) -> Extent:
        """Intersect source extent with parameter extents."""
        extent = self._source.extent()

        if self._freq_is_pe:
            freq_extent = self._frequency.extent()
            extent = extent.intersection(freq_extent) or extent

        if self._fb_is_pe:
            fb_extent = self._feedback.extent()
            extent = extent.intersection(fb_extent) or extent

        return extent

    def _allocate_buffer(self, channels: int) -> None:
        """
        Allocate delay buffer and reset state.

        Buffer size is determined by the minimum frequency setting:
        lower min_frequency requires a longer buffer to hold one period.

        Args:
            channels: Number of audio channels
        """
        # Buffer must hold at least one period at minimum frequency
        max_delay = int(np.ceil(self.sample_rate / self._min_frequency))
        max_delay = max(2, max_delay + 1)
        self._buffer = np.zeros((max_delay, channels), dtype=np.float64)
        self._write_pos = 0
        self._smoothed_freq = -1.0  # -1 signals uninitialized

    def _on_start(self) -> None:
        """Allocate delay buffer and reset state."""
        channels = self._source.channel_count() or 1
        self._allocate_buffer(channels)

    def _on_stop(self) -> None:
        """Clear state."""
        self._buffer = None
        self._write_pos = 0
        self._smoothed_freq = -1.0

    def _clamp_feedback(self, value: float) -> float:
        """
        Clamp feedback to prevent instability.

        Feedback values >= 1.0 would cause the filter to self-oscillate
        with growing amplitude. We limit to MAX_FEEDBACK (0.995) to
        allow sustained resonance without runaway.

        Args:
            value: Raw feedback value

        Returns:
            Clamped feedback in range [-MAX_FEEDBACK, MAX_FEEDBACK]
        """
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, -self._MAX_FEEDBACK, self._MAX_FEEDBACK))

    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render comb-filtered audio.

        Args:
            start: Starting sample index
            duration: Number of samples to render

        Returns:
            Snippet containing comb-filtered audio
        """
        source_snippet = self._source.render(start, duration)
        data = source_snippet.data.astype(np.float64)
        samples, channels = data.shape

        # Ensure delay buffer is allocated with correct channel count
        if self._buffer is None or self._buffer.shape[1] != channels:
            self._allocate_buffer(channels)

        buffer = self._buffer
        if buffer is None:
            buffer = np.zeros((2, channels), dtype=np.float64)

        buffer_len = buffer.shape[0]

        # Smoothing coefficient for frequency changes
        smooth_alpha = 1.0 / self._smoothing_samples

        # Get frequency values (either constant or from PE)
        freq_values = self._scalar_or_pe_values(self._frequency, start, duration, dtype=np.float64)

        # Get feedback values (either constant or from PE)
        fb_values = self._scalar_or_pe_values(self._feedback, start, duration, dtype=np.float64)

        # Use Numba-accelerated path when available
        if NUMBA_AVAILABLE:
            output, buffer, self._write_pos, self._smoothed_freq = _comb_process_numba(
                data,
                freq_values,
                fb_values,
                buffer,
                int(self._write_pos),
                float(self._smoothed_freq),
                float(self.sample_rate),
                float(self._min_frequency),
                int(self._smoothing_samples),
                float(self._MAX_FEEDBACK),
            )
            self._buffer = buffer
            return Snippet(start, output.astype(np.float32))

        # --- Pure Python fallback path ---
        output = np.zeros_like(data, dtype=np.float64)

        for i in range(samples):
            # Smooth frequency changes to prevent zipper noise
            raw_freq = float(freq_values[i])
            raw_freq = max(self._min_frequency, raw_freq)
            if self._smoothed_freq < 0:
                self._smoothed_freq = raw_freq
            else:
                self._smoothed_freq += (raw_freq - self._smoothed_freq) * smooth_alpha

            # Calculate delay length: one period at target frequency
            delay_samples = int(round(self.sample_rate / max(self._smoothed_freq, 1.0)))
            if delay_samples < 1:
                delay_samples = 1
            if delay_samples >= buffer_len:
                delay_samples = buffer_len - 1

            # Read from circular buffer (delayed output)
            read_pos = self._write_pos - delay_samples
            if read_pos < 0:
                read_pos += buffer_len

            # Comb filter: y[n] = x[n] + feedback * y[n - delay]
            delayed = buffer[read_pos]
            fb = self._clamp_feedback(float(fb_values[i]))
            out_sample = data[i] + fb * delayed

            # Write to circular buffer and output
            buffer[self._write_pos] = out_sample
            output[i] = out_sample

            # Advance circular buffer position
            self._write_pos += 1
            if self._write_pos >= buffer_len:
                self._write_pos = 0

        self._buffer = buffer
        return Snippet(start, output.astype(np.float32))

    def __repr__(self) -> str:
        freq_repr = self._frequency if not self._freq_is_pe else f"{self._frequency.__class__.__name__}(...)"
        fb_repr = self._feedback if not self._fb_is_pe else f"{self._feedback.__class__.__name__}(...)"
        return (
            f"CombPE(source={self._source.__class__.__name__}, "
            f"frequency={freq_repr}, feedback={fb_repr})"
        )
