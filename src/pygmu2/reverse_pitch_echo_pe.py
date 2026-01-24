"""
ReversePitchEchoPE - Pitch-shifted reverse echo effect.

Inspired by CCRMA "Pitch Shifting Reverse Echo" technique and msynth.
    https://ccrma.stanford.edu/~jingjiez/portfolio/echoing-harmonics/pdfs/A%20Pitch%20Shifting%20Reverse%20Echo%20Audio%20Effect.pdf
Implements block-based reverse playback with optional alternating direction,
Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from typing import Optional, Union
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
def _reverse_pitch_echo_numba(
    x: np.ndarray,
    block_values: np.ndarray,
    pitch_values: np.ndarray,
    fb_values: np.ndarray,
    alt_values: np.ndarray,
    buffer_a: np.ndarray,
    buffer_b: np.ndarray,
    current_is_a: int,
    pitch_buffer: np.ndarray,
    pitch_write_pos: int,
    pitch_read_pos: float,
    write_idx: int,
    read_idx: int,
    smoothed_block_samples: float,
    current_block_samples: int,
    previous_block_samples: int,
    playback_reverse: int,
    sample_rate: float,
    min_block_samples: int,
    max_delay_samples: int,
    max_feedback: float,
    smoothing_samples: int,
) -> tuple:
    """
    Numba-accelerated reverse pitch echo processing.

    This effect works by:
    1. Pitch-shifting the input using a dual-head time-domain pitch shifter
    2. Writing the pitch-shifted audio into fixed-length blocks
    3. Playing back completed blocks in reverse (or alternating direction)
    4. Applying a Hann window during playback to prevent clicks at block edges
    5. Feeding the windowed output back into the input with adjustable feedback

    The algorithm uses double-buffering: while writing to one buffer, the
    previous buffer is read back (in reverse or forward). When a block
    completes, the buffers swap roles.

    Pitch shifting uses two read heads 180 degrees apart with crossfading
    based on distance from the write head. This prevents discontinuities
    when the read head crosses the write position.

    Returns tuple of updated state variables.
    """
    duration = x.shape[0]
    channels = x.shape[1]
    y = np.zeros((duration, channels), dtype=np.float64)

    # Smoothing coefficient for block size changes
    smooth_alpha = 1.0 / smoothing_samples

    pitch_len = pitch_buffer.shape[0]
    max_block_samples = max_delay_samples - 1

    for n in range(duration):
        # --- Update block size with smoothing ---
        # Convert block_seconds to samples and smooth changes
        target_samples = block_values[n] * sample_rate
        if not np.isfinite(target_samples):
            target_samples = min_block_samples
        if target_samples < min_block_samples:
            target_samples = min_block_samples
        if target_samples > max_block_samples:
            target_samples = max_block_samples
        target_samples = float(np.round(target_samples))
        smoothed_block_samples += (target_samples - smoothed_block_samples) * smooth_alpha

        # Lock in block size at start of each new block
        if write_idx == 0:
            current_block_samples = int(np.round(smoothed_block_samples))
            if current_block_samples < min_block_samples:
                current_block_samples = min_block_samples
            if current_block_samples > max_block_samples:
                current_block_samples = max_block_samples

        # --- Pitch shifting ---
        ratio = pitch_values[n]
        if ratio < 0.001:
            ratio = 0.001

        # Write input to pitch buffer (circular)
        for c in range(channels):
            pitch_buffer[pitch_write_pos, c] = x[n, c]
        pitch_write_pos += 1
        if pitch_write_pos >= pitch_len:
            pitch_write_pos = 0

        # Bypass pitch shifting if ratio is ~1.0 (unity)
        if abs(ratio - 1.0) < 1e-4:
            pitch_read_pos += ratio
            if pitch_read_pos >= pitch_len:
                pitch_read_pos -= pitch_len
            for c in range(channels):
                # No pitch shift needed - use input directly
                pitched_c = x[n, c]

                # Select current and previous buffers
                if current_is_a == 1:
                    current_buffer = buffer_a
                    previous_buffer = buffer_b
                else:
                    current_buffer = buffer_b
                    previous_buffer = buffer_a

                # --- Read from previous block with Hann window ---
                wet = 0.0
                if previous_block_samples > 0 and read_idx < previous_block_samples:
                    # Calculate read position (forward or reverse)
                    idx = previous_block_samples - 1 - read_idx if playback_reverse == 1 else read_idx
                    if 0 <= idx < previous_block_samples:
                        # Hann window: 0.5 - 0.5*cos(2*pi*t) for t in [0,1]
                        if previous_block_samples > 1:
                            pos = read_idx / (previous_block_samples - 1.0)
                        else:
                            pos = 0.0
                        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * pos)
                        wet = previous_buffer[idx, c] * window

                # Clamp feedback to prevent instability
                fb = fb_values[n]
                if not np.isfinite(fb):
                    fb = 0.0
                if fb > max_feedback:
                    fb = max_feedback
                if fb < -max_feedback:
                    fb = -max_feedback

                # Write pitched input + feedback to current buffer
                current_buffer[write_idx, c] = pitched_c + wet * fb
                # Output is the windowed playback from previous block
                y[n, c] = wet
        else:
            # --- Full pitch shifting with dual read heads ---
            # Primary read head position
            pos = pitch_read_pos % pitch_len
            idx0 = int(np.floor(pos))
            idx1 = idx0 + 1
            if idx1 >= pitch_len:
                idx1 = 0
            frac = pos - idx0

            # Secondary read head (180 degrees ahead)
            pos2 = pos + pitch_len / 2.0
            if pos2 >= pitch_len:
                pos2 -= pitch_len
            idx2 = int(np.floor(pos2))
            idx3 = idx2 + 1
            if idx3 >= pitch_len:
                idx3 = 0
            frac2 = pos2 - idx2

            # Crossfade factor based on distance from write head
            # This prevents clicks when read head crosses write position
            dist = pitch_read_pos - pitch_write_pos
            if dist < 0:
                dist = -dist
            if dist > pitch_len / 2.0:
                dist = pitch_len - dist
            f = dist / (pitch_len / 2.0)

            # Advance read position by pitch ratio
            pitch_read_pos += ratio
            if pitch_read_pos >= pitch_len:
                pitch_read_pos -= pitch_len

            for c in range(channels):
                # Linear interpolation for each read head
                s1 = (1.0 - frac) * pitch_buffer[idx0, c] + frac * pitch_buffer[idx1, c]
                s2 = (1.0 - frac2) * pitch_buffer[idx2, c] + frac2 * pitch_buffer[idx3, c]
                # Crossfade between the two read heads
                pitched_c = f * s1 + (1.0 - f) * s2

                # Select current and previous buffers
                if current_is_a == 1:
                    current_buffer = buffer_a
                    previous_buffer = buffer_b
                else:
                    current_buffer = buffer_b
                    previous_buffer = buffer_a

                # --- Read from previous block with Hann window ---
                wet = 0.0
                if previous_block_samples > 0 and read_idx < previous_block_samples:
                    idx = previous_block_samples - 1 - read_idx if playback_reverse == 1 else read_idx
                    if 0 <= idx < previous_block_samples:
                        if previous_block_samples > 1:
                            pos = read_idx / (previous_block_samples - 1.0)
                        else:
                            pos = 0.0
                        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * pos)
                        wet = previous_buffer[idx, c] * window

                # Clamp feedback
                fb = fb_values[n]
                if not np.isfinite(fb):
                    fb = 0.0
                if fb > max_feedback:
                    fb = max_feedback
                if fb < -max_feedback:
                    fb = -max_feedback

                # Write to current buffer with feedback
                current_buffer[write_idx, c] = pitched_c + wet * fb
                y[n, c] = wet

        # Advance buffer indices
        write_idx += 1
        read_idx += 1

        # --- Block boundary: swap buffers ---
        if write_idx >= current_block_samples:
            # Swap which buffer is "current" vs "previous"
            current_is_a = 1 - current_is_a
            previous_block_samples = current_block_samples
            write_idx = 0
            read_idx = 0

            # Update playback direction for next block
            alternate = alt_values[n] >= 0.5
            if alternate:
                playback_reverse = 1 - playback_reverse
            else:
                playback_reverse = 1  # Always reverse when not alternating

    return (
        y,
        current_is_a,
        pitch_write_pos,
        pitch_read_pos,
        write_idx,
        read_idx,
        smoothed_block_samples,
        current_block_samples,
        previous_block_samples,
        playback_reverse,
    )

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class _PitchShifter:
    """
    Lightweight time-domain pitch shifter using dual read heads.

    This implements a simple pitch shifting technique where:
    - Audio is written to a circular buffer at the normal rate
    - Audio is read back at a different rate (faster = higher pitch)
    - Two read heads 180 degrees apart are crossfaded to prevent
      discontinuities when a read head crosses the write position

    The crossfade factor is based on the distance between the read head
    and write head: when closest to write head, we fade to the other
    read head which is far away.
    """

    def __init__(self, buffer_size: int, channels: int):
        """
        Initialize the pitch shifter.

        Args:
            buffer_size: Size of the circular buffer (determines latency)
            channels: Number of audio channels
        """
        self._buffer = np.zeros((buffer_size, channels), dtype=np.float64)
        self._write_pos = 0
        self._read_pos = 0.0

    @property
    def buffer_size(self) -> int:
        """Return the buffer size in samples."""
        return self._buffer.shape[0]

    def _interpolated_read(self, position: float) -> np.ndarray:
        """
        Read from buffer with linear interpolation.

        Args:
            position: Fractional read position in the circular buffer

        Returns:
            Interpolated sample (all channels)
        """
        size = self._buffer.shape[0]
        pos = position % size
        idx0 = int(np.floor(pos))
        idx1 = (idx0 + 1) % size
        frac = pos - idx0
        return (1.0 - frac) * self._buffer[idx0] + frac * self._buffer[idx1]

    def process(self, input_sample: np.ndarray, ratio: float) -> np.ndarray:
        """
        Process one sample through the pitch shifter.

        Args:
            input_sample: Input sample (all channels)
            ratio: Pitch ratio (>1 = higher, <1 = lower, 1 = unity)

        Returns:
            Pitch-shifted output sample (all channels)
        """
        buffer = self._buffer
        size = buffer.shape[0]

        # Write input to circular buffer
        buffer[self._write_pos] = input_sample
        self._write_pos = (self._write_pos + 1) % size

        # Clamp ratio to valid range
        ratio = max(0.001, float(ratio))

        # Bypass pitch shifting if ratio is ~1.0 (unity)
        if abs(ratio - 1.0) < 1e-4:
            self._read_pos = (self._read_pos + ratio) % size
            return input_sample

        # Read from two heads 180 degrees apart
        s1 = self._interpolated_read(self._read_pos)
        s2 = self._interpolated_read((self._read_pos + size / 2.0) % size)

        # Crossfade based on distance from write head
        # When read head is close to write head, fade to the other head
        dist = abs(self._read_pos - self._write_pos)
        if dist > size / 2.0:
            dist = size - dist
        f = dist / (size / 2.0)
        sample = f * s1 + (1.0 - f) * s2

        # Advance read position by pitch ratio
        self._read_pos = (self._read_pos + ratio) % size
        return sample


class ReversePitchEchoPE(ProcessingElement):
    """
    Pitch-shifted reverse echo effect.

    The input is pitch-shifted and written into blocks. Each completed block
    is then played back in reverse (or alternating forward/reverse), with a
    Hann window applied to suppress clicks.

    Args:
        source: Input audio PE
        block_seconds: Block length in seconds (float or PE)
        pitch_ratio: Pitch shift ratio (float or PE, 1.0 = unity)
        feedback: Feedback amount (0..0.995, float or PE)
        alternate_direction: If >= 0.5, alternate playback direction each block
    """

    _MAX_DELAY_SECONDS = 10.0
    _MIN_BLOCK_SAMPLES = 64
    _MAX_FEEDBACK = 0.995

    def __init__(
        self,
        source: ProcessingElement,
        block_seconds: Union[float, ProcessingElement] = 0.25,
        pitch_ratio: Union[float, ProcessingElement] = 1.0,
        feedback: Union[float, ProcessingElement] = 0.85,
        alternate_direction: Union[float, ProcessingElement] = 0.0,
        smoothing_samples: int = 2400,
    ):
        self._source = source
        self._block_seconds = block_seconds
        self._pitch_ratio = pitch_ratio
        self._feedback = feedback
        self._alternate_direction = alternate_direction
        self._smoothing_samples = max(1, int(smoothing_samples))

        self._block_is_pe = isinstance(block_seconds, ProcessingElement)
        self._pitch_is_pe = isinstance(pitch_ratio, ProcessingElement)
        self._fb_is_pe = isinstance(feedback, ProcessingElement)
        self._alt_is_pe = isinstance(alternate_direction, ProcessingElement)

        self._buffer_a: Optional[np.ndarray] = None
        self._buffer_b: Optional[np.ndarray] = None
        self._current_buffer: Optional[np.ndarray] = None
        self._previous_buffer: Optional[np.ndarray] = None
        self._pitch_shifter: Optional[_PitchShifter] = None
        self._pitch_buffer: Optional[np.ndarray] = None
        self._pitch_write_pos: int = 0
        self._pitch_read_pos: float = 0.0
        self._current_is_a: int = 1

        self._write_idx = 0
        self._read_idx = 0
        self._smoothed_block_samples: float = 0.0
        self._current_block_samples = 0
        self._previous_block_samples = 0
        self._playback_reverse = True

    @property
    def source(self) -> ProcessingElement:
        """The input audio PE."""
        return self._source

    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs (source and any parameter PEs)."""
        inputs = [self._source]
        if self._block_is_pe:
            inputs.append(self._block_seconds)
        if self._pitch_is_pe:
            inputs.append(self._pitch_ratio)
        if self._fb_is_pe:
            inputs.append(self._feedback)
        if self._alt_is_pe:
            inputs.append(self._alternate_direction)
        return inputs

    def is_pure(self) -> bool:
        """ReversePitchEchoPE maintains state and is not pure."""
        return False

    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()

    def _compute_extent(self) -> Extent:
        """Intersect source extent with parameter extents."""
        extent = self._source.extent()

        if self._block_is_pe:
            block_extent = self._block_seconds.extent()
            extent = extent.intersection(block_extent) or extent

        if self._pitch_is_pe:
            pitch_extent = self._pitch_ratio.extent()
            extent = extent.intersection(pitch_extent) or extent

        if self._fb_is_pe:
            fb_extent = self._feedback.extent()
            extent = extent.intersection(fb_extent) or extent

        if self._alt_is_pe:
            alt_extent = self._alternate_direction.extent()
            extent = extent.intersection(alt_extent) or extent

        return extent

    def on_start(self) -> None:
        """
        Allocate buffers and reset state for rendering.

        Creates double buffers for the block-based echo effect and
        initializes the pitch shifter. Buffer size is determined by
        the maximum delay time setting.
        """
        channels = self._source.channel_count() or 1

        # Calculate maximum buffer size from max delay setting
        max_delay_samples = int(self._MAX_DELAY_SECONDS * self.sample_rate)
        max_delay_samples = max(self._MIN_BLOCK_SAMPLES + 1, max_delay_samples)

        # Allocate double buffers for ping-pong block processing
        self._buffer_a = np.zeros((max_delay_samples, channels), dtype=np.float64)
        self._buffer_b = np.zeros((max_delay_samples, channels), dtype=np.float64)
        self._current_buffer = self._buffer_a
        self._previous_buffer = self._buffer_b
        self._current_is_a = 1

        # Allocate pitch shifter buffer (sized for lowest expected pitch)
        pitch_buffer_size = max(2, int(self.sample_rate / 60))
        self._pitch_buffer = np.zeros((pitch_buffer_size, channels), dtype=np.float64)
        self._pitch_write_pos = 0
        self._pitch_read_pos = 0.0

        # Create pitch shifter only for Python fallback path
        if NUMBA_AVAILABLE:
            self._pitch_shifter = None
        else:
            self._pitch_shifter = _PitchShifter(pitch_buffer_size, channels)

        # Initialize block processing state
        self._write_idx = 0
        self._read_idx = 0

        # Initialize smoothed block size
        if self._block_is_pe:
            initial_seconds = 0.25  # Default if block_seconds is a PE
        else:
            initial_seconds = float(self._block_seconds)
        self._smoothed_block_samples = self._clamp_block_samples(initial_seconds * self.sample_rate)
        self._current_block_samples = int(self._smoothed_block_samples)
        self._previous_block_samples = 0
        self._playback_reverse = True

    def on_stop(self) -> None:
        """Clear state."""
        self._buffer_a = None
        self._buffer_b = None
        self._current_buffer = None
        self._previous_buffer = None
        self._pitch_shifter = None
        self._pitch_buffer = None
        self._pitch_write_pos = 0
        self._pitch_read_pos = 0.0
        self._current_is_a = 1
        self._write_idx = 0
        self._read_idx = 0
        self._previous_block_samples = 0

    def _clamp_block_samples(self, samples: float) -> int:
        """
        Clamp block size to valid range.

        Args:
            samples: Desired block size in samples

        Returns:
            Clamped block size between MIN_BLOCK_SAMPLES and max delay
        """
        if not np.isfinite(samples):
            return self._MIN_BLOCK_SAMPLES
        max_samples = int(self._MAX_DELAY_SECONDS * self.sample_rate) - 1
        clamped = int(np.round(np.clip(samples, self._MIN_BLOCK_SAMPLES, max_samples)))
        return clamped

    def _clamp_feedback(self, value: float) -> float:
        """
        Clamp feedback to prevent instability.

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
        Render reverse pitch echo effect.

        The effect captures audio into blocks and plays them back in reverse
        (or alternating forward/reverse), creating ethereal echo textures.
        Pitch shifting is applied before block capture.

        Args:
            start: Starting sample index
            duration: Number of samples to render

        Returns:
            Snippet containing the processed audio (wet signal only)
        """
        source_snippet = self._source.render(start, duration)
        data = source_snippet.data.astype(np.float64)
        samples, channels = data.shape

        # Ensure buffers are allocated with correct channel count
        if (
            self._current_buffer is None
            or self._previous_buffer is None
            or self._current_buffer.shape[1] != channels
            or self._pitch_buffer is None
            or self._pitch_buffer.shape[1] != channels
        ):
            self.on_start()

        current_buffer = self._current_buffer
        previous_buffer = self._previous_buffer
        pitch_shifter = self._pitch_shifter
        pitch_buffer = self._pitch_buffer
        if current_buffer is None or previous_buffer is None or pitch_buffer is None:
            return Snippet(start, np.zeros_like(data, dtype=np.float32))

        # Get block size values (either constant or from PE)
        if self._block_is_pe:
            block_values = self._block_seconds.render(start, duration).data[:, 0].astype(np.float64)
        else:
            block_values = np.full(samples, float(self._block_seconds), dtype=np.float64)

        # Get pitch ratio values (either constant or from PE)
        if self._pitch_is_pe:
            pitch_values = self._pitch_ratio.render(start, duration).data[:, 0].astype(np.float64)
        else:
            pitch_values = np.full(samples, float(self._pitch_ratio), dtype=np.float64)

        # Get feedback values (either constant or from PE)
        if self._fb_is_pe:
            fb_values = self._feedback.render(start, duration).data[:, 0].astype(np.float64)
        else:
            fb_values = np.full(samples, float(self._feedback), dtype=np.float64)

        # Get alternate direction values (either constant or from PE)
        if self._alt_is_pe:
            alt_values = self._alternate_direction.render(start, duration).data[:, 0].astype(np.float64)
        else:
            alt_values = np.full(samples, float(self._alternate_direction), dtype=np.float64)

        # Use Numba-accelerated path when available
        if NUMBA_AVAILABLE:
            output, self._current_is_a, self._pitch_write_pos, self._pitch_read_pos, self._write_idx, self._read_idx, self._smoothed_block_samples, self._current_block_samples, self._previous_block_samples, playback_reverse = _reverse_pitch_echo_numba(
                data,
                block_values,
                pitch_values,
                fb_values,
                alt_values,
                self._buffer_a,
                self._buffer_b,
                int(self._current_is_a),
                pitch_buffer,
                int(self._pitch_write_pos),
                float(self._pitch_read_pos),
                int(self._write_idx),
                int(self._read_idx),
                float(self._smoothed_block_samples),
                int(self._current_block_samples),
                int(self._previous_block_samples),
                1 if self._playback_reverse else 0,
                float(self.sample_rate),
                int(self._MIN_BLOCK_SAMPLES),
                int(self._buffer_a.shape[0]),
                float(self._MAX_FEEDBACK),
                int(self._smoothing_samples),
            )
            self._playback_reverse = playback_reverse == 1
            # Update buffer references based on which is current
            if self._current_is_a == 1:
                self._current_buffer = self._buffer_a
                self._previous_buffer = self._buffer_b
            else:
                self._current_buffer = self._buffer_b
                self._previous_buffer = self._buffer_a
            return Snippet(start, output.astype(np.float32))

        # --- Pure Python fallback path ---
        output = np.zeros_like(data, dtype=np.float64)
        smooth_alpha = 1.0 / self._smoothing_samples

        for i in range(samples):
            # Smooth block size changes to prevent abrupt transitions
            target_samples = self._clamp_block_samples(max(0.0, float(block_values[i])) * self.sample_rate)
            self._smoothed_block_samples += (target_samples - self._smoothed_block_samples) * smooth_alpha

            # Lock in block size at start of each new block
            if self._write_idx == 0:
                self._current_block_samples = self._clamp_block_samples(self._smoothed_block_samples)

            # Pitch-shift the input sample
            pitched = pitch_shifter.process(data[i], float(pitch_values[i]))

            # --- Read from previous block with Hann window ---
            wet = output[i]
            if self._previous_block_samples > 0 and self._read_idx < self._previous_block_samples:
                # Calculate read index (forward or reverse)
                idx = (
                    self._previous_block_samples - 1 - self._read_idx
                    if self._playback_reverse
                    else self._read_idx
                )
                if 0 <= idx < self._previous_block_samples:
                    # Apply Hann window to prevent clicks at block edges
                    if self._previous_block_samples > 1:
                        pos = self._read_idx / (self._previous_block_samples - 1)
                    else:
                        pos = 0.0
                    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * pos)
                    wet[:] = previous_buffer[idx] * window
            else:
                wet[:] = 0.0

            # Write pitched input + feedback to current buffer
            fb = self._clamp_feedback(float(fb_values[i]))
            write_sample = pitched + wet * fb
            current_buffer[self._write_idx] = write_sample

            # Advance buffer indices
            self._write_idx += 1
            self._read_idx += 1

            # --- Block boundary: swap buffers ---
            if self._write_idx >= self._current_block_samples:
                # Swap current and previous buffers
                self._current_buffer, self._previous_buffer = previous_buffer, current_buffer
                current_buffer = self._current_buffer
                previous_buffer = self._previous_buffer

                self._previous_block_samples = self._current_block_samples
                self._write_idx = 0
                self._read_idx = 0

                # Update playback direction for next block
                alternate = float(alt_values[i]) >= 0.5
                self._playback_reverse = not self._playback_reverse if alternate else True

        return Snippet(start, output.astype(np.float32))

    def __repr__(self) -> str:
        block_repr = self._block_seconds if not self._block_is_pe else f"{self._block_seconds.__class__.__name__}(...)"
        pitch_repr = self._pitch_ratio if not self._pitch_is_pe else f"{self._pitch_ratio.__class__.__name__}(...)"
        fb_repr = self._feedback if not self._fb_is_pe else f"{self._feedback.__class__.__name__}(...)"
        return (
            f"ReversePitchEchoPE(source={self._source.__class__.__name__}, "
            f"block_seconds={block_repr}, pitch_ratio={pitch_repr}, "
            f"feedback={fb_repr})"
        )
