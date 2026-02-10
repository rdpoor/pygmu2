"""
ConvolvePE - streaming convolution of a source with a finite FIR filter.

ConvolvePE applies a finite filter (FIR) to a source signal via convolution.
This implementation is designed for streaming use: it supports infinite sources
and uses FFT-based overlap-save with a fixed FFT size.

Contract:
- filter must have finite extent and start at 0 (Extent(0, N)).
- filter may be mono or match src channel count:
  - if filter is mono, it is applied to every src channel
  - if filter is multi-channel, it must match src channels exactly

Notes:
- This PE is stateful (keeps filter history for streaming). For non-contiguous
  render() calls, history is cleared and prior samples are treated as zeros.
- For finite src: output extent end is src.end + (filter_len - 1)

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


def _next_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class ConvolvePE(ProcessingElement):
    """
    Streaming convolution: y = x * h

    Args:
        src: Source audio PE
        fir: FIR filter PE (must be finite and start at 0)
        fft_size: Optional FFT size for overlap-save. Must be >= filter_len.
                  If None, a default is chosen based on filter length.
    """

    def __init__(
        self,
        src: ProcessingElement,
        fir: ProcessingElement,
        *,
        fft_size: Optional[int] = None,
    ):
        self._src = src
        self._fir = fir
        self._fft_size = int(fft_size) if fft_size is not None else None

        # Cached filter/FFT state (prepared on start / first render)
        self._fir_len: Optional[int] = None
        self._H: Optional[np.ndarray] = None  # shape (bins,) or (bins, channels)

        # Streaming history
        self._tail: Optional[np.ndarray] = None  # shape (filter_len-1, channels)
        self._last_render_end: Optional[int] = None

    @property
    def src(self) -> ProcessingElement:
        return self._src

    @property
    def fir(self) -> ProcessingElement:
        return self._fir

    @property
    def fft_size(self) -> Optional[int]:
        return self._fft_size

    def inputs(self) -> list[ProcessingElement]:
        return [self._src, self._fir]

    @staticmethod
    def ir_energy_norm(filter_pe: ProcessingElement) -> float:
        """
        Compute the energy norm of a filter/IR PE: sqrt(sum of squared samples).

        Useful for normalizing convolution output so it has similar energy to the
        input. Divide the wet gain by this value when mixing dry/wet.

        The filter PE must have finite extent (extent().start and extent().end
        must be not None). If the extent is unbounded, returns 1.0. If the
        computed norm is zero or very small, returns 1.0 to avoid division by zero.

        Returns:
            The energy norm (sqrt of sum of squared samples), or 1.0 if
            unbounded or zero.
        """
        extent = filter_pe.extent()
        if extent.start is None or extent.end is None:
            return 1.0
        duration = extent.end - extent.start
        data = filter_pe.render(extent.start, duration).data
        energy_norm = np.sqrt(np.sum(data.astype(np.float64) ** 2))
        return float(energy_norm) if energy_norm > 1e-10 else 1.0

    def is_pure(self) -> bool:
        # Keeps history for streaming overlap-save
        return False

    def channel_count(self) -> Optional[int]:
        """
        Determine output channel count based on source and filter.

        Channel handling:
        - mono filter: out_ch == src_ch
        - multi-channel filter with same channel count as src: out_ch == src_ch
        - mono src + multi-channel filter: out_ch == filter channels
        """
        src_ch = self._src.channel_count()
        filt_ch = self._fir.channel_count()

        if src_ch is None and filt_ch is None:
            return None
        if src_ch is None:
            return filt_ch
        if filt_ch is None or int(filt_ch) == 1:
            # Mono (or unknown) filter: output matches source
            return src_ch

        # At this point, filt_ch > 1
        if int(src_ch) == 1:
            # Fan-out: mono src -> multi-channel via multi-channel filter
            return int(filt_ch)

        if int(filt_ch) == int(src_ch):
            # Multi-channel filter matches source channels
            return src_ch

        # Mismatched multi-channel counts; will be rejected at prepare time.
        return src_ch

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._tail = None
        self._last_render_end = None

    def _compute_extent(self) -> Extent:
        """
        Convolution extent:
        - start: same as src.start (filter starts at 0)
        - end: if src finite, src.end + (filter_len - 1)
        """
        src_ext = self._src.extent()
        filt_ext = self._fir.extent()

        # Validate filter extent contract (without rendering)
        if filt_ext.start is not None and filt_ext.start != 0:
            raise ValueError(f"ConvolvePE filter extent must start at 0, got {filt_ext}")
        if filt_ext.start is None:
            # Treat unknown start as invalid for a filter definition.
            raise ValueError(f"ConvolvePE filter extent must be finite and start at 0, got {filt_ext}")
        if filt_ext.end is None:
            raise ValueError(f"ConvolvePE filter extent must be finite, got {filt_ext}")

        filt_len = int(filt_ext.end - filt_ext.start)
        if filt_len < 1:
            # empty or invalid
            return Extent(0, 0)

        # Start matches src start (can be None)
        start = src_ext.start
        if src_ext.end is None:
            return Extent(start, None)
        return Extent(start, int(src_ext.end + (filt_len - 1)))

    def _ensure_filter_prepared(self) -> None:
        if self._H is not None and self._fir_len is not None:
            return

        filt_ext = self._fir.extent()
        if filt_ext.start != 0 or filt_ext.end is None:
            raise ValueError(f"ConvolvePE filter must have extent Extent(0, N), got {filt_ext}")

        filt_len = int(filt_ext.end)
        if filt_len < 1:
            raise ValueError("ConvolvePE filter must be non-empty")

        # Render the FIR once
        h = self._fir.render(0, filt_len).data.astype(np.float64, copy=False)
        if h.ndim != 2 or h.shape[0] != filt_len:
            raise ValueError(f"ConvolvePE filter returned invalid shape {getattr(h, 'shape', None)}")

        src_ch = self._src.channel_count()
        if src_ch is None:
            # Best effort: resolve from rendered audio at runtime; require src to be configured
            src_ch = self._src.render(0, 1).channels

        filt_ch = int(h.shape[1])
        if filt_ch == 1:
            # Mono filter applies to every source channel
            out_ch = int(src_ch)
        else:
            # Multi-channel filter:
            # - if source is mono, fan out to filter channels (e.g. mono -> stereo)
            # - otherwise, require filter channels to match source channels
            if int(src_ch) == 1:
                out_ch = filt_ch
            elif filt_ch == int(src_ch):
                out_ch = filt_ch
            else:
                raise ValueError(
                    f"ConvolvePE filter channels ({filt_ch}) must match src channels ({src_ch}), "
                    f"or be mono, or be multi-channel with a mono source."
                )

        # Choose FFT size
        if self._fft_size is None:
            # Default: power-of-two with at least 2048 samples, and enough room for the FIR.
            # This gives low latency for small filters but remains efficient for larger FIRs.
            self._fft_size = _next_pow2(max(2048, filt_len))
        if self._fft_size < filt_len:
            raise ValueError(f"fft_size ({self._fft_size}) must be >= filter length ({filt_len})")

        nfft = int(self._fft_size)

        # Precompute H in frequency domain
        if filt_ch == 1:
            H = np.fft.rfft(h[:, 0], n=nfft)
        else:
            H = np.fft.rfft(h, n=nfft, axis=0)  # shape (bins, channels)

        self._fir_len = filt_len
        self._H = H

        # Initialize tail to zeros
        if filt_len > 1:
            self._tail = np.zeros((filt_len - 1, out_ch), dtype=np.float64)
        else:
            self._tail = np.zeros((0, out_ch), dtype=np.float64)

    def _render(self, start: int, duration: int) -> Snippet:
        self._ensure_filter_prepared()
        assert self._fir_len is not None and self._H is not None and self._tail is not None

        # Handle non-contiguous render: clear history
        if self._last_render_end is None or start != self._last_render_end:
            self._tail[:] = 0.0

        nfft = int(self._fft_size)  # type: ignore[arg-type]
        filt_len = int(self._fir_len)
        tail_len = filt_len - 1
        hop = nfft - tail_len
        if hop < 1:
            raise ValueError(f"fft_size ({nfft}) too small for filter length ({filt_len})")

        # Source input
        x = self._src.render(start, duration).data.astype(np.float64, copy=False)
        if x.ndim != 2:
            raise ValueError(f"ConvolvePE src returned invalid shape {getattr(x, 'shape', None)}")

        src_ch = x.shape[1]

        # Determine output channels
        if self._H.ndim == 1:
            # Mono filter: output matches source channels
            out_ch = src_ch
        else:
            # Multi-channel filter: number of channels dictated by filter
            out_ch = self._H.shape[1]

        # Ensure tail has correct channel count (can differ from src if filter dictates)
        if self._tail.shape[1] != out_ch:
            # This can happen if src channel_count was unknown at prepare time and changed.
            self._tail = np.zeros((tail_len, out_ch), dtype=np.float64)

        y = np.zeros((duration, out_ch), dtype=np.float64)

        out_pos = 0
        in_pos = 0
        while in_pos < duration:
            n = min(hop, duration - in_pos)
            x_seg = x[in_pos:in_pos + n, :]

            # Build FFT input block: [tail, x_seg, zeros]
            x_block = np.zeros((nfft, out_ch), dtype=np.float64)

            # Tail (shape (tail_len,out_ch))
            if tail_len > 0:
                x_block[:tail_len, :] = self._tail

            # Current segment: copy/match/fan-out channels
            if out_ch == src_ch:
                # 1:1 channel mapping
                x_block[tail_len:tail_len + n, :] = x_seg
            elif src_ch == 1:
                # Mono source, multi-channel filter: fan out mono channel
                x_block[tail_len:tail_len + n, :] = np.repeat(x_seg, out_ch, axis=1)
            else:
                # Multi-channel source, fewer/equal output channels (should only happen
                # when filter channels <= src channels and were validated earlier).
                x_block[tail_len:tail_len + n, :] = x_seg[:, :out_ch]

            # FFT -> multiply -> IFFT
            X = np.fft.rfft(x_block, axis=0)
            if self._H.ndim == 1:
                Y = X * self._H.reshape(-1, 1)
            else:
                Y = X * self._H
            y_block = np.fft.irfft(Y, n=nfft, axis=0)

            # Overlap-save: discard first tail_len samples
            y_seg = y_block[tail_len:tail_len + n, :]
            y[out_pos:out_pos + n, :] = y_seg

            # Update tail = last tail_len samples of [tail, x_seg]
            if tail_len > 0:
                if n >= tail_len:
                    self._tail = x_block[tail_len + n - tail_len:tail_len + n, :].copy()
                else:
                    if out_ch == src_ch:
                        new_seg = x_seg
                    elif src_ch == 1:
                        new_seg = np.repeat(x_seg, out_ch, axis=1)
                    else:
                        new_seg = x_seg[:, :out_ch]
                    combined = np.vstack([self._tail, new_seg])
                    self._tail = combined[-tail_len:, :].copy()

            out_pos += n
            in_pos += n

        self._last_render_end = start + duration
        return Snippet(start, y.astype(np.float32))

    def __repr__(self) -> str:
        return (
            f"ConvolvePE(src={self._src.__class__.__name__}, "
            f"fir={self._fir.__class__.__name__}, fft_size={self._fft_size})"
        )

