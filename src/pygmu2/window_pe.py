"""
WindowPE - bidirectional windowed statistics (envelope, smoothing).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from enum import Enum
from typing import Optional

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class WindowMode(Enum):
    """Window statistic modes."""
    MAX = "max"    # Maximum absolute value in window
    MEAN = "mean"  # Mean of absolute values in window
    RMS = "rms"    # Root mean square in window
    MIN = "min"    # Minimum absolute value in window


class WindowPE(ProcessingElement):
    """
    Bidirectional windowed statistics - computes statistics over a symmetric
    window centered on each sample.
    
    Unlike causal envelope followers, this PE looks both backward and forward
    in time, producing a zero-phase (no latency) result. This is possible
    because pygmu2 operates offline rather than in real-time.
    
    The window is centered on each output sample, looking half the window
    duration into the past and half into the future.
    
    Args:
        source: Input audio PE
        window: Window duration in seconds (total width)
        mode: Statistic to compute (MAX, MEAN, RMS, MIN)
        rectify: Whether to take absolute value first (default True)
    
    Example:
        # Symmetric envelope (no attack/release lag)
        env_stream = WindowPE(source_stream, window=0.05, mode=WindowMode.MAX)
        
        # RMS envelope
        env_stream = WindowPE(source_stream, window=0.02, mode=WindowMode.RMS)
        
        # Smoothing a control signal (without rectification)
        smooth_stream = WindowPE(control_stream, window=0.01, mode=WindowMode.MEAN, rectify=False)
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        window: float = 0.05,
        mode: WindowMode = WindowMode.MAX,
        rectify: bool = True,
    ):
        self._source = source
        self._window = max(0.0, window)
        self._mode = mode
        self._rectify = rectify
    
    @property
    def source(self) -> ProcessingElement:
        """The input audio PE."""
        return self._source
    
    @property
    def window(self) -> float:
        """Window duration in seconds."""
        return self._window
    
    @property
    def mode(self) -> WindowMode:
        """Window statistic mode."""
        return self._mode
    
    @property
    def rectify(self) -> bool:
        """Whether to rectify (absolute value) the input."""
        return self._rectify
    
    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs."""
        return [self._source]
    
    def is_pure(self) -> bool:
        """
        WindowPE is pure - it has no state between render calls.
        Each output sample depends only on the input window around it.
        """
        return True
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def _compute_extent(self) -> Extent:
        """Return the extent of this PE (matches source)."""
        return self._source.extent()
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render windowed statistics.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)
        
        Returns:
            Snippet containing windowed statistic values
        """
        sample_rate = self.sample_rate
        
        # Calculate window size in samples (ensure at least 1)
        half_window = max(1, int(self._window * sample_rate / 2))
        
        # Fetch source data with padding for the window
        # We need samples from (start - half_window) to (start + duration + half_window)
        fetch_start = start - half_window
        fetch_duration = duration + 2 * half_window
        
        source_snippet = self._source.render(fetch_start, fetch_duration)
        x = source_snippet.data.astype(np.float64)
        
        # Optionally rectify
        if self._rectify:
            x = np.abs(x)
        
        channels = x.shape[1]
        output = np.zeros((duration, channels), dtype=np.float64)
        
        # Compute windowed statistic for each output sample
        if self._mode == WindowMode.MAX:
            output = self._compute_windowed_max(x, duration, half_window)
        elif self._mode == WindowMode.MIN:
            output = self._compute_windowed_min(x, duration, half_window)
        elif self._mode == WindowMode.MEAN:
            output = self._compute_windowed_mean(x, duration, half_window)
        elif self._mode == WindowMode.RMS:
            output = self._compute_windowed_rms(x, duration, half_window)
        
        return Snippet(start, output.astype(np.float32))
    
    def _compute_windowed_max(
        self, x: np.ndarray, duration: int, half_window: int
    ) -> np.ndarray:
        """
        Compute windowed maximum using vectorized operations.
        
        Uses scipy.ndimage.maximum_filter1d if available (O(n) algorithm),
        otherwise falls back to numpy sliding_window_view.
        """
        window_size = 2 * half_window + 1
        channels = x.shape[1]
        output = np.zeros((duration, channels), dtype=np.float64)
        
        try:
            from scipy.ndimage import maximum_filter1d
            # scipy's maximum_filter1d uses an efficient O(n) algorithm
            for ch in range(channels):
                filtered = maximum_filter1d(x[:, ch], size=window_size, mode='nearest')
                # Extract the centered portion
                output[:, ch] = filtered[half_window:half_window + duration]
        except ImportError:
            # Fallback: use numpy sliding_window_view (numpy >= 1.20)
            for ch in range(channels):
                windows = np.lib.stride_tricks.sliding_window_view(x[:, ch], window_size)
                output[:, ch] = np.max(windows[:duration], axis=1)
        
        return output
    
    def _compute_windowed_min(
        self, x: np.ndarray, duration: int, half_window: int
    ) -> np.ndarray:
        """
        Compute windowed minimum using vectorized operations.
        
        Uses scipy.ndimage.minimum_filter1d if available (O(n) algorithm),
        otherwise falls back to numpy sliding_window_view.
        """
        window_size = 2 * half_window + 1
        channels = x.shape[1]
        output = np.zeros((duration, channels), dtype=np.float64)
        
        try:
            from scipy.ndimage import minimum_filter1d
            for ch in range(channels):
                filtered = minimum_filter1d(x[:, ch], size=window_size, mode='nearest')
                output[:, ch] = filtered[half_window:half_window + duration]
        except ImportError:
            for ch in range(channels):
                windows = np.lib.stride_tricks.sliding_window_view(x[:, ch], window_size)
                output[:, ch] = np.min(windows[:duration], axis=1)
        
        return output
    
    def _compute_windowed_mean(
        self, x: np.ndarray, duration: int, half_window: int
    ) -> np.ndarray:
        """
        Compute windowed mean using vectorized cumulative sum.
        
        Fully vectorized O(n) implementation - no Python loops over samples.
        """
        window_size = 2 * half_window + 1
        channels = x.shape[1]
        output = np.zeros((duration, channels), dtype=np.float64)
        
        for ch in range(channels):
            # Cumulative sum with leading zero for easy differencing
            cumsum = np.cumsum(np.concatenate([[0], x[:, ch]]))
            
            # Vectorized window sum: cumsum[end] - cumsum[start] for all positions
            # Window for output[n] spans [n, n + window_size) in cumsum indices
            starts = np.arange(duration)
            ends = starts + window_size
            output[:, ch] = (cumsum[ends] - cumsum[starts]) / window_size
        
        return output
    
    def _compute_windowed_rms(
        self, x: np.ndarray, duration: int, half_window: int
    ) -> np.ndarray:
        """
        Compute windowed RMS using vectorized cumulative sum.
        
        Fully vectorized O(n) implementation.
        """
        window_size = 2 * half_window + 1
        channels = x.shape[1]
        output = np.zeros((duration, channels), dtype=np.float64)
        
        # Square the signal once
        x_squared = x ** 2
        
        for ch in range(channels):
            # Cumulative sum of squared values
            cumsum = np.cumsum(np.concatenate([[0], x_squared[:, ch]]))
            
            # Vectorized: compute all window sums at once
            starts = np.arange(duration)
            ends = starts + window_size
            mean_squared = (cumsum[ends] - cumsum[starts]) / window_size
            output[:, ch] = np.sqrt(mean_squared)
        
        return output
    
    def __repr__(self) -> str:
        return (
            f"WindowPE(source={self._source.__class__.__name__}, "
            f"window={self._window}, mode={self._mode.value}, "
            f"rectify={self._rectify})"
        )
