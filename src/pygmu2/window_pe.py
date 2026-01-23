"""
WindowPE - bidirectional windowed statistics (envelope, smoothing).

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

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
        env = WindowPE(source, window=0.05, mode=WindowMode.MAX)
        
        # RMS envelope
        env = WindowPE(source, window=0.02, mode=WindowMode.RMS)
        
        # Smoothing a control signal (without rectification)
        smooth = WindowPE(control, window=0.01, mode=WindowMode.MEAN, rectify=False)
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
    
    def render(self, start: int, duration: int) -> Snippet:
        """
        Render windowed statistics.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render
        
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
        """Compute windowed maximum."""
        channels = x.shape[1]
        output = np.zeros((duration, channels), dtype=np.float64)
        
        for n in range(duration):
            # Window is centered at position (n + half_window) in the fetched data
            center = n + half_window
            win_start = center - half_window
            win_end = center + half_window + 1
            
            for ch in range(channels):
                output[n, ch] = np.max(x[win_start:win_end, ch])
        
        return output
    
    def _compute_windowed_min(
        self, x: np.ndarray, duration: int, half_window: int
    ) -> np.ndarray:
        """Compute windowed minimum."""
        channels = x.shape[1]
        output = np.zeros((duration, channels), dtype=np.float64)
        
        for n in range(duration):
            center = n + half_window
            win_start = center - half_window
            win_end = center + half_window + 1
            
            for ch in range(channels):
                output[n, ch] = np.min(x[win_start:win_end, ch])
        
        return output
    
    def _compute_windowed_mean(
        self, x: np.ndarray, duration: int, half_window: int
    ) -> np.ndarray:
        """Compute windowed mean using cumulative sum for efficiency."""
        channels = x.shape[1]
        output = np.zeros((duration, channels), dtype=np.float64)
        window_size = 2 * half_window + 1
        
        for ch in range(channels):
            # Cumulative sum for O(1) window sum computation
            cumsum = np.cumsum(np.concatenate([[0], x[:, ch]]))
            
            for n in range(duration):
                center = n + half_window
                win_start = center - half_window
                win_end = center + half_window + 1
                
                output[n, ch] = (cumsum[win_end] - cumsum[win_start]) / window_size
        
        return output
    
    def _compute_windowed_rms(
        self, x: np.ndarray, duration: int, half_window: int
    ) -> np.ndarray:
        """Compute windowed RMS using cumulative sum for efficiency."""
        channels = x.shape[1]
        output = np.zeros((duration, channels), dtype=np.float64)
        window_size = 2 * half_window + 1
        
        # Square the signal
        x_squared = x ** 2
        
        for ch in range(channels):
            # Cumulative sum of squared values
            cumsum = np.cumsum(np.concatenate([[0], x_squared[:, ch]]))
            
            for n in range(duration):
                center = n + half_window
                win_start = center - half_window
                win_end = center + half_window + 1
                
                mean_squared = (cumsum[win_end] - cumsum[win_start]) / window_size
                output[n, ch] = np.sqrt(mean_squared)
        
        return output
    
    def __repr__(self) -> str:
        return (
            f"WindowPE(source={self._source.__class__.__name__}, "
            f"window={self._window}, mode={self._mode.value}, "
            f"rectify={self._rectify})"
        )
