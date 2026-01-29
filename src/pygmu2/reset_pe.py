"""
ResetPE - resets source state on trigger rising edge.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from typing import Optional

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class ResetPE(ProcessingElement):
    """
    Resets the state of a source PE when a trigger signal rises and restarts
    the source's time at 0.
    
    Detects rising edges (trigger transitions from <= 0 to > 0) and at each edge:
    1. Calls reset_state() on the source PE to reset its internal state
    2. Renders the source from time 0 (making it think it's starting fresh)
    
    This enables analog-like behavior where oscillators restart on each gate/trigger,
    ensuring that each "note on" event produces the same output regardless of when
    it occurs. The source's output is passed through unchanged - this PE only affects
    the source's internal state and time reference, not the audio signal itself.
    
    Args:
        source: The PE whose state should be reset
        trigger: Control signal indicating when to reset (rising edge detection)
    
    Example:
        # Reset oscillator on each gate
        gate_stream = SequencePE([(make_gate(1.0), 0), (make_gate(1.0), 44100)])
        osc_stream = SuperSawPE(frequency=440.0)
        reset_osc_stream = ResetPE(osc_stream, gate_stream)
        adsr_stream = AdsrPE(gate_stream)
        output_stream = GainPE(reset_osc_stream, adsr_stream)
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        trigger: ProcessingElement,
    ):
        self._source = source
        self._trigger = trigger
        
        # State for edge detection
        self._prev_trigger: float = 0.0
        self._last_render_end: Optional[int] = None
    
    @property
    def source(self) -> ProcessingElement:
        """The source PE whose state will be reset."""
        return self._source
    
    @property
    def trigger(self) -> ProcessingElement:
        """The trigger signal."""
        return self._trigger
    
    def inputs(self) -> list[ProcessingElement]:
        """Return the source and trigger inputs."""
        return [self._source, self._trigger]
    
    def is_pure(self) -> bool:
        """ResetPE is not pure - it maintains trigger state."""
        return False
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def _compute_extent(self) -> Extent:
        """Return the extent of the source."""
        return self._source.extent()
    
    def on_start(self) -> None:
        """Reset trigger state at start of rendering."""
        self._prev_trigger = 0.0
        self._last_render_end = None
    
    def on_stop(self) -> None:
        """Reset trigger state at end of rendering."""
        self._prev_trigger = 0.0
        self._last_render_end = None
    
    def _reset_state(self) -> None:
        """Reset this PE's state (not the source's)."""
        self._prev_trigger = 0.0
        self._last_render_end = None
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render source signal, resetting source state on trigger rising edges.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet containing the source's output (unchanged)
        """
        # Get trigger signal (use first channel)
        trigger_snippet = self._trigger.render(start, duration)
        trigger_signal = trigger_snippet.data[:, 0] if trigger_snippet.data.shape[1] > 0 else np.zeros(duration)
        
        # Check for gap in rendering (non-continuous chunks)
        # If there's a gap, we can't know the trigger value at start-1,
        # so we assume it was 0 (conservative - might miss an edge, but won't false-trigger)
        if self._last_render_end is not None and start != self._last_render_end:
            # Gap detected - assume trigger was 0 before gap
            self._prev_trigger = 0.0
        
        # Note: Even for pure sources, rendering from time 0 vs absolute time produces
        # different output (e.g., SinePE phase depends on start time). So we need
        # segmented rendering for all sources to restart them at each trigger.
        
        # Non-pure sources: need segmented rendering with resets
        # Vectorized rising edge detection
        # Create array of previous values: [prev_trigger, trigger[0], trigger[1], ..., trigger[n-2]]
        prev_array = np.concatenate([[self._prev_trigger], trigger_signal[:-1]])
        
        # Detect all rising edges at once: current > 0 and previous <= 0
        rising_edge_indices = np.where((trigger_signal > 0.0) & (prev_array <= 0.0))[0]
        
        # Initialize output
        channels = self.channel_count() or 1
        output_data = np.zeros((duration, channels), dtype=np.float32)
        
        # Segment-based rendering: reset and render at each rising edge
        # At each rising edge, reset source and render from that point to next edge (or end)
        current_idx = 0
        
        for i, edge_idx in enumerate(rising_edge_indices):
            # Render segment before this edge (no reset, source continues from previous state)
            if edge_idx > current_idx:
                segment_len = edge_idx - current_idx
                segment_snippet = self._source.render(start + current_idx, segment_len)
                output_data[current_idx:current_idx + segment_len] = segment_snippet.data
                current_idx = edge_idx
            
            # Reset source state at the rising edge
            self._source.reset_state()
            
            # Determine segment length: from this edge to next edge (or end of buffer)
            if i < len(rising_edge_indices) - 1:
                # There's another edge after this one
                next_edge = rising_edge_indices[i + 1]
                segment_len = next_edge - current_idx
            else:
                # This is the last edge - render to end of buffer
                segment_len = duration - current_idx
            
            if segment_len > 0:
                # Render from time 0 (source thinks it's starting fresh after reset)
                # but place output at the correct absolute position
                segment_snippet = self._source.render(0, segment_len)
                output_data[current_idx:current_idx + segment_len] = segment_snippet.data
                current_idx += segment_len
        
        # Render any remaining segment after last edge (no reset)
        if current_idx < duration:
            segment_len = duration - current_idx
            segment_snippet = self._source.render(start + current_idx, segment_len)
            output_data[current_idx:current_idx + segment_len] = segment_snippet.data
        
        # Update prev_trigger for next chunk (last value in current chunk)
        self._prev_trigger = trigger_signal[-1] if duration > 0 else self._prev_trigger
        
        # Update last render end
        self._last_render_end = start + duration
        
        return Snippet(start, output_data)
    
    def __repr__(self) -> str:
        return f"ResetPE(source={self._source.__class__.__name__}, trigger={self._trigger.__class__.__name__})"
