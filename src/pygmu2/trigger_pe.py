"""
TriggerPE - trigger-based time-shifted rendering.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from enum import Enum
from typing import List, Optional

import numpy as np

from pygmu2.config import handle_error
from pygmu2.extent import Extent
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet


class TriggerMode(Enum):
    """Trigger behavior modes."""
    ONE_SHOT = "one_shot"  # Start on first trigger, ignore subsequent
    GATED = "gated"        # Start on trigger, stop on release, restart on next trigger


class TriggerPE(ProcessingElement):
    """
    Renders a source PE starting from t=0 when triggered.

    TriggerPE allows time-shifted rendering of a source signal based on a control
    trigger input. This enables "note-like" behavior where a sound (like an envelope)
    starts its lifecycle at an arbitrary time in response to an event.

    Modes:
        ONE_SHOT: Waits for trigger > 0. Once triggered, it starts rendering the
                  source from t=0 and continues indefinitely, ignoring further
                  triggers. Ideally used with finite-duration sources.
        GATED:    Active while trigger > 0. When trigger goes high, source restarts
                  from t=0. When trigger goes low (<= 0), output is silenced.

    Args:
        source: The ProcessingElement to render (the "signal").
        trigger: The control ProcessingElement (trigger signal).
        mode: TriggerMode (default: ONE_SHOT).
    """

    def __init__(
        self,
        source: ProcessingElement,
        trigger: ProcessingElement,
        mode: TriggerMode = TriggerMode.ONE_SHOT,
    ):
        self._source = source
        self._trigger = trigger
        self._mode = mode

        # State
        self._is_active = False
        self._start_time = 0  # Absolute sample time when the current trigger started

    @property
    def source(self) -> ProcessingElement:
        """The signal source."""
        return self._source

    @property
    def trigger(self) -> ProcessingElement:
        """The trigger source."""
        return self._trigger

    @property
    def mode(self) -> TriggerMode:
        """The trigger mode."""
        return self._mode

    def inputs(self) -> List[ProcessingElement]:
        return [self._source, self._trigger]

    def is_pure(self) -> bool:
        """TriggerPE maintains internal state (active status, start time)."""
        return False

    def channel_count(self) -> Optional[int]:
        return self._source.channel_count()

    def _compute_extent(self) -> Extent:
        # The extent is effectively infinite or determined by the trigger,
        # but locally it behaves like the source's extent shifted in time.
        # For simplicity, we report indefinite extent as it depends on runtime triggers.
        return Extent(0, None)

    def _reset_state(self) -> None:
        """Reset trigger state."""
        self._is_active = False
        self._start_time = 0
    
    def on_start(self) -> None:
        """Reset state at start of rendering."""
        self._reset_state()

    def on_stop(self) -> None:
        """Reset state at end of rendering."""
        self._reset_state()

    def _render(self, start: int, duration: int) -> Snippet:
        trigger_snippet = self._trigger.render(start, duration)
        trigger_data = trigger_snippet.data
        
        # If trigger is mono, broadcast if necessary (though we just check > 0)
        # We'll use the first channel of the trigger signal for logic
        trig_signal = trigger_data[:, 0] if trigger_data.shape[1] > 0 else np.zeros(duration)

        channels = self.channel_count() or 1
        output_data = np.zeros((duration, channels), dtype=np.float32)

        # Segment-based processing
        current_idx = 0
        while current_idx < duration:
            # Find next state change
            # If active, look for trigger <= 0 (if GATED)
            # If inactive, look for trigger > 0
            
            remaining = duration - current_idx
            chunk_len = remaining

            if self._mode == TriggerMode.ONE_SHOT:
                if self._is_active:
                    # Already triggered, just render the rest
                    chunk_len = remaining
                    # Render source
                    local_start = (start + current_idx) - self._start_time
                    src = self._source.render(local_start, chunk_len)
                    output_data[current_idx : current_idx + chunk_len] = src.data
                    current_idx += chunk_len
                else:
                    # Not active, look for trigger
                    # Find first index where trig > 0
                    trig_slice = trig_signal[current_idx:]
                    nz_indices = np.where(trig_slice > 0)[0]
                    
                    if len(nz_indices) > 0:
                        # Found a trigger
                        offset = nz_indices[0]
                        # Silence until trigger
                        # (output_data is already zero init)
                        
                        # Update state
                        self._is_active = True
                        self._start_time = start + current_idx + offset
                        
                        current_idx += offset
                        # Next iteration will handle the active state
                    else:
                        # No trigger in this block
                        current_idx += remaining

            elif self._mode == TriggerMode.GATED:
                if self._is_active:
                    # Look for gate close (<= 0)
                    trig_slice = trig_signal[current_idx:]
                    # Find first index where trig <= 0
                    # Note: we assume 'active' means we are currently playing.
                    # Ideally we should also check for re-trigger (low->high) 
                    # but typically gated implies high=on, low=off.
                    # If the signal drops to zero then goes high again in one block,
                    # we need to handle that.
                    
                    off_indices = np.where(trig_slice <= 0)[0]
                    
                    if len(off_indices) > 0:
                        chunk_len = off_indices[0]
                    else:
                        chunk_len = remaining
                    
                    # Render active portion
                    if chunk_len > 0:
                        local_start = (start + current_idx) - self._start_time
                        src = self._source.render(local_start, chunk_len)
                        output_data[current_idx : current_idx + chunk_len] = src.data
                    
                    current_idx += chunk_len
                    
                    if len(off_indices) > 0:
                        # We hit a gate close
                        self._is_active = False
                else:
                    # Look for gate open (> 0)
                    trig_slice = trig_signal[current_idx:]
                    on_indices = np.where(trig_slice > 0)[0]
                    
                    if len(on_indices) > 0:
                        offset = on_indices[0]
                        # Silence until trigger
                        current_idx += offset
                        
                        # Start new note
                        self._is_active = True
                        self._start_time = start + current_idx
                    else:
                        current_idx += remaining

        return Snippet(start, output_data)
