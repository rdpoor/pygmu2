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
    ONE_SHOT = "one_shot"   # When ARMED and trigger > 0, start source; then run indefinitely
    GATED = "gated"         # Like ONE_SHOT but stop when trigger <= 0; does not retrigger (unless _reset_state/_on_start)
    RETRIGGER = "retrigger" # When trigger > 0 start source; when trigger <= 0 output zeros


class TriggerState(Enum):
    """Internal state: armed -> active [-> inactive for GATED only]."""
    ARMED = "armed"       # Waiting for positive edge
    ACTIVE = "active"     # Playing source
    INACTIVE = "inactive" # Gate closed (GATED only); no retrigger until _reset_state/_on_start


class TriggerPE(ProcessingElement):
    """
    Renders a source PE starting from t=0 when triggered.

    TriggerPE allows time-shifted rendering of a source signal based on a control
    trigger input. This enables "note-like" behavior where a sound (like an envelope)
    starts its lifecycle at an arbitrary time in response to an event.

    Trigger is treated as mono (channel 0). The PE enters triggered state when
    the current state is ARMED (idle) and the current trigger sample is positive.
    No previous-sample comparison; state + current sample suffice.

    Modes:
        ONE_SHOT: When ARMED and trigger > 0, start source from t=0 and continue
                  indefinitely (output zeros after source extent if finite).
        GATED:    Like ONE_SHOT, but stop when trigger <= 0. Does not retrigger
                  (unless _reset_state or _on_start is called).
        RETRIGGER: When trigger > 0 start source from t=0; when trigger <= 0 output zeros.

    Non-pure: maintains internal state. Requires contiguous render() calls.

    Args:
        source: The ProcessingElement to render (the "signal").
        trigger: The control ProcessingElement (trigger signal, mono = channel 0).
        trigger_mode: TriggerMode (default: ONE_SHOT).
    """

    def __init__(
        self,
        source: ProcessingElement,
        trigger: ProcessingElement,
        trigger_mode: TriggerMode = TriggerMode.ONE_SHOT,
    ):
        self._source = source
        self._trigger = trigger
        self._mode = trigger_mode

        # State: ARMED -> ACTIVE; in GATED, ACTIVE -> INACTIVE (no retrigger); in RETRIGGER, ACTIVE -> ARMED
        self._state = TriggerState.ARMED
        self._start_time = 0  # Absolute sample time when the current gate/trigger started

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
        """The trigger mode (same as trigger_mode)."""
        return self._mode

    def inputs(self) -> List[ProcessingElement]:
        return [self._source, self._trigger]

    def is_pure(self) -> bool:
        """TriggerPE maintains internal state (active status, start time)."""
        return False

    def channel_count(self) -> Optional[int]:
        return self._source.channel_count()

    def _compute_extent(self) -> Extent:
        # Trigger can fire at any time; output can extend arbitrarily in both directions.
        return Extent(None, None)

    def _reset_state(self) -> None:
        """Reset trigger state."""
        self._state = TriggerState.ARMED
        self._start_time = 0
    
    def _on_start(self) -> None:
        """Reset state at start of rendering."""
        self._reset_state()

    def _on_stop(self) -> None:
        """Reset state at end of rendering."""
        self._reset_state()

    def _render(self, start: int, duration: int) -> Snippet:
        trigger_snippet = self._trigger.render(start, duration)
        trigger_data = trigger_snippet.data
        trig_signal = trigger_data[:, 0] if trigger_data.shape[1] > 0 else np.zeros(duration, dtype=np.float32)

        # Enter triggered when state is ARMED and current sample is positive (no previous-sample check).
        channels = self.channel_count() or 1
        output_data = np.zeros((duration, channels), dtype=np.float32)

        current_idx = 0
        while current_idx < duration:
            remaining = duration - current_idx

            if self._mode == TriggerMode.ONE_SHOT:
                if self._state == TriggerState.ACTIVE:
                    local_start = (start + current_idx) - self._start_time
                    src = self._source.render(local_start, remaining)
                    output_data[current_idx : current_idx + remaining] = src.data
                    current_idx += remaining
                else:  # ARMED: first positive sample triggers
                    found = None
                    for j in range(current_idx, duration):
                        if trig_signal[j] > 0:
                            found = j
                            break
                    if found is not None:
                        self._state = TriggerState.ACTIVE
                        self._start_time = start + found
                        current_idx = found
                    else:
                        current_idx += remaining

            elif self._mode == TriggerMode.GATED:
                if self._state == TriggerState.ACTIVE:
                    off_idx = None
                    for j in range(current_idx, duration):
                        if trig_signal[j] <= 0:
                            off_idx = j
                            break
                    chunk_end = off_idx if off_idx is not None else duration
                    chunk_len = chunk_end - current_idx
                    if chunk_len > 0:
                        local_start = (start + current_idx) - self._start_time
                        src = self._source.render(local_start, chunk_len)
                        output_data[current_idx : current_idx + chunk_len] = src.data
                    current_idx = chunk_end
                    if off_idx is not None:
                        self._state = TriggerState.INACTIVE
                elif self._state == TriggerState.INACTIVE:
                    current_idx += remaining
                else:  # ARMED: first positive sample triggers
                    found = None
                    for j in range(current_idx, duration):
                        if trig_signal[j] > 0:
                            found = j
                            break
                    if found is not None:
                        self._state = TriggerState.ACTIVE
                        self._start_time = start + found
                        current_idx = found
                    else:
                        current_idx += remaining

            else:  # RETRIGGER: each positive sample starts a segment (when not in a segment, first >0 triggers)
                edge_at = None
                for j in range(current_idx, duration):
                    if trig_signal[j] > 0:
                        edge_at = j
                        break
                if edge_at is None:
                    current_idx += remaining
                    continue
                current_idx = edge_at
                off_idx = None
                for j in range(edge_at, duration):
                    if trig_signal[j] <= 0:
                        off_idx = j
                        break
                gate_end = off_idx if off_idx is not None else duration
                gate_len = gate_end - edge_at
                if gate_len > 0:
                    src = self._source.render(0, gate_len)
                    output_data[edge_at : edge_at + gate_len] = src.data
                current_idx = gate_end

        return Snippet(start, output_data)
