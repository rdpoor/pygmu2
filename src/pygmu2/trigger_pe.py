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
    RETRIGGER = "retrigger" # Like GATED but retriggers when gate goes high again after low


class TriggerState(Enum):
    """Internal state: armed -> active [-> inactive for GATED/RETRIGGER]."""
    ARMED = "armed"       # Waiting for positive edge
    ACTIVE = "active"     # Playing source
    INACTIVE = "inactive" # Gate closed; GATED stays here until reset; RETRIGGER retriggers on next gate high


class TriggerPE(ProcessingElement):
    """
    Renders a source PE starting from t=0 when triggered.

    TriggerPE allows time-shifted rendering of a source signal based on a control
    trigger input. This enables "note-like" behavior where a sound (like an envelope)
    starts its lifecycle at an arbitrary time in response to an event.

    Trigger is treated as mono (channel 0). The PE enters triggered state when
    the current state is ARMED (idle) and the current trigger sample is positive.
    No previous-sample comparison; state + current sample suffice.

    Each time the PE enters ACTIVE state (first trigger or retrigger), it calls
    source.reset_state() so the source plays from the beginning.

    Modes:
        ONE_SHOT: When ARMED and trigger > 0, start source from t=0 and continue
                  indefinitely (output zeros after source extent if finite).
        GATED:    Like ONE_SHOT, but stop when trigger <= 0. Does not retrigger
                  when gate goes high again (stay INACTIVE until _reset_state/_on_start).
        RETRIGGER: Like GATED (continuous playback while gate high, stop when low),
                   but retriggers from t=0 when gate goes high again after being low.

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

        # State: ARMED -> ACTIVE; GATED/RETRIGGER: ACTIVE -> INACTIVE on gate low.
        # GATED: INACTIVE stays until reset. RETRIGGER: INACTIVE -> ACTIVE on gate high (retrigger).
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

    @staticmethod
    def _first_positive(trig: np.ndarray, start: int, end: int) -> Optional[int]:
        """First index in [start, end) where trig[i] > 0, or None."""
        segment = trig[start:end]
        pos = np.flatnonzero(segment > 0)
        return start + int(pos[0]) if len(pos) > 0 else None

    @staticmethod
    def _first_non_positive(trig: np.ndarray, start: int, end: int) -> Optional[int]:
        """First index in [start, end) where trig[i] <= 0, or None."""
        segment = trig[start:end]
        pos = np.flatnonzero(segment <= 0)
        return start + int(pos[0]) if len(pos) > 0 else None

    def _render(self, start: int, duration: int) -> Snippet:
        trigger_snippet = self._trigger.render(start, duration)
        trigger_data = trigger_snippet.data
        trig_signal = trigger_data[:, 0] if trigger_data.shape[1] > 0 else np.zeros(duration, dtype=np.float32)

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
                    found = self._first_positive(trig_signal, current_idx, duration)
                    if found is not None:
                        self._state = TriggerState.ACTIVE
                        self._start_time = start + found
                        self._source.reset_state()
                        current_idx = found
                    else:
                        current_idx += remaining

            elif self._mode == TriggerMode.GATED:
                if self._state == TriggerState.ACTIVE:
                    off_idx = self._first_non_positive(trig_signal, current_idx, duration)
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
                else:  # ARMED
                    found = self._first_positive(trig_signal, current_idx, duration)
                    if found is not None:
                        self._state = TriggerState.ACTIVE
                        self._start_time = start + found
                        self._source.reset_state()
                        current_idx = found
                    else:
                        current_idx += remaining

            else:  # RETRIGGER
                if self._state == TriggerState.ACTIVE:
                    off_idx = self._first_non_positive(trig_signal, current_idx, duration)
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
                    found = self._first_positive(trig_signal, current_idx, duration)
                    if found is not None:
                        self._state = TriggerState.ACTIVE
                        self._start_time = start + found
                        self._source.reset_state()
                        current_idx = found
                    else:
                        current_idx += remaining
                else:  # ARMED
                    found = self._first_positive(trig_signal, current_idx, duration)
                    if found is not None:
                        self._state = TriggerState.ACTIVE
                        self._start_time = start + found
                        self._source.reset_state()
                        current_idx = found
                    else:
                        current_idx += remaining

        return Snippet(start, output_data)
