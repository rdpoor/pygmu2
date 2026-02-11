"""
ControlPE - a source PE whose value can be set from an external thread.

Like ConstantPE, but with a thread-safe mutable value.  The GUI (or any
external thread) calls ``set_value(v)`` which pushes *v* onto a
``queue.Queue``.  During ``_render()`` the queue is drained and the latest
value becomes the output.

Threading pattern follows MidiInPE: producer calls ``set_value()`` from any
thread; consumer (the audio callback thread) drains in ``_render()``.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import queue

import numpy as np

from pygmu2.extent import Extent
from pygmu2.processing_element import SourcePE
from pygmu2.snippet import Snippet


class ControlPE(SourcePE):
    """
    A SourcePE whose output value can be changed at any time from any thread.

    Args:
        initial_value: Starting output value (default: 0.0)
        channels: Number of output channels (default: 1)

    Example:
        rate_control = ControlPE(initial_value=1.0)
        warped = TimeWarpPE(source, rate=rate_control)

        # From the GUI thread:
        rate_control.set_value(2.0)  # double speed
    """

    def __init__(self, initial_value: float = 0.0, channels: int = 1):
        self._value = initial_value
        self._channels = channels
        self._queue: queue.Queue[float] = queue.Queue()

    def set_value(self, value: float) -> None:
        """Thread-safe: push a new value from any thread (GUI, MIDI, etc.)."""
        self._queue.put_nowait(value)

    @property
    def value(self) -> float:
        """The current output value (last value consumed by render)."""
        return self._value

    def _render(self, start: int, duration: int) -> Snippet:
        # Drain queue, keep latest value (same pattern as MidiInPE)
        try:
            while True:
                self._value = self._queue.get_nowait()
        except queue.Empty:
            pass
        data = np.full((duration, self._channels), self._value, dtype=np.float32)
        return Snippet(start, data)

    def _compute_extent(self) -> Extent:
        """Infinite extent, like ConstantPE."""
        return Extent(None, None)

    def is_pure(self) -> bool:
        return False  # value can change between renders

    def channel_count(self) -> int:
        return self._channels

    def __repr__(self) -> str:
        return f"ControlPE(value={self._value}, channels={self._channels})"
