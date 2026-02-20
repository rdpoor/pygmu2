"""
MidiInPE - MIDI input source using Mido with a thread-safe queue.

Receives MIDI messages via Mido's open_input(callback=...). The callback runs
in a separate thread and pushes messages into a queue.Queue; render() drains
the queue and invokes an optional user callback (sample_index, message) for
each message.

The MIDI connection is opened in _on_start() and closed in _on_stop().

Output: always 1 channel of zeros. Use the callback to drive other PEs (e.g.
MidiGateStatePE) or external state.

Requires: mido (pip install mido). Optional dependency: rtmidi for best port support.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Callable

import numpy as np

from pygmu2.extent import Extent
from pygmu2.source_pe import SourcePE
from pygmu2.snippet import Snippet

if TYPE_CHECKING:
    import mido
else:
    try:
        import mido
    except ImportError:
        mido = None  # type: ignore[assignment]


# Type for optional callback: (sample_index: int, message: mido.Message) -> None
MidiMessageCallback = Callable[[int, "mido.Message"], None] | None


class MidiInPE(SourcePE):
    """
    Source PE that receives MIDI input via Mido and exposes messages via callback.

    Uses mido.open_input(callback=...) so the callback runs in a different thread.
    Incoming messages are put into a thread-safe queue.Queue. On each render(),
    queued messages are drained and the optional user callback is invoked with
    (start, message) for each message, where start is the start of the current
    render block.

    The MIDI port is opened in _on_start() (called by the renderer before the first
    render) and closed in _on_stop() (called after the last render).

    Output: always 1 channel of zeros. Use the callback to drive synthesis or
    state (e.g. MidiGateStatePE.handle_message).

    Extent: infinite. This PE is impure (stateful, contiguous render only).

    Args:
        port_name: Name of the MIDI input port (e.g. "My Keyboard"). If None,
                   uses the system default input (often the first available).
        callback: Optional (sample_index: int, message: mido.Message) -> None.
                  Invoked for each queued message during render; sample_index
                  is the start of the current render block.
    """

    def __init__(
        self,
        port_name: str | None = None,
        callback: MidiMessageCallback = None,
    ):
        if mido is None:
            raise RuntimeError(
                "MidiInPE requires mido. Install with: pip install mido"
            )
        self._port_name = port_name
        self._callback = callback
        self._message_queue: queue.Queue = queue.Queue()
        self._port: "mido.ports.BaseInput" | None = None

    def _mido_callback(self, msg: "mido.Message") -> None:
        """Called by Mido from its input thread; put message on queue."""
        self._message_queue.put_nowait(msg)

    def _on_start(self) -> None:
        """Open the MIDI input connection with callback."""
        self._port = mido.open_input(
            name=self._port_name, callback=self._mido_callback
        )

    def _on_stop(self) -> None:
        """Close the MIDI input connection."""
        if self._port is not None:
            self._port.close()
            self._port = None

    def _render(self, start: int, duration: int) -> Snippet:
        """Drain queued MIDI messages and invoke callback for each; return 1ch zeros."""
        try:
            while True:
                msg = self._message_queue.get_nowait()
                if self._callback is not None:
                    self._callback(start, msg)
        except queue.Empty:
            pass
        return Snippet(start, np.zeros((duration, 1), dtype=np.float32))

    def _compute_extent(self) -> Extent:
        """Live MIDI source has infinite extent."""
        return Extent(None, None)

    def channel_count(self) -> int:
        return 1

    def is_pure(self) -> bool:
        return False

    def __repr__(self) -> str:
        name = repr(self._port_name) if self._port_name is not None else "default"
        cb = f", callback={self._callback}" if self._callback else ""
        return f"MidiInPE(port_name={name}{cb})"
