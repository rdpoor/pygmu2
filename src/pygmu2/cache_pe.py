"""
CachePE - memoize the most recent render from a source PE.

CachePE is a single-entry cache for (start, duration) render requests.
If the same request is repeated, the cached Snippet is returned without
re-rendering the source. This is useful for avoiding redundant pulls in
composed graphs (e.g., dry/wet splits in ReverbPE).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

from typing import Optional

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class CachePE(ProcessingElement):
    """
    Single-entry render cache for a source PE.

    Args:
        source: Input ProcessingElement to cache.

    Notes:
        - CachePE is impure (stateful cache).
        - Only identical (start, duration) requests are cached.
        - Cache is cleared on on_start(), on_stop(), and reset_state().
    """

    def __init__(self, source: ProcessingElement):
        self._source = source
        self._last_start: Optional[int] = None
        self._last_duration: Optional[int] = None
        self._last_snippet: Optional[Snippet] = None

    @property
    def source(self) -> ProcessingElement:
        return self._source

    def inputs(self) -> list[ProcessingElement]:
        return [self._source]

    def is_pure(self) -> bool:
        # CachePE does not change the signal; it memoizes renders.
        # It is safe to treat as pure to allow multiple sinks.
        return True

    def channel_count(self) -> Optional[int]:
        return self._source.channel_count()

    def _compute_extent(self) -> Extent:
        return self._source.extent()

    def _reset_state(self) -> None:
        self._last_start = None
        self._last_duration = None
        self._last_snippet = None

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._reset_state()

    def _render(self, start: int, duration: int) -> Snippet:
        if (
            self._last_snippet is not None
            and self._last_start == start
            and self._last_duration == duration
        ):
            return self._last_snippet

        snippet = self._source.render(start, duration)
        self._last_start = start
        self._last_duration = duration
        self._last_snippet = snippet
        return snippet

    def __repr__(self) -> str:
        return f"CachePE(source={self._source.__class__.__name__})"
