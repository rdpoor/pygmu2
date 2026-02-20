"""
SourcePE abstract base class.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from abc import abstractmethod

from pygmu2.processing_element import ProcessingElement


class SourcePE(ProcessingElement):
    """
    Abstract base class for source ProcessingElements (no inputs).

    Sources generate audio from external data (files, synthesis, etc.)
    rather than processing input from other PEs.

    Sources are typically pure (arbitrary render times, multi-sink OK) and must
    declare their output channel count explicitly.
    """

    def inputs(self) -> list[ProcessingElement]:
        """Sources have no inputs."""
        return []

    def is_pure(self) -> bool:
        """
        Sources are typically pure (arbitrary render times, multi-sink OK).

        Override and return False for sources with state that require
        contiguous render requests.
        """
        return True

    def required_input_channels(self) -> int | None:
        """Not applicable for sources."""
        return None

    @abstractmethod
    def channel_count(self) -> int:
        """
        Sources MUST declare their output channel count.

        Returns:
            Number of output channels (must be concrete int, not None)
        """
        pass
