"""
Global configuration for pygmu2: the sample rate.

The global sample rate is the single source of truth, and must be set
(via set_sample_rate) before any ProcessingElement is constructed.

Errors in pygmu2 are plain exceptions raised where the fact surfaces;
there is no configurable error mode (see DESIGN_PHILOSOPHY.md R3).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

_SAMPLE_RATE = None


def set_sample_rate(rate: int) -> None:
    """Set the global sample rate (Hz). Must be set before PE construction."""
    global _SAMPLE_RATE
    _SAMPLE_RATE = int(rate)


def get_sample_rate() -> int | None:
    """Return the global sample rate, if set."""
    return _SAMPLE_RATE
