"""
Package-level assets (e.g. KEMAR HRTFs) for pygmu2.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path


def get_kemar_dir() -> Path:
    """Return the path to the KEMAR HRTF WAV directory (MIT compact set).

    When the package is installed, this points to pygmu2/assets/kemar/.
    Use with SpatialHRTF: load the filename returned by
    SpatialHRTF.hrtf_filename_for(azimuth, elevation) from this directory.
    """
    return Path(__file__).parent / "kemar"
