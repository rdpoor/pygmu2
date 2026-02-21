"""
AudioReaderPE - decode and play back compressed audio files.

Supports any format that miniaudio can decode: MP3, FLAC, OGG Vorbis, and WAV.
The entire file is decoded into memory on _on_start() and resampled to the
system sample rate if the file's native rate differs. Subsequent render()
calls are served from the in-memory buffer with no further I/O.

Requires the miniaudio package:
    pip install miniaudio

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.source_pe import SourcePE
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.logger import get_logger

logger = get_logger(__name__)


def _import_miniaudio():
    try:
        import miniaudio
        return miniaudio
    except ImportError as e:
        raise ImportError(
            "miniaudio is required for AudioReaderPE. "
            "Install it with: pip install miniaudio"
        ) from e


class AudioReaderPE(SourcePE):
    """
    A SourcePE that decodes a compressed audio file (MP3, FLAC, OGG, WAV)
    into memory and serves samples on demand.

    The file is decoded once on _on_start() and resampled to the system
    sample rate if necessary. The decoded buffer is released on _on_stop().

    Args:
        path: Path to the audio file.
        max_level_db: If given, peak-normalize the decoded audio so the loudest
            sample equals this level in dBFS (e.g. -1.0 leaves 1 dB of
            headroom). None (default) leaves the decoded samples unchanged.

    Notes:
        - is_pure() is True: the in-memory buffer supports arbitrary
          (start, duration) requests in any order and by multiple sinks.
        - Requires the miniaudio package: pip install miniaudio.
        - To shift playback position in time, use DelayPE.
    """

    def __init__(self, path: str, max_level_db: float | None = None):
        self._path = path
        self._max_level_db = max_level_db
        self._file_info = None          # populated lazily by _ensure_file_info()
        self._data: np.ndarray | None = None  # (frames, channels) float32

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        """Path to the audio file."""
        return self._path

    @property
    def file_sample_rate(self) -> int:
        """Native sample rate of the file (reads metadata if needed)."""
        self._ensure_file_info()
        return self._file_info.sample_rate

    # ------------------------------------------------------------------
    # SourcePE / ProcessingElement interface
    # ------------------------------------------------------------------

    def is_pure(self) -> bool:
        return True

    def channel_count(self) -> int:
        self._ensure_file_info()
        return self._file_info.nchannels

    def _compute_extent(self) -> Extent:
        self._ensure_file_info()
        sr = self._sample_rate or self._file_info.sample_rate
        frames = round(self._file_info.num_frames * sr / self._file_info.sample_rate)
        return Extent(0, frames)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        miniaudio = _import_miniaudio()
        self._ensure_file_info()
        decoded = miniaudio.decode_file(
            self._path,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=self._file_info.nchannels,
            sample_rate=self._sample_rate,
        )
        samples = np.frombuffer(decoded.samples, dtype=np.float32).copy()
        self._data = samples.reshape(-1, decoded.nchannels)
        if self._max_level_db is not None:
            peak = np.max(np.abs(self._data))
            if peak > 0.0:
                target = 10.0 ** (self._max_level_db / 20.0)
                self._data *= target / peak
        logger.info(
            f"Decoded {self._path}: {self._data.shape[0]} frames, "
            f"{self._data.shape[1]} ch, {self._sample_rate} Hz"
        )

    def _on_stop(self) -> None:
        self._data = None
        logger.info(f"Released {self._path}")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _ensure_data(self) -> None:
        """Decode the file if not already done (supports render without Renderer)."""
        if self._data is None:
            self._on_start()

    def _render(self, start: int, duration: int) -> Snippet:
        self._ensure_data()
        n, ch = self._data.shape
        out = np.zeros((duration, ch), dtype=np.float32)
        overlap_start = max(start, 0)
        overlap_end = min(start + duration, n)
        if overlap_start < overlap_end:
            out_start = overlap_start - start
            chunk = self._data[overlap_start:overlap_end]
            out[out_start : out_start + len(chunk)] = chunk
        return Snippet(start, out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_file_info(self) -> None:
        if self._file_info is None:
            miniaudio = _import_miniaudio()
            self._file_info = miniaudio.get_file_info(self._path)

    def __repr__(self) -> str:
        if self._max_level_db is not None:
            return f"AudioReaderPE(path={self._path!r}, max_level_db={self._max_level_db})"
        return f"AudioReaderPE(path={self._path!r})"
