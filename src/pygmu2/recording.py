"""
Recording — sample-stamped audio captured by a DuplexRenderer.

A Recording accumulates the input blocks a duplex callback delivers, each
stamped with the render position of the output block from the same
callback. Because input and output share one device clock (a single
full-duplex stream), the captured audio is sample-accurate relative to the
render timeline up to a constant offset (the round-trip latency, which
DuplexRenderer.calibrate() can measure exactly).

Raw callback-aligned stamps are the ground truth; a measured calibration
offset (if the session was calibrated) is carried on the Recording and
applied by as_pe(), so a calibrated take lands sample-exact on the
timeline it was performed against.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.extent import Extent
from pygmu2.processing_element import ProcessingElement


class Recording:
    """
    Audio captured against the render timeline.

    Blocks arrive from the duplex callback in order, each stamped with the
    render position of the same callback's output block — so the recording
    is contiguous by construction and lives on the same sample timeline as
    the graph that was playing.

    Attributes:
        sample_rate: Session sample rate (Hz).
        channels: Input channel count.
        calibration: Measured round-trip offset in samples (set by
            DuplexRenderer when the session was calibrated), or None.
        status_events: [(block_start, status_string)] for every callback
            that reported a PortAudio status flag (overflow/underflow).
            An input overflow means the driver dropped samples that cannot
            be recovered; the event marks the region.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        calibration: int | None = None,
    ):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.calibration = calibration
        self.status_events: list[tuple[int, str]] = []
        self._blocks: list[np.ndarray] = []
        self._start: int | None = None  # stamp of the first block
        self._data: np.ndarray | None = None  # concatenation cache

    # ------------------------------------------------------------------
    # Filling (called from the duplex callback)
    # ------------------------------------------------------------------

    def _append(self, start: int, block: np.ndarray, status: object) -> None:
        """Append one callback's input block, stamped with the render
        position of the same callback's output block."""
        if self._start is None:
            self._start = int(start)
        self._blocks.append(block)
        self._data = None
        if status:
            self.status_events.append((int(start), str(status)))

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def start(self) -> int:
        """Render-timeline position of the first captured sample (raw,
        callback-aligned)."""
        if self._start is None:
            raise RuntimeError("Recording is empty — nothing was captured.")
        return self._start

    @property
    def data(self) -> np.ndarray:
        """All captured audio as one float32 array of shape (N, channels)."""
        if self._data is None:
            if not self._blocks:
                self._data = np.zeros((0, self.channels), dtype=np.float32)
            else:
                self._data = np.concatenate(self._blocks, axis=0).astype(
                    np.float32, copy=False
                )
        return self._data

    @property
    def duration(self) -> int:
        """Number of captured samples."""
        return int(self.data.shape[0])

    def as_pe(self, shift: int | None = None) -> ProcessingElement:
        """
        The recording as timeline-positioned PE material, ready to mix.

        The material is placed at ``start - offset`` where ``offset`` is
        the measured calibration (if the session was calibrated) — so a
        calibrated overdub lands exactly where it was performed against
        the playback. Pass ``shift`` to override: ``shift=0`` gives the
        raw callback-aligned placement.

        Returns a stateless graph (ArrayPE -> DelayPE): seekable and
        shareable like any other finite source.
        """
        from pygmu2.array_pe import ArrayPE
        from pygmu2.delay_pe import DelayPE

        offset = shift if shift is not None else (self.calibration or 0)
        return DelayPE(ArrayPE(self.data), self.start - offset)

    def save(self, path: str) -> None:
        """Write the captured audio to a sound file (format from suffix).
        WAV files are written as 32-bit float so the capture is lossless."""
        import soundfile as sf

        subtype = "FLOAT" if str(path).lower().endswith(".wav") else None
        sf.write(path, self.data, self.sample_rate, subtype=subtype)

    def summary(self) -> str:
        """Human-readable description, including any dropout events."""
        cal = (
            f"calibration={self.calibration} samples "
            f"({1000.0 * self.calibration / self.sample_rate:.1f} ms)"
            if self.calibration is not None
            else "uncalibrated (raw callback-aligned stamps)"
        )
        lines = [
            f"Recording: {self.duration} samples "
            f"({self.duration / self.sample_rate:.2f} s), "
            f"{self.channels} ch @ {self.sample_rate} Hz, "
            f"start={self._start}, {cal}",
        ]
        if self.status_events:
            lines.append(
                f"  {len(self.status_events)} stream status event(s) — "
                "captured audio may have gaps at:"
            )
            for pos, status in self.status_events:
                lines.append(f"    sample {pos}: {status}")
        else:
            lines.append("  no stream status events (clean capture)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Recording(duration={self.duration}, channels={self.channels}, "
            f"start={self._start}, calibration={self.calibration}, "
            f"status_events={len(self.status_events)})"
        )


class Segment:
    """
    A punch-in/punch-out recording segment: a finite Extent on the render
    timeline plus the WAV file its capture is written to.

    Recording starts at the extent's onset and stops at its end or at
    Transport.stop(), whichever comes first; the captured audio is then
    written to the file (auto-numbered by default — see
    DuplexRenderer.transport()).

    In a calibrated session the capture window is shifted by the measured
    round-trip offset, so the FILE IS THE MUSICAL REGION: sample 0 of the
    WAV is exactly the performance at extent.start. Reloading later is
    simply ``DelayPE(WavReaderPE(path), extent.start)`` — no metadata, no
    post-hoc alignment.

    Args:
        extent: Finite Extent [start, end) on the render timeline.
        path:  WAV filename to write the captured take to.
    """

    def __init__(self, extent: Extent, path: str):
        if extent.start is None or extent.end is None:
            raise ValueError(f"Segment extent must be finite, got {extent}")
        if extent.is_empty():
            raise ValueError(f"Segment extent must be non-empty, got {extent}")
        self.extent = extent
        self.path = str(path)
        self.recording: Recording | None = None  # filled during transport
        self.written_path: str | None = None  # actual file written (policy)
        self._complete = False

    @property
    def complete(self) -> bool:
        """True once the punch-out point was reached during capture."""
        return self._complete

    @property
    def captured(self) -> int:
        """Samples captured so far (may be < extent duration if stopped)."""
        return self.recording.duration if self.recording is not None else 0

    def as_pe(self) -> ProcessingElement:
        """The captured take as timeline-positioned PE material (the
        capture was compensated, so it sits exactly on the extent)."""
        if self.recording is None or self.recording.duration == 0:
            raise RuntimeError(f"Segment {self.path!r}: nothing captured yet.")
        from pygmu2.array_pe import ArrayPE
        from pygmu2.delay_pe import DelayPE

        return DelayPE(ArrayPE(self.recording.data), self.extent.start)

    def __repr__(self) -> str:
        return (
            f"Segment(extent={self.extent}, path={self.path!r}, "
            f"captured={self.captured}, complete={self._complete}, "
            f"written={self.written_path!r})"
        )
