"""
DuplexRenderer - simultaneous playback and sample-synchronized recording.

Uses a single full-duplex sounddevice Stream, so input and output share one
device clock: every callback delivers the next input block alongside the
request for the next output block. Captured audio is stamped with the render
position of the same callback's output — sample-accurate relative to the
render timeline up to a constant round-trip offset (output latency + input
latency + the acoustic path), which calibrate() measures exactly by playing
a click through the normal path and cross-correlating the recording. No
estimates, no manually tuned latency numbers.

Intended workflow (musician monitors their instrument externally — amp,
acoustic — not through software):

    renderer = DuplexRenderer(input_channels=1)
    renderer.set_source(backing_track)
    renderer.start()
    renderer.calibrate()                    # once per session/setup
    take = renderer.record_extent()         # plays backing, records input
    renderer.stop()

    overdub = MixPE(backing_track, take.as_pe())   # sample-exact

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from pygmu2.recording import Recording
from pygmu2.renderer import Renderer
from pygmu2.snippet import Snippet
from pygmu2.logger import get_logger

logger = get_logger(__name__)

# Calibration click template parameters (seeded noise burst; a bare impulse
# is too weak acoustically to survive speaker -> room -> mic reliably)
_CLICK_SEED = 20260826
_CLICK_MS = 10.0
_CLICK_AMPLITUDE = 0.7
# Peak dominance required to accept a calibration measurement: the highest
# cross-correlation peak must exceed the best peak outside its neighborhood
# by this factor, or we refuse to guess.
_PEAK_DOMINANCE = 3.0


class DuplexRenderer(Renderer):
    """
    Renderer that plays the source graph and records input simultaneously
    on one device clock.

    Args:
        sample_rate: Hz; None reads the global set_sample_rate().
        device: sounddevice device — a single id/name for both directions,
            or an (input, output) pair. None = system defaults.
        blocksize: Frames per callback (default 1024).
        latency: sounddevice latency setting ('low', 'high', or seconds).
        input_channels: Channels to capture (default 1).
    """

    def __init__(
        self,
        sample_rate: int | None = None,
        device: object = None,
        blocksize: int = 1024,
        latency: str | float = "low",
        *,
        input_channels: int = 1,
    ):
        super().__init__(sample_rate=sample_rate)
        self._device = device
        self._blocksize = int(blocksize)
        self._latency = latency
        self._input_channels = int(input_channels)
        self.calibration_offset: int | None = None

    # ------------------------------------------------------------------
    # Renderer interface
    # ------------------------------------------------------------------

    def _output(self, snippet: Snippet) -> None:
        raise RuntimeError(
            "DuplexRenderer is callback-driven; use record_range() or "
            "record_extent() (or AudioRenderer for playback-only)."
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_range(self, start: int, duration: int) -> Recording:
        """
        Play the source for [start, start+duration) while recording input.
        Blocking; returns the stamped Recording.
        """
        if self._source is None:
            raise RuntimeError("No source set. Call set_source() first.")
        if not self._started:
            raise RuntimeError("Not started. Call start() first.")
        if duration < 1:
            raise ValueError(f"duration must be >= 1, got {duration}")

        source = self._source
        out_channels = source.channel_count() or 1
        recording = Recording(
            sample_rate=self._sample_rate,
            channels=self._input_channels,
            calibration=self.calibration_offset,
        )

        end = start + duration
        position = start
        done = threading.Event()
        errors: list[BaseException] = []

        def callback(indata, outdata, frames, time_info, status):
            nonlocal position
            try:
                remaining = end - position
                if remaining <= 0:
                    outdata.fill(0)
                    raise sd.CallbackStop()
                n = min(frames, remaining)

                # Render the graph into the output block
                snippet = source.render(position, n)
                outdata[:n] = snippet.data
                if n < frames:
                    outdata[n:] = 0

                # Capture the input block, stamped with this callback's
                # render position (same device clock => same timeline)
                recording._append(position, np.copy(indata[:n]), status)

                position += n
                if position >= end:
                    raise sd.CallbackStop()
            except sd.CallbackStop:
                raise
            except BaseException as exc:  # surface, never swallow (R2)
                errors.append(exc)
                raise sd.CallbackAbort() from exc

        stream = sd.Stream(
            samplerate=self._sample_rate,
            blocksize=self._blocksize,
            device=self._device,
            channels=(self._input_channels, out_channels),
            dtype="float32",
            latency=self._latency,
            callback=callback,
            finished_callback=done.set,
        )
        with stream:
            done.wait()

        if errors:
            raise errors[0]

        logger.info(
            f"Recorded {recording.duration} samples "
            f"({len(recording.status_events)} status events)"
        )
        return recording

    def record_extent(self) -> Recording:
        """Play the source's entire (finite) extent while recording."""
        if self._source is None:
            raise RuntimeError("No source set. Call set_source() first.")
        extent = self._source.extent()
        if extent.start is None or extent.end is None:
            raise RuntimeError(
                "Cannot record_extent() on an infinite source. "
                "Use CropPE to bound it, or record_range()."
            )
        return self.record_range(extent.start, extent.end - extent.start)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def _click_template(cls, sample_rate: int) -> np.ndarray:
        """The known calibration burst: seeded noise, exponential decay."""
        n = max(8, int(sample_rate * _CLICK_MS / 1000.0))
        rng = np.random.default_rng(_CLICK_SEED)
        burst = rng.uniform(-1.0, 1.0, n) * np.exp(-np.arange(n) / (n / 4.0))
        return (_CLICK_AMPLITUDE * burst).astype(np.float32)

    def calibrate(self, duration_seconds: float = 1.0) -> int:
        """
        Measure the exact round-trip offset (in samples) between the render
        timeline and the recorded timeline: output latency + input latency
        + the acoustic path from speaker to microphone.

        Plays a short click through the normal playback path while
        recording, then cross-correlates the recording against the known
        click. The measured offset is stored on this renderer and stamped
        into every subsequent Recording, whose as_pe() applies it — making
        overdubs land sample-exact.

        Run once per session/setup (device pair, buffer settings, and mic/
        speaker positions all contribute; re-run if any change).

        Returns:
            The measured offset in samples.

        Raises:
            RuntimeError: If no dominant click is detected in the recording
                (wrong input device, muted mic, or level too low).
        """
        source = self._source
        started = self._started
        sr = self._sample_rate

        template = self._click_template(sr)
        total = max(int(duration_seconds * sr), len(template) * 4)
        # Place the click early but not at 0, leaving room for negative
        # measurement error margins.
        click_at = len(template)

        from pygmu2.array_pe import ArrayPE
        from pygmu2.crop_pe import CropPE
        from pygmu2.delay_pe import DelayPE

        click_source = CropPE(
            DelayPE(ArrayPE(template), click_at), 0, total, clip=False
        )

        # Temporarily swap in the click source
        if started:
            self.stop()
        self.set_source(click_source)
        self.start()
        try:
            recording = self.record_range(0, total)
        finally:
            self.stop()
            if source is not None:
                self.set_source(source)
                if started:
                    self.start()

        captured = recording.data[:, 0].astype(np.float64)
        corr = np.correlate(captured, template.astype(np.float64), mode="full")
        # lag k means the template starts at captured sample k
        lags = np.arange(-len(template) + 1, len(captured))
        peak_idx = int(np.argmax(np.abs(corr)))
        peak = float(np.abs(corr[peak_idx]))

        # Dominance guard: best peak outside the main peak's neighborhood
        guard = len(template)
        masked = np.abs(corr).copy()
        lo = max(0, peak_idx - guard)
        masked[lo : peak_idx + guard] = 0.0
        runner_up = float(np.max(masked)) if masked.size else 0.0
        if peak <= 1e-9 or (runner_up > 0 and peak / runner_up < _PEAK_DOMINANCE):
            raise RuntimeError(
                "Calibration failed: no dominant click detected in the "
                f"recording (peak={peak:.3g}, "
                f"need >= {_PEAK_DOMINANCE}). Check that the input device "
                "is the right microphone, unmuted, and can hear the "
                "speakers; then re-run calibrate()."
            )

        offset = int(lags[peak_idx]) - click_at
        if offset < 0:
            raise RuntimeError(
                f"Calibration measured a negative offset ({offset} samples) "
                "— the 'recorded' click arrived before it was played, which "
                "is physically impossible. Check the device configuration."
            )

        self.calibration_offset = offset
        logger.info(
            f"Calibrated: round-trip offset {offset} samples "
            f"({1000.0 * offset / sr:.1f} ms)"
        )
        return offset

    def __repr__(self) -> str:
        return (
            f"DuplexRenderer(sample_rate={self._sample_rate}, "
            f"device={self._device}, blocksize={self._blocksize}, "
            f"input_channels={self._input_channels}, "
            f"calibration_offset={self.calibration_offset})"
        )
