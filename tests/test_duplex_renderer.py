"""
Tests for DuplexRenderer and Recording — fully device-free.

A FakeDuplexStream stands in for sounddevice.Stream and drives the
callback synchronously, simulating a physical loopback: everything sent
to the output reappears at the input exactly `loopback_delay` samples
later. That lets the calibration test assert the measured round-trip
offset equals the simulated one EXACTLY, exercising the whole
click/cross-correlation path with zero hardware.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np
import pytest
import sounddevice as sd
from unittest.mock import patch

import pygmu2 as pg
from pygmu2 import DuplexRenderer, Recording

BLOCK = 256


class FakeDuplexStream:
    """Synchronous stand-in for sd.Stream (duplex).

    Entering the context runs the callback loop to completion. Input is a
    software loopback of the output, delayed by `loopback_delay` samples
    (class attribute, set per test); with delay d, the sample written to
    output position T arrives at input position T + d — one clock, like
    real duplex hardware.
    """

    loopback_delay = 0
    last_instance: "FakeDuplexStream | None" = None

    def __init__(
        self,
        samplerate=None,
        blocksize=None,
        device=None,
        channels=None,
        dtype=None,
        latency=None,
        callback=None,
        finished_callback=None,
    ):
        self.blocksize = blocksize or BLOCK
        self.in_channels, self.out_channels = channels
        self.callback = callback
        self.finished_callback = finished_callback
        # pending output samples not yet "arrived" at the input
        self.pending = [np.zeros(self.loopback_delay, dtype=np.float32)]
        FakeDuplexStream.last_instance = self

    def _pop_input(self, frames: int) -> np.ndarray:
        buf = np.concatenate(self.pending) if self.pending else np.zeros(0)
        take = buf[:frames]
        if len(take) < frames:
            take = np.concatenate([take, np.zeros(frames - len(take))])
        self.pending = [buf[frames:].astype(np.float32)]
        out = np.repeat(take.reshape(-1, 1), self.in_channels, axis=1)
        return out.astype(np.float32)

    def __enter__(self):
        frames = self.blocksize
        while True:
            indata = self._pop_input(frames)
            outdata = np.zeros((frames, self.out_channels), dtype=np.float32)
            try:
                self.callback(indata, outdata, frames, None, "")
            except sd.CallbackStop:
                self.pending.append(outdata[:, 0].copy())
                break
            except sd.CallbackAbort:
                break
            self.pending.append(outdata[:, 0].copy())
        if self.finished_callback:
            self.finished_callback()
        return self

    def __exit__(self, *exc):
        return False


def _renderer(source, input_channels=1):
    renderer = DuplexRenderer(
        sample_rate=44100, blocksize=BLOCK, input_channels=input_channels
    )
    renderer.set_source(source)
    renderer.start()
    return renderer


def _ramp_source(n):
    """Finite source whose value IS the sample index (timestamp probe)."""
    from tests.probes import IdentityPE

    return pg.CropPE(IdentityPE(), 0, n)


class TestRecordingStamps:
    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_blocks_stamped_with_render_positions(self):
        FakeDuplexStream.loopback_delay = 0
        renderer = _renderer(_ramp_source(4 * BLOCK))
        rec = renderer.record_range(0, 4 * BLOCK)
        assert rec.start == 0
        assert rec.duration == 4 * BLOCK
        assert rec.data.shape == (4 * BLOCK, 1)
        assert rec.data.dtype == np.float32

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_loopback_content_lands_delayed_by_k(self):
        """With a k-sample loopback, output sample T appears in the
        recording at stamped position T + k — the constant-offset property
        the whole design rests on."""
        k = 2 * BLOCK + 37
        FakeDuplexStream.loopback_delay = k
        total = 8 * BLOCK
        renderer = _renderer(_ramp_source(total))
        rec = renderer.record_range(0, total)
        captured = rec.data[:, 0]
        # captured[t] should equal output value at (t - k) == t - k (ramp)
        t = np.arange(k, total)
        np.testing.assert_allclose(captured[t], (t - k).astype(np.float32))
        assert np.all(captured[:k] == 0.0)

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_record_range_offset_start(self):
        FakeDuplexStream.loopback_delay = 0
        renderer = _renderer(pg.ConstantPE(0.5))
        rec = renderer.record_range(1000, 2 * BLOCK)
        assert rec.start == 1000

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_record_extent(self):
        FakeDuplexStream.loopback_delay = 0
        renderer = _renderer(_ramp_source(3 * BLOCK))
        rec = renderer.record_extent()
        assert rec.start == 0
        assert rec.duration == 3 * BLOCK

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_stereo_input(self):
        FakeDuplexStream.loopback_delay = 0
        renderer = _renderer(pg.ConstantPE(0.5), input_channels=2)
        rec = renderer.record_range(0, BLOCK)
        assert rec.channels == 2
        assert rec.data.shape == (BLOCK, 2)

    def test_output_raises(self):
        renderer = DuplexRenderer(sample_rate=44100)
        with pytest.raises(RuntimeError, match="callback-driven"):
            renderer._output(None)

    def test_record_extent_infinite_raises(self):
        renderer = DuplexRenderer(sample_rate=44100)
        renderer.set_source(pg.ConstantPE(0.5))
        renderer.start()
        with pytest.raises(RuntimeError, match="infinite"):
            renderer.record_extent()


class TestRecordingContainer:
    def _filled(self, calibration=None):
        rec = Recording(sample_rate=44100, channels=1, calibration=calibration)
        rec._append(1000, np.full((100, 1), 0.25, dtype=np.float32), "")
        rec._append(1100, np.full((100, 1), 0.5, dtype=np.float32), "")
        return rec

    def test_as_pe_raw_placement(self):
        rec = self._filled()
        pe = rec.as_pe()
        assert pe.extent() == pg.Extent(1000, 1200)
        renderer = pg.NullRenderer(sample_rate=44100)
        renderer.set_source(pe)
        renderer.start()
        data = pe.render(1000, 200).data[:, 0]
        assert np.all(data[:100] == np.float32(0.25))
        assert np.all(data[100:] == np.float32(0.5))

    def test_as_pe_applies_calibration(self):
        rec = self._filled(calibration=300)
        assert rec.as_pe().extent() == pg.Extent(700, 900)
        assert rec.as_pe(shift=0).extent() == pg.Extent(1000, 1200)  # raw
        assert rec.as_pe(shift=100).extent() == pg.Extent(900, 1100)

    def test_status_events_surface_in_summary(self):
        rec = Recording(sample_rate=44100, channels=1)
        rec._append(0, np.zeros((10, 1), dtype=np.float32), "input overflow")
        assert rec.status_events == [(0, "input overflow")]
        assert "input overflow" in rec.summary()
        clean = self._filled()
        assert "clean capture" in clean.summary()

    def test_save_roundtrip(self, tmp_path):
        import soundfile as sf

        rec = self._filled()
        path = str(tmp_path / "take.wav")
        rec.save(path)
        data, sr = sf.read(path, dtype="float32")
        assert sr == 44100
        assert len(data) == 200

    def test_empty_recording_raises(self):
        rec = Recording(sample_rate=44100, channels=1)
        with pytest.raises(RuntimeError, match="empty"):
            rec.start


class TestCalibration:
    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_measures_simulated_loopback_exactly(self):
        """The headline test: with a simulated k-sample round trip, the
        click calibration must measure exactly k."""
        k = 3 * BLOCK + 123
        FakeDuplexStream.loopback_delay = k
        renderer = DuplexRenderer(sample_rate=44100, blocksize=BLOCK)
        measured = renderer.calibrate(duration_seconds=0.5)
        assert measured == k
        assert renderer.calibration_offset == k

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_calibration_stamps_subsequent_recordings(self):
        k = BLOCK + 11
        FakeDuplexStream.loopback_delay = k
        renderer = DuplexRenderer(sample_rate=44100, blocksize=BLOCK)
        renderer.calibrate(duration_seconds=0.5)
        renderer.set_source(_ramp_source(2 * BLOCK))
        renderer.start()
        rec = renderer.record_range(0, 2 * BLOCK)
        assert rec.calibration == k
        # calibrated placement: raw start minus measured offset
        assert rec.as_pe().extent().start == 0 - k

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_restores_source_and_run_state(self):
        k = BLOCK
        FakeDuplexStream.loopback_delay = k
        renderer = DuplexRenderer(sample_rate=44100, blocksize=BLOCK)
        source = _ramp_source(BLOCK)
        renderer.set_source(source)
        renderer.start()
        renderer.calibrate(duration_seconds=0.25)
        assert renderer.source is source
        assert renderer.started

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_silent_input_raises(self):
        """No click in the capture (e.g. muted mic) must refuse loudly,
        not guess."""

        class SilentStream(FakeDuplexStream):
            def _pop_input(self, frames):
                return np.zeros((frames, self.in_channels), dtype=np.float32)

        with patch("pygmu2.duplex_renderer.sd.Stream", SilentStream):
            renderer = DuplexRenderer(sample_rate=44100, blocksize=BLOCK)
            with pytest.raises(RuntimeError, match="no dominant click"):
                renderer.calibrate(duration_seconds=0.25)
