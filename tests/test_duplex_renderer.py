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

    manual = False  # True: test drives blocks via pump(); False: run to end

    def _run_block(self) -> bool:
        """Run one callback iteration; False when the stream finished."""
        frames = self.blocksize
        indata = self._pop_input(frames)
        outdata = np.zeros((frames, self.out_channels), dtype=np.float32)
        try:
            self.callback(indata, outdata, frames, None, "")
        except (sd.CallbackStop, sd.CallbackAbort):
            self.pending.append(outdata[:, 0].copy())
            if self.finished_callback:
                self.finished_callback()
            return False
        self.pending.append(outdata[:, 0].copy())
        return True

    def _run_loop(self):
        while self._run_block():
            pass

    def pump(self, blocks: int) -> None:
        """Manual mode: advance the stream by N callback blocks."""
        for _ in range(blocks):
            if not self._run_block():
                break

    def start(self):
        if not FakeDuplexStream.manual:
            self._run_loop()

    def stop(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        self._run_loop()
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


class TestSegmentTransport:
    """Punch-in/punch-out via Segments (extent + WAV file)."""

    def _segment(self, start, end, path):
        from pygmu2 import Segment

        return Segment(pg.Extent(start, end), str(path))

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_compensated_capture_file_is_the_musical_region(self, tmp_path):
        """The headline property: with a calibrated session and a
        k-sample loopback, a segment's file contains EXACTLY the output
        that played over its extent — sample 0 == extent.start."""
        import soundfile as sf

        k = 2 * BLOCK + 37
        FakeDuplexStream.loopback_delay = k
        FakeDuplexStream.manual = False
        renderer = DuplexRenderer(sample_rate=44100, blocksize=BLOCK)
        renderer.calibrate(duration_seconds=0.5)
        assert renderer.calibration_offset == k

        total = 12 * BLOCK
        renderer.set_source(_ramp_source(total))
        renderer.start()
        a, b = 3 * BLOCK + 10, 5 * BLOCK - 20
        seg = self._segment(a, b, tmp_path / "verse.wav")
        t = renderer.transport([seg])
        t.wait()

        assert seg.complete
        assert seg.written_path == str(tmp_path / "verse.wav")
        data, sr = sf.read(seg.written_path, dtype="float32")
        assert sr == 44100
        assert len(data) == b - a
        # loopback: input == output delayed k; compensation removes k, so
        # the file content is the ramp values a..b-1 exactly
        np.testing.assert_allclose(data, np.arange(a, b, dtype=np.float32))

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_multiple_and_overlapping_segments(self, tmp_path):
        FakeDuplexStream.loopback_delay = 0
        FakeDuplexStream.manual = False
        renderer = _renderer(_ramp_source(8 * BLOCK))
        s1 = self._segment(0, 2 * BLOCK, tmp_path / "a.wav")
        s2 = self._segment(BLOCK, 3 * BLOCK, tmp_path / "b.wav")  # overlaps s1
        t = renderer.transport([s1, s2])
        t.wait()
        assert s1.complete and s2.complete
        assert s1.captured == 2 * BLOCK
        assert s2.captured == 2 * BLOCK
        assert s2.recording.start == BLOCK

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_stop_punches_out_partial_take(self, tmp_path):
        """Transport.stop() before the extent end writes the partial."""
        import soundfile as sf

        FakeDuplexStream.loopback_delay = 0
        FakeDuplexStream.manual = True
        try:
            renderer = _renderer(_ramp_source(20 * BLOCK))
            seg = self._segment(0, 10 * BLOCK, tmp_path / "take.wav")
            later = self._segment(15 * BLOCK, 16 * BLOCK, tmp_path / "never.wav")
            t = renderer.transport([seg, later])
            FakeDuplexStream.last_instance.pump(3)  # 3 blocks in
            t.stop()
            assert not seg.complete
            assert seg.written_path is not None  # partial written
            data, _ = sf.read(seg.written_path, dtype="float32")
            assert len(data) == 3 * BLOCK
            # a segment never reached writes nothing
            assert later.captured == 0 and later.written_path is None
        finally:
            FakeDuplexStream.manual = False

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_take_numbering_never_clobbers(self, tmp_path):
        import soundfile as sf

        FakeDuplexStream.loopback_delay = 0
        FakeDuplexStream.manual = False
        path = tmp_path / "riff.wav"
        written = []
        for _ in range(3):
            renderer = _renderer(_ramp_source(2 * BLOCK))
            seg = self._segment(0, BLOCK, path)
            renderer.transport([seg]).wait()
            written.append(seg.written_path)
        assert written == [
            str(path),
            str(tmp_path / "riff-1.wav"),
            str(tmp_path / "riff-2.wav"),
        ]
        for w in written:
            data, _ = sf.read(w, dtype="float32")
            assert len(data) == BLOCK

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_overwrite_policy(self, tmp_path):
        FakeDuplexStream.loopback_delay = 0
        FakeDuplexStream.manual = False
        path = tmp_path / "riff.wav"
        for _ in range(2):
            renderer = _renderer(_ramp_source(2 * BLOCK))
            seg = self._segment(0, BLOCK, path)
            renderer.transport([seg], on_exists="overwrite").wait()
            assert seg.written_path == str(path)
        assert not (tmp_path / "riff-1.wav").exists()

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_segment_as_pe_positions_on_extent(self, tmp_path):
        FakeDuplexStream.loopback_delay = 0
        FakeDuplexStream.manual = False
        renderer = _renderer(_ramp_source(4 * BLOCK))
        a, b = BLOCK, 3 * BLOCK
        seg = self._segment(a, b, tmp_path / "mid.wav")
        renderer.transport([seg]).wait()
        pe = seg.as_pe()
        assert pe.extent() == pg.Extent(a, b)

    @patch("pygmu2.duplex_renderer.sd.Stream", FakeDuplexStream)
    def test_transport_end_covers_segment_tail(self, tmp_path):
        """With calibration, the stream runs past the last punch-out by
        the offset so the compensated window is fully captured — even
        when the source extent ends earlier."""
        k = BLOCK + 5
        FakeDuplexStream.loopback_delay = k
        FakeDuplexStream.manual = False
        renderer = DuplexRenderer(sample_rate=44100, blocksize=BLOCK)
        renderer.calibrate(duration_seconds=0.5)
        total = 4 * BLOCK
        renderer.set_source(_ramp_source(total))
        renderer.start()
        seg = self._segment(2 * BLOCK, total, tmp_path / "tail.wav")
        t = renderer.transport([seg])
        t.wait()
        assert seg.complete
        assert seg.captured == total - 2 * BLOCK

    def test_bad_policy_raises(self):
        renderer = DuplexRenderer(sample_rate=44100)
        renderer.set_source(pg.ConstantPE(0.5))
        renderer.start()
        with pytest.raises(ValueError, match="on_exists"):
            renderer.transport([], on_exists="clobber")

    def test_segment_validation(self):
        from pygmu2 import Segment

        with pytest.raises(ValueError, match="finite"):
            Segment(pg.Extent(0, None), "x.wav")
        with pytest.raises(ValueError, match="non-empty"):
            Segment(pg.Extent(5, 5), "x.wav")
