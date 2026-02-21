"""
Tests for AudioReaderPE.

miniaudio is mocked throughout so no real MP3 files or the miniaudio
package itself are required to run the test suite.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import sys
import types
import numpy as np
import pytest

import pygmu2 as pg


# ---------------------------------------------------------------------------
# Fake miniaudio module
# ---------------------------------------------------------------------------

def _make_fake_miniaudio(
    nchannels: int = 2,
    file_sample_rate: int = 44100,
    num_frames: int = 1000,
    decoded_data: np.ndarray | None = None,
):
    """
    Build a fake miniaudio module that returns deterministic data.

    decoded_data: if None, a ramp (0..1) repeated across all channels is used.
    """
    ma = types.ModuleType("miniaudio")

    class SampleFormat:
        FLOAT32 = "float32"

    class FileInfo:
        def __init__(self):
            self.nchannels = nchannels
            self.sample_rate = file_sample_rate
            self.num_frames = num_frames

    class DecodedSoundFile:
        def __init__(self, data: np.ndarray):
            self.nchannels = data.shape[1]
            self.sample_rate = file_sample_rate
            self.num_frames = data.shape[0]
            self.samples = data.astype(np.float32).tobytes()

    if decoded_data is None:
        ramp = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
        decoded_data = np.column_stack([ramp] * nchannels)

    ma.SampleFormat = SampleFormat
    ma.get_file_info = lambda path: FileInfo()
    ma.decode_file = lambda path, output_format, nchannels, sample_rate: DecodedSoundFile(decoded_data)

    return ma


@pytest.fixture(autouse=True)
def _inject_fake_miniaudio():
    """Inject a fake miniaudio into sys.modules for every test."""
    fake = _make_fake_miniaudio()
    sys.modules["miniaudio"] = fake
    # Also clear any cached _file_info / _data from previous tests
    yield
    sys.modules.pop("miniaudio", None)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAudioReaderPEConstruction:

    def test_path_property(self):
        pe = pg.AudioReaderPE("song.mp3")
        assert pe.path == "song.mp3"

    def test_is_pure(self):
        pe = pg.AudioReaderPE("song.mp3")
        assert pe.is_pure() is True

    def test_no_inputs(self):
        pe = pg.AudioReaderPE("song.mp3")
        assert pe.inputs() == []

    def test_repr(self):
        pe = pg.AudioReaderPE("song.mp3")
        assert "AudioReaderPE" in repr(pe)
        assert "song.mp3" in repr(pe)

    def test_channel_count_from_file_info(self):
        sys.modules["miniaudio"] = _make_fake_miniaudio(nchannels=2)
        pe = pg.AudioReaderPE("song.mp3")
        assert pe.channel_count() == 2

    def test_channel_count_mono(self):
        sys.modules["miniaudio"] = _make_fake_miniaudio(nchannels=1)
        pe = pg.AudioReaderPE("song.mp3")
        assert pe.channel_count() == 1

    def test_file_sample_rate(self):
        sys.modules["miniaudio"] = _make_fake_miniaudio(file_sample_rate=48000)
        pe = pg.AudioReaderPE("song.mp3")
        assert pe.file_sample_rate == 48000


# ---------------------------------------------------------------------------
# Extent
# ---------------------------------------------------------------------------

class TestAudioReaderPEExtent:

    def test_extent_matches_frame_count_at_same_rate(self):
        """When file rate == system rate, extent == num_frames."""
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            file_sample_rate=44100, num_frames=1000
        )
        pe = pg.AudioReaderPE("song.mp3")
        ext = pe.extent()
        assert ext.start == 0
        assert ext.end == 1000

    def test_extent_scaled_when_rates_differ(self):
        """Extent is scaled by system_rate / file_rate when resampling."""
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            file_sample_rate=22050, num_frames=1000
        )
        pe = pg.AudioReaderPE("song.mp3")
        # system rate is 44100; file is 22050 → 2x as many output frames
        ext = pe.extent()
        assert ext.start == 0
        assert ext.end == 2000


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestAudioReaderPERender:

    def test_render_returns_correct_shape(self):
        sys.modules["miniaudio"] = _make_fake_miniaudio(nchannels=2, num_frames=500)
        pe = pg.AudioReaderPE("song.mp3")
        snip = pe.render(0, 100)
        assert snip.data.shape == (100, 2)

    def test_render_mono_returns_correct_shape(self):
        ramp = np.linspace(0.0, 1.0, 500, dtype=np.float32).reshape(-1, 1)
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            nchannels=1, num_frames=500, decoded_data=ramp
        )
        pe = pg.AudioReaderPE("song.mp3")
        snip = pe.render(0, 100)
        assert snip.data.shape == (100, 1)

    def test_render_values_match_decoded_data(self):
        """Rendered samples should equal the decoded buffer."""
        ramp = np.linspace(0.0, 1.0, 200, dtype=np.float32)
        data = np.column_stack([ramp, ramp])
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            nchannels=2, num_frames=200, decoded_data=data
        )
        pe = pg.AudioReaderPE("song.mp3")
        snip = pe.render(0, 200)
        np.testing.assert_allclose(snip.data, data, atol=1e-6)

    def test_render_partial_window(self):
        """Rendering a sub-range returns the correct slice."""
        ramp = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(-1, 1)
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            nchannels=1, num_frames=100, decoded_data=ramp
        )
        pe = pg.AudioReaderPE("song.mp3")
        snip = pe.render(10, 20)
        np.testing.assert_allclose(snip.data, ramp[10:30], atol=1e-6)

    def test_render_before_extent_is_zeros(self):
        """Samples before frame 0 are zero-filled."""
        data = np.ones((100, 1), dtype=np.float32)
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            nchannels=1, num_frames=100, decoded_data=data
        )
        pe = pg.AudioReaderPE("song.mp3")
        snip = pe.render(-10, 20)
        # First 10 samples are before the file → zeros
        np.testing.assert_array_equal(snip.data[:10], 0.0)
        # Next 10 samples overlap with file → ones
        np.testing.assert_allclose(snip.data[10:], 1.0, atol=1e-6)

    def test_render_past_end_is_zeros(self):
        """Samples past the end of the file are zero-filled."""
        data = np.ones((50, 1), dtype=np.float32)
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            nchannels=1, num_frames=50, decoded_data=data
        )
        pe = pg.AudioReaderPE("song.mp3")
        snip = pe.render(40, 20)
        # Samples 40-49 are in the file → ones
        np.testing.assert_allclose(snip.data[:10], 1.0, atol=1e-6)
        # Samples 50-59 are past the end → zeros
        np.testing.assert_array_equal(snip.data[10:], 0.0)

    def test_render_entirely_outside_is_zeros(self):
        """Rendering entirely outside the file returns zeros."""
        sys.modules["miniaudio"] = _make_fake_miniaudio(num_frames=100)
        pe = pg.AudioReaderPE("song.mp3")
        snip = pe.render(200, 50)
        np.testing.assert_array_equal(snip.data, 0.0)

    def test_render_is_repeatable(self):
        """Pure PE: same request returns same data (no state consumed)."""
        ramp = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(-1, 1)
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            nchannels=1, num_frames=100, decoded_data=ramp
        )
        pe = pg.AudioReaderPE("song.mp3")
        snip1 = pe.render(0, 50)
        snip2 = pe.render(0, 50)
        np.testing.assert_array_equal(snip1.data, snip2.data)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestAudioReaderPELifecycle:

    def test_on_stop_releases_buffer(self):
        sys.modules["miniaudio"] = _make_fake_miniaudio(num_frames=100)
        pe = pg.AudioReaderPE("song.mp3")
        pe.render(0, 10)          # triggers decode
        assert pe._data is not None
        pe.on_stop()
        assert pe._data is None

    def test_render_without_renderer_still_works(self):
        """_ensure_data() should trigger decode if _on_start() was never called."""
        sys.modules["miniaudio"] = _make_fake_miniaudio(num_frames=100)
        pe = pg.AudioReaderPE("song.mp3")
        snip = pe.render(0, 10)
        assert snip.data.shape == (10, 2)

    def test_on_start_with_null_renderer(self):
        """Full renderer lifecycle should work."""
        sys.modules["miniaudio"] = _make_fake_miniaudio(nchannels=1, num_frames=200)
        pe = pg.AudioReaderPE("song.mp3")
        renderer = pg.NullRenderer(sample_rate=44100)
        renderer.set_source(pe)
        renderer.start()
        snip = pe.render(0, 100)
        assert snip.data.shape == (100, 1)
        renderer.stop()
        assert pe._data is None


# ---------------------------------------------------------------------------
# Gain / normalization
# ---------------------------------------------------------------------------

class TestAudioReaderPEGain:

    def test_max_level_db_none_leaves_data_unchanged(self):
        """Default (max_level_db=None) should not alter the decoded samples."""
        data = np.full((100, 1), 0.5, dtype=np.float32)
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            nchannels=1, num_frames=100, decoded_data=data
        )
        pe = pg.AudioReaderPE("song.mp3")
        snip = pe.render(0, 100)
        np.testing.assert_allclose(snip.data, data, atol=1e-6)

    def test_max_level_db_zero_normalizes_to_full_scale(self):
        """max_level_db=0.0 should scale so the peak sample equals 1.0."""
        data = np.full((100, 1), 0.25, dtype=np.float32)
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            nchannels=1, num_frames=100, decoded_data=data
        )
        pe = pg.AudioReaderPE("song.mp3", max_level_db=0.0)
        snip = pe.render(0, 100)
        np.testing.assert_allclose(np.max(np.abs(snip.data)), 1.0, atol=1e-5)

    def test_max_level_db_negative_sets_headroom(self):
        """max_level_db=-6.0 should leave ~6 dB of headroom (peak ≈ 0.5)."""
        data = np.full((100, 1), 0.25, dtype=np.float32)
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            nchannels=1, num_frames=100, decoded_data=data
        )
        pe = pg.AudioReaderPE("song.mp3", max_level_db=-6.0206)  # -6.0206 dB ≈ 0.5
        snip = pe.render(0, 100)
        np.testing.assert_allclose(np.max(np.abs(snip.data)), 0.5, atol=1e-4)

    def test_max_level_db_silence_does_not_raise(self):
        """All-zero input with max_level_db set should return zeros without error."""
        data = np.zeros((100, 1), dtype=np.float32)
        sys.modules["miniaudio"] = _make_fake_miniaudio(
            nchannels=1, num_frames=100, decoded_data=data
        )
        pe = pg.AudioReaderPE("song.mp3", max_level_db=0.0)
        snip = pe.render(0, 100)
        np.testing.assert_array_equal(snip.data, 0.0)

    def test_repr_includes_max_level_db_when_set(self):
        pe = pg.AudioReaderPE("song.mp3", max_level_db=-1.0)
        assert "max_level_db" in repr(pe)
        assert "-1.0" in repr(pe)

    def test_repr_omits_max_level_db_when_none(self):
        pe = pg.AudioReaderPE("song.mp3")
        assert "max_level_db" not in repr(pe)


# ---------------------------------------------------------------------------
# Missing dependency
# ---------------------------------------------------------------------------

class TestAudioReaderPEMissingDep:

    def test_missing_miniaudio_raises_import_error(self):
        """A helpful ImportError is raised if miniaudio is not installed."""
        # Remove the fake module to simulate miniaudio being absent
        sys.modules.pop("miniaudio", None)
        # Also make import fail
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None
        import builtins
        original = builtins.__import__

        def _block_miniaudio(name, *args, **kwargs):
            if name == "miniaudio":
                raise ImportError("No module named 'miniaudio'")
            return original(name, *args, **kwargs)

        builtins.__import__ = _block_miniaudio
        try:
            pe = pg.AudioReaderPE("song.mp3")
            with pytest.raises(ImportError, match="miniaudio"):
                pe._ensure_file_info()
        finally:
            builtins.__import__ = original
            # Restore fake for other tests
            sys.modules["miniaudio"] = _make_fake_miniaudio()
