"""
Tests for AudioRenderer.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License

Note: These tests verify the API but don't actually play audio
(that would require speakers and be disruptive during testing).
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pygmu2 import (
    AudioRenderer,
    SinePE,
    ConstantPE,
    CropPE,
    RampPE,
    Extent,
)


class TestAudioRendererBasics:
    """Test basic AudioRenderer creation and properties."""

    def test_create_default(self):
        renderer = AudioRenderer()
        assert renderer.sample_rate == 44100
        assert renderer.device is None
        assert renderer.blocksize == 1024

    def test_create_custom_sample_rate(self):
        renderer = AudioRenderer(sample_rate=48000)
        assert renderer.sample_rate == 48000

    def test_create_custom_device(self):
        renderer = AudioRenderer(device=0)
        assert renderer.device == 0

    def test_create_custom_blocksize(self):
        renderer = AudioRenderer(blocksize=512)
        assert renderer.blocksize == 512

    def test_repr(self):
        renderer = AudioRenderer(sample_rate=44100, blocksize=1024)
        repr_str = repr(renderer)
        assert "AudioRenderer" in repr_str
        assert "44100" in repr_str
        assert "1024" in repr_str

    def test_is_renderer_subclass(self):
        from pygmu2 import Renderer
        renderer = AudioRenderer()
        assert isinstance(renderer, Renderer)


class TestAudioRendererLifecycle:
    """Test AudioRenderer lifecycle (set_source, start, stop)."""

    def test_set_source(self):
        renderer = AudioRenderer()
        sine = SinePE(frequency=440.0)
        cropped = CropPE(sine, Extent(0, 44100))

        renderer.set_source(cropped)
        assert renderer.source is cropped

    def test_start_stop(self):
        renderer = AudioRenderer()
        sine = SinePE(frequency=440.0)
        cropped = CropPE(sine, Extent(0, 44100))

        renderer.set_source(cropped)
        renderer.start()
        assert renderer.started is True

        renderer.stop()
        assert renderer.started is False

    def test_context_manager(self):
        sine = SinePE(frequency=440.0)
        cropped = CropPE(sine, Extent(0, 44100))

        with AudioRenderer() as renderer:
            renderer.set_source(cropped)
            renderer.start()
            assert renderer.started is True
        # Should be stopped after context
        assert renderer.started is False

    def test_stop_without_start(self):
        renderer = AudioRenderer()
        # Should not raise
        renderer.stop()


class TestAudioRendererPlayback:
    """Test AudioRenderer playback methods (mocked)."""

    @patch('sounddevice.OutputStream')
    def test_render_calls_sounddevice(self, mock_output_stream):
        mock_stream = MagicMock()
        mock_output_stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
        mock_output_stream.return_value.__exit__ = MagicMock(return_value=False)

        renderer = AudioRenderer()
        source = ConstantPE(0.5)
        cropped = CropPE(source, Extent(0, 1000))

        renderer.set_source(cropped)
        renderer.start()
        renderer.render(0, 100)

        # Verify OutputStream was used
        mock_output_stream.assert_called_once()
        # Verify write was called
        mock_stream.write.assert_called_once()
        written_data = mock_stream.write.call_args[0][0]
        assert written_data.shape == (100, 1)
        np.testing.assert_array_equal(
            written_data, np.full((100, 1), 0.5, dtype=np.float32)
        )

        renderer.stop()

    @patch('sounddevice.OutputStream')
    def test_play_range(self, mock_output_stream):
        mock_stream = MagicMock()
        mock_output_stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
        mock_output_stream.return_value.__exit__ = MagicMock(return_value=False)

        renderer = AudioRenderer()
        source = ConstantPE(0.5)
        cropped = CropPE(source, Extent(0, 1000))

        renderer.set_source(cropped)
        renderer.start()
        renderer.play_range(start=0, duration=100)

        mock_output_stream.assert_called_once()
        renderer.stop()

    @patch('sounddevice.OutputStream')
    def test_play_extent(self, mock_output_stream):
        mock_stream = MagicMock()
        mock_output_stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
        mock_output_stream.return_value.__exit__ = MagicMock(return_value=False)

        renderer = AudioRenderer()
        source = ConstantPE(0.5)
        cropped = CropPE(source, Extent(0, 1000))

        renderer.set_source(cropped)
        renderer.start()
        renderer.play_extent()

        # Should be called multiple times (chunks)
        assert mock_output_stream.call_count >= 1
        renderer.stop()

    def test_play_extent_infinite_raises(self):
        renderer = AudioRenderer()
        source = ConstantPE(0.5)  # Infinite extent

        renderer.set_source(source)
        renderer.start()

        with pytest.raises(RuntimeError, match="infinite"):
            renderer.play_extent()

        renderer.stop()

    def test_play_extent_no_source_raises(self):
        renderer = AudioRenderer()

        with pytest.raises(RuntimeError, match="No source"):
            renderer.play_extent()


class TestAudioRendererStreaming:
    """Test AudioRenderer streaming methods (mocked)."""

    @patch('sounddevice.OutputStream')
    def test_stream_start_stop(self, mock_output_stream):
        mock_stream = MagicMock()
        mock_stream.active = False
        mock_output_stream.return_value = mock_stream

        renderer = AudioRenderer()
        source = ConstantPE(0.5)
        cropped = CropPE(source, Extent(0, 44100))

        renderer.set_source(cropped)
        renderer.start()

        renderer.stream_start(start=0, end=44100)
        assert mock_output_stream.called
        assert mock_stream.start.called

        renderer.stream_stop()
        assert mock_stream.stop.called
        assert mock_stream.close.called

        renderer.stop()

    def test_stream_start_not_started_raises(self):
        renderer = AudioRenderer()
        source = ConstantPE(0.5)
        renderer.set_source(source)

        with pytest.raises(RuntimeError, match="Not started"):
            renderer.stream_start()

    @patch('sounddevice.OutputStream')
    def test_stream_start_twice_raises(self, mock_output_stream):
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_output_stream.return_value = mock_stream

        renderer = AudioRenderer()
        source = ConstantPE(0.5)
        cropped = CropPE(source, Extent(0, 44100))

        renderer.set_source(cropped)
        renderer.start()
        renderer.stream_start()

        with pytest.raises(RuntimeError, match="Already streaming"):
            renderer.stream_start()

        renderer.stop()

    def test_stream_position(self):
        renderer = AudioRenderer()
        assert renderer.stream_position == 0

    def test_is_streaming_false_initially(self):
        renderer = AudioRenderer()
        assert renderer.is_streaming is False


class TestAudioRendererDevices:
    """Test AudioRenderer device listing."""

    @patch('sounddevice.query_devices')
    def test_list_devices(self, mock_query):
        mock_query.return_value = "Device list"
        # Just verify it doesn't raise
        AudioRenderer.list_devices()
        mock_query.assert_called_once()

    @patch('sounddevice.query_devices')
    def test_get_default_device(self, mock_query):
        mock_query.return_value = {'name': 'Default Device'}
        result = AudioRenderer.get_default_device()
        mock_query.assert_called_once_with(kind='output')
        assert result['name'] == 'Default Device'


class TestAudioRendererIntegration:
    """Integration tests (still mocked, but fuller scenarios)."""

    @patch('sounddevice.OutputStream')
    def test_play_sine_wave(self, mock_output_stream):
        """Simulate playing a 1-second sine wave."""
        mock_stream = MagicMock()
        mock_output_stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
        mock_output_stream.return_value.__exit__ = MagicMock(return_value=False)

        sine = SinePE(frequency=440.0, amplitude=0.5)
        one_second = CropPE(sine, Extent(0, 44100))

        with AudioRenderer(sample_rate=44100) as renderer:
            renderer.set_source(one_second)
            renderer.start()
            renderer.play_extent()

        # Verify multiple chunks were played
        assert mock_output_stream.call_count >= 1

    @patch('sounddevice.OutputStream')
    def test_play_stereo(self, mock_output_stream):
        """Test stereo output."""
        mock_stream = MagicMock()
        mock_output_stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
        mock_output_stream.return_value.__exit__ = MagicMock(return_value=False)

        source = ConstantPE(0.5, channels=2)
        cropped = CropPE(source, Extent(0, 1000))

        renderer = AudioRenderer()
        renderer.set_source(cropped)
        renderer.start()
        renderer.render(0, 100)

        written_data = mock_stream.write.call_args[0][0]
        assert written_data.shape == (100, 2)

        renderer.stop()

    @patch('sounddevice.OutputStream')
    def test_play_with_gain(self, mock_output_stream):
        """Test playing with gain applied."""
        mock_stream = MagicMock()
        mock_output_stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
        mock_output_stream.return_value.__exit__ = MagicMock(return_value=False)

        from pygmu2 import GainPE

        source = ConstantPE(1.0)
        gained = GainPE(source, gain=0.5)
        cropped = CropPE(gained, Extent(0, 1000))

        renderer = AudioRenderer()
        renderer.set_source(cropped)
        renderer.start()
        renderer.render(0, 100)

        written_data = mock_stream.write.call_args[0][0]
        np.testing.assert_array_almost_equal(
            written_data, np.full((100, 1), 0.5, dtype=np.float32)
        )

        renderer.stop()
