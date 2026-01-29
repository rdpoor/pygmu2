"""
Tests for WavReaderPE and WavWriterPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
import tempfile
import os
import soundfile as sf
from pygmu2 import (
    WavReaderPE,
    WavWriterPE,
    SinePE,
    ConstantPE,
    MixPE,
    NullRenderer,
    Extent,
)


@pytest.fixture
def temp_wav_mono(tmp_path):
    """Create a temporary mono WAV file with known content (FLOAT format for precision)."""
    path = tmp_path / "test_mono.wav"
    # Create a simple ramp 0.0 to 1.0 over 1000 samples
    data = np.linspace(0.0, 1.0, 1000, dtype=np.float32).reshape(-1, 1)
    sf.write(str(path), data, 44100, subtype='FLOAT')  # Use FLOAT to avoid quantization
    return str(path), data


@pytest.fixture
def temp_wav_stereo(tmp_path):
    """Create a temporary stereo WAV file with known content (FLOAT format for precision)."""
    path = tmp_path / "test_stereo.wav"
    # Create stereo: left channel ramp up, right channel ramp down
    left = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    right = np.linspace(1.0, 0.0, 1000, dtype=np.float32)
    data = np.column_stack([left, right])
    sf.write(str(path), data, 44100, subtype='FLOAT')  # Use FLOAT to avoid quantization
    return str(path), data


class TestWavReaderPEBasics:
    """Test basic WavReaderPE creation and properties."""
    
    def test_create_reader(self, temp_wav_mono):
        path, _ = temp_wav_mono
        reader = WavReaderPE(path)
        assert reader.path == path
    
    def test_is_pure(self, temp_wav_mono):
        path, _ = temp_wav_mono
        reader = WavReaderPE(path)
        assert reader.is_pure() is True
    
    def test_no_inputs(self, temp_wav_mono):
        path, _ = temp_wav_mono
        reader = WavReaderPE(path)
        assert reader.inputs() == []
    
    def test_channel_count_mono(self, temp_wav_mono):
        path, _ = temp_wav_mono
        reader = WavReaderPE(path)
        assert reader.channel_count() == 1
    
    def test_channel_count_stereo(self, temp_wav_stereo):
        path, _ = temp_wav_stereo
        reader = WavReaderPE(path)
        assert reader.channel_count() == 2
    
    def test_extent(self, temp_wav_mono):
        path, data = temp_wav_mono
        reader = WavReaderPE(path)
        extent = reader.extent()
        assert extent.start == 0
        assert extent.end == len(data)
    
    def test_repr(self, temp_wav_mono):
        path, _ = temp_wav_mono
        reader = WavReaderPE(path)
        repr_str = repr(reader)
        assert "WavReaderPE" in repr_str
        # Path may have escaped backslashes on Windows
        assert "test_mono.wav" in repr_str


class TestWavReaderPERender:
    """Test WavReaderPE rendering."""
    
    def test_render_full_file(self, temp_wav_mono):
        path, expected_data = temp_wav_mono
        reader = WavReaderPE(path)
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(reader)
        renderer.start()
        
        snippet = reader.render(0, 1000)
        assert snippet.start == 0
        assert snippet.duration == 1000
        np.testing.assert_array_almost_equal(snippet.data, expected_data, decimal=5)
        
        renderer.stop()
    
    def test_render_partial(self, temp_wav_mono):
        path, expected_data = temp_wav_mono
        reader = WavReaderPE(path)
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(reader)
        renderer.start()
        
        snippet = reader.render(100, 200)
        assert snippet.start == 100
        assert snippet.duration == 200
        np.testing.assert_array_almost_equal(
            snippet.data, expected_data[100:300], decimal=5
        )
        
        renderer.stop()
    
    def test_render_stereo(self, temp_wav_stereo):
        path, expected_data = temp_wav_stereo
        reader = WavReaderPE(path)
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(reader)
        renderer.start()
        
        snippet = reader.render(0, 100)
        assert snippet.channels == 2
        np.testing.assert_array_almost_equal(
            snippet.data, expected_data[:100], decimal=5
        )
        
        renderer.stop()
    
    def test_render_before_extent(self, temp_wav_mono):
        path, _ = temp_wav_mono
        reader = WavReaderPE(path)
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(reader)
        renderer.start()
        
        snippet = reader.render(-100, 50)
        # Should be all zeros (before file start)
        np.testing.assert_array_equal(
            snippet.data, np.zeros((50, 1), dtype=np.float32)
        )
        
        renderer.stop()
    
    def test_render_after_extent(self, temp_wav_mono):
        path, _ = temp_wav_mono
        reader = WavReaderPE(path)
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(reader)
        renderer.start()
        
        snippet = reader.render(1000, 50)
        # Should be all zeros (after file end)
        np.testing.assert_array_equal(
            snippet.data, np.zeros((50, 1), dtype=np.float32)
        )
        
        renderer.stop()
    
    def test_render_spanning_start(self, temp_wav_mono):
        path, expected_data = temp_wav_mono
        reader = WavReaderPE(path)
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(reader)
        renderer.start()
        
        snippet = reader.render(-50, 100)
        # First 50 samples should be zero
        np.testing.assert_array_equal(
            snippet.data[:50], np.zeros((50, 1), dtype=np.float32)
        )
        # Next 50 should be from file
        np.testing.assert_array_almost_equal(
            snippet.data[50:], expected_data[:50], decimal=5
        )
        
        renderer.stop()
    
    def test_render_spanning_end(self, temp_wav_mono):
        path, expected_data = temp_wav_mono
        reader = WavReaderPE(path)
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(reader)
        renderer.start()
        
        snippet = reader.render(950, 100)
        # First 50 should be from file
        np.testing.assert_array_almost_equal(
            snippet.data[:50], expected_data[950:], decimal=5
        )
        # Last 50 should be zeros
        np.testing.assert_array_equal(
            snippet.data[50:], np.zeros((50, 1), dtype=np.float32)
        )
        
        renderer.stop()


class TestWavWriterPEBasics:
    """Test basic WavWriterPE creation and properties."""
    
    def test_create_writer(self, tmp_path):
        source = ConstantPE(0.5)
        path = str(tmp_path / "output.wav")
        writer = WavWriterPE(source, path)
        assert writer.path == path
    
    def test_has_input(self, tmp_path):
        source = ConstantPE(0.5)
        path = str(tmp_path / "output.wav")
        writer = WavWriterPE(source, path)
        assert writer.inputs() == [source]
    
    def test_is_not_pure(self, tmp_path):
        source = ConstantPE(0.5)
        path = str(tmp_path / "output.wav")
        writer = WavWriterPE(source, path)
        assert writer.is_pure() is False
    
    def test_repr(self, tmp_path):
        source = ConstantPE(0.5)
        path = str(tmp_path / "output.wav")
        writer = WavWriterPE(source, path)
        repr_str = repr(writer)
        assert "WavWriterPE" in repr_str
        assert "ConstantPE" in repr_str


class TestWavWriterPERender:
    """Test WavWriterPE rendering and file output."""
    
    def test_write_constant(self, tmp_path):
        source = ConstantPE(0.5)
        path = str(tmp_path / "output.wav")
        writer = WavWriterPE(source, path)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(writer)
        renderer.start()
        
        # Render 1000 samples
        snippet = writer.render(0, 1000)
        
        # Snippet should pass through
        np.testing.assert_array_almost_equal(
            snippet.data, np.full((1000, 1), 0.5, dtype=np.float32)
        )
        
        renderer.stop()
        
        # Verify file was written
        assert os.path.exists(path)
        data, rate = sf.read(path, dtype='float32')
        assert rate == 44100
        assert len(data) == 1000
        # PCM_16 has some quantization error
        np.testing.assert_array_almost_equal(data.reshape(-1, 1), snippet.data, decimal=3)
    
    def test_write_stereo(self, tmp_path):
        source = ConstantPE(0.5, channels=2)
        path = str(tmp_path / "output_stereo.wav")
        writer = WavWriterPE(source, path)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(writer)
        renderer.start()
        renderer.render(0, 1000)
        renderer.stop()
        
        # Verify stereo file
        data, rate = sf.read(path)
        assert data.shape == (1000, 2)
    
    def test_write_float_subtype(self, tmp_path):
        source = ConstantPE(0.5)
        path = str(tmp_path / "output_float.wav")
        writer = WavWriterPE(source, path, subtype='FLOAT')
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(writer)
        renderer.start()
        
        snippet = writer.render(0, 1000)
        renderer.stop()
        
        # Verify file - FLOAT should have no quantization error
        data, rate = sf.read(path, dtype='float32')
        np.testing.assert_array_almost_equal(data.reshape(-1, 1), snippet.data, decimal=6)
    
    def test_write_multiple_renders(self, tmp_path):
        source = ConstantPE(0.5)
        path = str(tmp_path / "output_multi.wav")
        writer = WavWriterPE(source, path)
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(writer)
        renderer.start()
        
        # Multiple render calls
        writer.render(0, 1000)
        writer.render(1000, 1000)
        writer.render(2000, 1000)
        
        assert writer.frames_written == 3000
        
        renderer.stop()
        
        # Verify total frames written
        data, _ = sf.read(path)
        assert len(data) == 3000
    
    def test_write_sine(self, tmp_path):
        sine = SinePE(frequency=440.0, amplitude=0.8)
        path = str(tmp_path / "sine.wav")
        writer = WavWriterPE(sine, path, subtype='FLOAT')
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(writer)
        renderer.start()
        renderer.render(0, 44100)  # 1 second
        renderer.stop()
        
        # Verify file exists and has correct length
        data, rate = sf.read(path)
        assert rate == 44100
        assert len(data) == 44100
        # Check amplitude
        assert np.max(np.abs(data)) <= 0.8 + 0.01


class TestWavRoundTrip:
    """Test reading and writing WAV files in a processing chain."""
    
    def test_read_write_chain(self, temp_wav_mono, tmp_path):
        """Test reading a file, processing, and writing to new file."""
        input_path, original_data = temp_wav_mono
        output_path = str(tmp_path / "output.wav")
        
        # Read -> Write chain
        reader = WavReaderPE(input_path)
        writer = WavWriterPE(reader, output_path, subtype='FLOAT')
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(writer)
        renderer.start()
        renderer.render(0, 1000)
        renderer.stop()
        
        # Verify output matches input
        output_data, _ = sf.read(output_path, dtype='float32')
        np.testing.assert_array_almost_equal(
            output_data.reshape(-1, 1), original_data, decimal=5
        )
    
    def test_mix_and_write(self, temp_wav_mono, tmp_path):
        """Test mixing file with generated audio."""
        input_path, _ = temp_wav_mono
        output_path = str(tmp_path / "mixed.wav")
        
        reader = WavReaderPE(input_path)
        sine = SinePE(frequency=440.0, amplitude=0.1)
        mixed = MixPE(reader, sine)
        writer = WavWriterPE(mixed, output_path, subtype='FLOAT')
        
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(writer)
        renderer.start()
        renderer.render(0, 1000)
        renderer.stop()
        
        # Verify output exists and has correct shape
        output_data, _ = sf.read(output_path)
        assert len(output_data) == 1000
