"""
Tests for TransformPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    TransformPE,
    ConstantPE,
    SinePE,
    PiecewisePE,
    NullRenderer,
    ratio_to_db,
    pitch_to_freq,
)


class TestTransformPEBasics:
    """Test basic TransformPE creation and properties."""
    
    def test_create_with_numpy_func(self):
        source = ConstantPE(1.0)
        transform = TransformPE(source, func=np.abs)
        
        assert transform.source is source
        assert transform.func is np.abs
        assert transform.name == "absolute"  # numpy function name
    
    def test_create_with_lambda(self):
        source = ConstantPE(1.0)
        transform = TransformPE(source, func=lambda x: x ** 2)
        
        assert transform.source is source
        assert transform.name == "<lambda>"
    
    def test_create_with_custom_name(self):
        source = ConstantPE(1.0)
        transform = TransformPE(source, func=np.tanh, name="soft_clip")
        
        assert transform.name == "soft_clip"
    
    def test_inputs(self):
        source = ConstantPE(1.0)
        transform = TransformPE(source, func=np.abs)
        
        assert transform.inputs() == [source]
    
    def test_is_pure(self):
        """TransformPE should be pure (stateless)."""
        source = ConstantPE(1.0)
        transform = TransformPE(source, func=np.abs)
        
        assert transform.is_pure() is True
    
    def test_channel_count_passthrough(self):
        source = ConstantPE(1.0, channels=2)
        transform = TransformPE(source, func=np.abs)
        
        assert transform.channel_count() == 2
    
    def test_extent_from_source(self):
        source = PiecewisePE(0.0, 1.0, duration=1000)
        transform = TransformPE(source, func=np.abs)
        
        extent = transform.extent()
        assert extent.start == 0
        assert extent.end == 1000
    
    def test_repr(self):
        source = ConstantPE(1.0)
        transform = TransformPE(source, func=np.abs)
        
        repr_str = repr(transform)
        assert "TransformPE" in repr_str
        assert "ConstantPE" in repr_str
        assert "absolute" in repr_str


class TestTransformPEFunctions:
    """Test various transformation functions."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_abs_function(self):
        """Test absolute value transformation."""
        source = SinePE(frequency=100.0, amplitude=1.0)
        transform = TransformPE(source, func=np.abs)
        
        self.renderer.set_source(transform)
        
        with self.renderer:
            self.renderer.start()
            snippet = transform.render(0, 1000)
            
            # All values should be non-negative
            assert np.all(snippet.data >= 0)
    
    def test_square_function(self):
        """Test squaring transformation."""
        source = ConstantPE(2.0)
        transform = TransformPE(source, func=np.square)
        
        self.renderer.set_source(transform)
        
        with self.renderer:
            self.renderer.start()
            snippet = transform.render(0, 100)
            
            # 2^2 = 4
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 4.0, dtype=np.float32),
                decimal=4
            )
    
    def test_sqrt_function(self):
        """Test square root transformation."""
        source = ConstantPE(4.0)
        transform = TransformPE(source, func=np.sqrt)
        
        self.renderer.set_source(transform)
        
        with self.renderer:
            self.renderer.start()
            snippet = transform.render(0, 100)
            
            # sqrt(4) = 2
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 2.0, dtype=np.float32),
                decimal=4
            )
    
    def test_tanh_soft_clip(self):
        """Test tanh for soft clipping."""
        source = ConstantPE(10.0)  # Large value
        transform = TransformPE(source, func=np.tanh, name="soft_clip")
        
        self.renderer.set_source(transform)
        
        with self.renderer:
            self.renderer.start()
            snippet = transform.render(0, 100)
            
            # tanh(10) ≈ 1.0
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 1.0, dtype=np.float32),
                decimal=4
            )
    
    def test_lambda_function(self):
        """Test lambda transformation."""
        source = ConstantPE(3.0)
        transform = TransformPE(source, func=lambda x: x * 2 + 1)
        
        self.renderer.set_source(transform)
        
        with self.renderer:
            self.renderer.start()
            snippet = transform.render(0, 100)
            
            # 3 * 2 + 1 = 7
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 7.0, dtype=np.float32),
                decimal=4
            )


class TestTransformPEWithConversions:
    """Test TransformPE with conversion functions."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_ratio_to_db_transform(self):
        """Test using ratio_to_db as transform."""
        source = ConstantPE(10.0)  # Amplitude ratio
        transform = TransformPE(source, func=ratio_to_db, name="to_dB")
        
        self.renderer.set_source(transform)
        
        with self.renderer:
            self.renderer.start()
            snippet = transform.render(0, 100)
            
            # ratio_to_db(10) = 20 dB
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 20.0, dtype=np.float32),
                decimal=2
            )
    
    def test_pitch_to_freq_transform(self):
        """Test using pitch_to_freq as transform."""
        source = ConstantPE(69.0)  # MIDI note A4
        transform = TransformPE(source, func=pitch_to_freq, name="pitch_to_Hz")
        
        self.renderer.set_source(transform)
        
        with self.renderer:
            self.renderer.start()
            snippet = transform.render(0, 100)
            
            # pitch_to_freq(69) = 440 Hz
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 440.0, dtype=np.float32),
                decimal=2
            )
    
    def test_varying_pitch_to_freq(self):
        """Test converting varying pitch to frequency."""
        # Ramp from MIDI 60 to 72 (C4 to C5)
        source = PiecewisePE([(0, 60.0), (1000, 72.0)])
        transform = TransformPE(source, func=pitch_to_freq)
        
        self.renderer.set_source(transform)
        
        with self.renderer:
            self.renderer.start()
            snippet = transform.render(0, 1000)
            
            # First sample should be ~261.6 Hz (C4)
            assert snippet.data[0, 0] == pytest.approx(261.6, rel=0.01)
            
            # Last sample should be ~523.3 Hz (C5)
            assert snippet.data[-1, 0] == pytest.approx(523.3, rel=0.01)
            
            # Should be monotonically increasing
            assert np.all(np.diff(snippet.data[:, 0]) > 0)


class TestTransformPEStereo:
    """Test stereo signal handling."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_stereo_transform(self):
        """Transform should work on all channels."""
        source = ConstantPE(2.0, channels=2)
        transform = TransformPE(source, func=np.square)
        
        self.renderer.set_source(transform)
        
        with self.renderer:
            self.renderer.start()
            snippet = transform.render(0, 100)
            
            assert snippet.channels == 2
            np.testing.assert_array_almost_equal(
                snippet.data[:, 0],
                np.full(100, 4.0, dtype=np.float32),
                decimal=4
            )
            np.testing.assert_array_almost_equal(
                snippet.data[:, 1],
                np.full(100, 4.0, dtype=np.float32),
                decimal=4
            )


class TestTransformPEChaining:
    """Test chaining multiple transforms."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_chained_transforms(self):
        """Test chaining multiple TransformPE instances."""
        source = ConstantPE(-3.0)
        
        # abs(-3) = 3, then 3^2 = 9
        step1 = TransformPE(source, func=np.abs)
        step2 = TransformPE(step1, func=np.square)
        
        self.renderer.set_source(step2)
        
        with self.renderer:
            self.renderer.start()
            snippet = step2.render(0, 100)
            
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 9.0, dtype=np.float32),
                decimal=4
            )
