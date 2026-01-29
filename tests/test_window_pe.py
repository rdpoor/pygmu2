"""
Tests for WindowPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    WindowPE,
    WindowMode,
    ConstantPE,
    SinePE,
    DiracPE,
    RampPE,
    NullRenderer,
)


class TestWindowPEBasics:
    """Test basic WindowPE creation and properties."""
    
    def test_create_default(self):
        source = ConstantPE(1.0)
        win = WindowPE(source)
        
        assert win.source is source
        assert win.window == 0.05
        assert win.mode == WindowMode.MAX
        assert win.rectify is True
    
    def test_create_with_params(self):
        source = ConstantPE(1.0)
        win = WindowPE(
            source,
            window=0.02,
            mode=WindowMode.RMS,
            rectify=False,
        )
        
        assert win.window == 0.02
        assert win.mode == WindowMode.RMS
        assert win.rectify is False
    
    def test_negative_window_clamped(self):
        """Negative window should be clamped to 0."""
        source = ConstantPE(1.0)
        win = WindowPE(source, window=-0.1)
        
        assert win.window == 0.0
    
    def test_inputs(self):
        source = ConstantPE(1.0)
        win = WindowPE(source)
        
        assert win.inputs() == [source]
    
    def test_is_pure(self):
        """WindowPE is stateless, so is_pure() should return True."""
        source = ConstantPE(1.0)
        win = WindowPE(source)
        
        assert win.is_pure() is True
    
    def test_channel_count_passthrough(self):
        source = ConstantPE(1.0, channels=2)
        win = WindowPE(source)
        
        assert win.channel_count() == 2
    
    def test_extent_from_source(self):
        source = RampPE(0.0, 1.0, duration=1000)
        win = WindowPE(source)
        
        extent = win.extent()
        assert extent.start == 0
        assert extent.end == 1000
    
    def test_repr(self):
        source = ConstantPE(1.0)
        win = WindowPE(source, window=0.02, mode=WindowMode.RMS)
        
        repr_str = repr(win)
        assert "WindowPE" in repr_str
        assert "ConstantPE" in repr_str
        assert "0.02" in repr_str
        assert "rms" in repr_str


class TestWindowPEMax:
    """Test MAX mode."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_max_of_constant(self):
        """MAX of constant should equal the constant."""
        source = ConstantPE(0.5)
        win = WindowPE(source, window=0.01, mode=WindowMode.MAX)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            snippet = win.render(0, 100)
            
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 0.5, dtype=np.float32),
                decimal=4
            )
    
    def test_max_captures_peak(self):
        """MAX should capture the peak value in the window."""
        source = DiracPE()  # 1.0 at sample 0, 0.0 elsewhere
        # Window of ~23ms = ~1000 samples
        win = WindowPE(source, window=0.023, mode=WindowMode.MAX)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            snippet = win.render(0, 1000)
            
            # Near sample 0, MAX should include the impulse
            # The window extends before and after each sample
            assert snippet.data[0, 0] == 1.0
            
            # Samples within half-window of the impulse should also be 1.0
            half_window_samples = int(0.023 * 44100 / 2)
            for i in range(min(half_window_samples, 100)):
                assert snippet.data[i, 0] == 1.0


class TestWindowPEMean:
    """Test MEAN mode."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_mean_of_constant(self):
        """MEAN of constant should equal the constant."""
        source = ConstantPE(0.7)
        win = WindowPE(source, window=0.01, mode=WindowMode.MEAN)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            snippet = win.render(0, 100)
            
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 0.7, dtype=np.float32),
                decimal=4
            )
    
    def test_mean_smoothes_impulse(self):
        """MEAN should spread out an impulse over the window."""
        source = DiracPE()
        win = WindowPE(source, window=0.01, mode=WindowMode.MEAN)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            snippet = win.render(0, 1000)
            
            # The impulse should be spread over the window
            # Peak value should be 1/window_samples
            window_samples = int(0.01 * 44100) + 1
            expected_peak = 1.0 / window_samples
            
            # The peak should be near sample 0
            assert snippet.data[0, 0] == pytest.approx(expected_peak, rel=0.1)


class TestWindowPERMS:
    """Test RMS mode."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_rms_of_constant(self):
        """RMS of constant should equal the constant (for positive values)."""
        source = ConstantPE(0.6)
        win = WindowPE(source, window=0.01, mode=WindowMode.RMS)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            snippet = win.render(0, 100)
            
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 0.6, dtype=np.float32),
                decimal=4
            )
    
    def test_rms_of_sine(self):
        """RMS of sine wave should be amplitude / sqrt(2)."""
        amplitude = 1.0
        source = SinePE(frequency=440.0, amplitude=amplitude)
        # Use larger window to average over multiple cycles
        win = WindowPE(source, window=0.05, mode=WindowMode.RMS)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            # Skip initial transient, render in steady state
            snippet = win.render(4410, 1000)
            
            expected_rms = amplitude / np.sqrt(2)
            mean_rms = np.mean(snippet.data)
            
            assert mean_rms == pytest.approx(expected_rms, rel=0.05)


class TestWindowPEMin:
    """Test MIN mode."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_min_of_constant(self):
        """MIN of constant should equal the constant."""
        source = ConstantPE(0.4)
        win = WindowPE(source, window=0.01, mode=WindowMode.MIN)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            snippet = win.render(0, 100)
            
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 0.4, dtype=np.float32),
                decimal=4
            )
    
    def test_min_finds_zero(self):
        """MIN should find zero values in the window."""
        source = DiracPE()  # 1.0 at sample 0, 0.0 elsewhere
        win = WindowPE(source, window=0.01, mode=WindowMode.MIN)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            snippet = win.render(0, 1000)
            
            # MIN should be 0 everywhere (since most samples are 0)
            # except possibly at the very edges
            assert np.all(snippet.data[10:, 0] == 0.0)


class TestWindowPERectify:
    """Test rectification option."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_rectify_true(self):
        """With rectify=True, negative values become positive."""
        source = SinePE(frequency=440.0, amplitude=1.0)
        win = WindowPE(source, window=0.05, mode=WindowMode.MEAN, rectify=True)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            snippet = win.render(0, 1000)
            
            # All values should be non-negative
            assert np.all(snippet.data >= 0)
    
    def test_rectify_false(self):
        """With rectify=False, negative values are preserved."""
        source = SinePE(frequency=440.0, amplitude=1.0)
        win = WindowPE(source, window=0.0001, mode=WindowMode.MEAN, rectify=False)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            snippet = win.render(0, 1000)
            
            # Should have both positive and negative values
            assert np.any(snippet.data < 0)
            assert np.any(snippet.data > 0)


class TestWindowPEBidirectional:
    """Test bidirectional (zero-phase) behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_symmetric_response(self):
        """Window should be centered, producing symmetric response to impulse."""
        source = DiracPE()
        win = WindowPE(source, window=0.02, mode=WindowMode.MAX)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            snippet = win.render(0, 2000)
            
            # Find the extent of the "plateau" where MAX is 1.0
            ones = (snippet.data[:, 0] == 1.0)
            first_one = np.argmax(ones)
            last_one = len(ones) - 1 - np.argmax(ones[::-1])
            
            # The plateau should be centered around sample 0
            # (first_one should be 0, last_one should be ~half_window)
            assert first_one == 0
            
            half_window_samples = int(0.02 * 44100 / 2)
            assert last_one == pytest.approx(half_window_samples, abs=2)


class TestWindowPEStereo:
    """Test stereo signal handling."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_stereo_independent_channels(self):
        """Each stereo channel should be processed independently."""
        source = ConstantPE(0.8, channels=2)
        win = WindowPE(source, window=0.01, mode=WindowMode.MEAN)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            snippet = win.render(0, 100)
            
            assert snippet.channels == 2
            np.testing.assert_array_almost_equal(
                snippet.data[:, 0],
                snippet.data[:, 1],
                decimal=5
            )


class TestWindowPENoState:
    """Test that WindowPE is truly stateless."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_render_order_independent(self):
        """Render results should not depend on render order."""
        source = SinePE(frequency=100.0, amplitude=1.0)
        win = WindowPE(source, window=0.01, mode=WindowMode.MAX)
        
        self.renderer.set_source(win)
        
        with self.renderer:
            self.renderer.start()
            
            # Render blocks in different orders
            snippet_a = win.render(500, 100)
            snippet_b = win.render(0, 100)
            snippet_c = win.render(500, 100)  # Same as snippet_a
            
            # snippet_a and snippet_c should be identical
            np.testing.assert_array_equal(snippet_a.data, snippet_c.data)
