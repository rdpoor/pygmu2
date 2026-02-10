"""
Tests for CompressorPE, LimiterPE, and GatePE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

from pygmu2 import (
    CompressorPE,
    LimiterPE,
    GatePE,
    ConstantPE,
    SinePE,
    GainPE,
    NullRenderer,
    DetectionMode,
)


class TestCompressorPEBasics:
    """Test basic CompressorPE creation and properties."""
    
    def test_create_default(self):
        """Test creating CompressorPE with default parameters."""
        source = SinePE(frequency=440.0)
        comp = CompressorPE(source)
        
        assert comp.threshold == -20.0
        assert comp.ratio == 4.0
        assert comp.attack == 0.01
        assert comp.release == 0.1
        assert comp.knee == 6.0
        assert comp.lookahead == 0.0
        assert comp.detection == DetectionMode.RMS
        assert comp.stereo_link is True
    
    def test_create_with_params(self):
        """Test creating CompressorPE with custom parameters."""
        source = SinePE(frequency=440.0)
        comp = CompressorPE(
            source,
            threshold=-15.0,
            ratio=8.0,
            attack=0.005,
            release=0.2,
            knee=12.0,
            makeup_gain=6.0,
            lookahead=0.003,
            detection=DetectionMode.PEAK,
            stereo_link=False,
        )
        
        assert comp.threshold == -15.0
        assert comp.ratio == 8.0
        assert comp.attack == 0.005
        assert comp.release == 0.2
        assert comp.knee == 12.0
        assert comp.makeup_gain == 6.0
        assert comp.lookahead == 0.003
        assert comp.detection == DetectionMode.PEAK
        assert comp.stereo_link is False
    
    def test_inputs(self):
        """Test that inputs() returns only source (internal PEs hidden)."""
        source = SinePE(frequency=440.0)
        comp = CompressorPE(source)
        
        inputs = comp.inputs()
        assert len(inputs) == 1
        assert source in inputs
    
    def test_is_not_pure(self):
        """CompressorPE is not pure due to envelope state."""
        source = SinePE(frequency=440.0)
        comp = CompressorPE(source)
        
        assert comp.is_pure() is False
    
    def test_channel_count_passthrough(self):
        """Channel count comes from source."""
        source = ConstantPE(1.0, channels=2)
        comp = CompressorPE(source)
        
        assert comp.channel_count() == 2
    
    def test_auto_makeup_gain(self):
        """Auto makeup gain is computed."""
        source = SinePE(frequency=440.0)
        comp = CompressorPE(source, threshold=-20, ratio=4, makeup_gain="auto")
        
        # Auto makeup should be positive
        assert comp.makeup_gain > 0
    
    def test_repr(self):
        """Test string representation."""
        source = SinePE(frequency=440.0)
        comp = CompressorPE(source, threshold=-20, ratio=4)
        
        repr_str = repr(comp)
        assert "CompressorPE" in repr_str
        assert "threshold=-20" in repr_str
        assert "ratio=4" in repr_str


class TestCompressorPERender:
    """Test CompressorPE rendering behavior."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_render_returns_snippet(self, renderer):
        """Render should return a valid Snippet."""
        source = SinePE(frequency=440.0)
        comp = CompressorPE(source)
        
        renderer.set_source(comp)
        renderer.start()
        
        snippet = comp.render(0, 1000)
        
        assert snippet.data.shape == (1000, 1)
        assert snippet.start == 0
    
    def test_render_stereo(self, renderer):
        """Stereo source should produce stereo output."""
        source = ConstantPE(1.0, channels=2)
        comp = CompressorPE(source)
        
        renderer.set_source(comp)
        renderer.start()
        
        snippet = comp.render(0, 1000)
        
        assert snippet.data.shape == (1000, 2)
    
    def test_compression_reduces_loud_signal(self, renderer):
        """Compression should reduce signal above threshold."""
        source = ConstantPE(1.0)  # 0dB
        comp = CompressorPE(source, threshold=-20, ratio=4, makeup_gain=0)
        
        renderer.set_source(comp)
        renderer.start()
        
        # Render enough samples for envelope to settle (attack = 0.01s = 441 samples)
        snippet = comp.render(0, 2000)
        
        # Check latter half after envelope settles
        latter_half = snippet.data[1000:, 0]
        assert np.all(latter_half < 0.5)
    
    def test_quiet_signal_passes_through(self, renderer):
        """Quiet signal (below threshold) should pass through."""
        source = ConstantPE(0.01)  # About -40dB
        comp = CompressorPE(source, threshold=-20, ratio=4, makeup_gain=0)
        
        renderer.set_source(comp)
        renderer.start()
        
        snippet = comp.render(0, 1000)
        
        # Should be approximately unchanged
        np.testing.assert_allclose(snippet.data, 0.01, rtol=0.1)
    
    def test_state_persists_across_renders(self, renderer):
        """Envelope state should persist across render calls."""
        source = SinePE(frequency=440.0)
        comp = CompressorPE(source, attack=0.1, release=0.5)
        
        renderer.set_source(comp)
        renderer.start()
        
        # First render
        snippet1 = comp.render(0, 4410)
        
        # Second render (continuous)
        snippet2 = comp.render(4410, 4410)
        
        # Outputs should be continuous (no discontinuity at boundary)
        # Check that last sample of snippet1 is close to first of snippet2
        diff = abs(snippet1.data[-1, 0] - snippet2.data[0, 0])
        assert diff < 0.1  # Reasonable continuity
    
    def test_zero_duration(self, renderer):
        """Zero duration should return empty snippet."""
        source = SinePE(frequency=440.0)
        comp = CompressorPE(source)
        
        renderer.set_source(comp)
        renderer.start()
        
        snippet = comp.render(0, 0)
        
        assert snippet.data.shape[0] == 0


class TestCompressorPELifecycle:
    """Test CompressorPE lifecycle methods."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_on_start_resets_state(self, renderer):
        """on_start should reset envelope state."""
        source = SinePE(frequency=440.0)
        comp = CompressorPE(source)
        
        renderer.set_source(comp)
        renderer.start()
        
        # Render to build up state
        comp.render(0, 44100)
        
        # Restart
        renderer.stop()
        renderer.start()
        
        # First sample after restart should be as if from fresh start
        snippet = comp.render(0, 100)
        assert snippet.data.shape == (100, 1)


class TestLimiterPE:
    """Test LimiterPE convenience class."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_create_default(self):
        """Test creating LimiterPE with defaults."""
        source = SinePE(frequency=440.0)
        limiter = LimiterPE(source)
        
        assert limiter.ceiling == -1.0
        assert limiter.release == 0.05
        assert limiter.lookahead == 0.005
    
    def test_create_with_params(self):
        """Test creating LimiterPE with custom params."""
        source = SinePE(frequency=440.0)
        limiter = LimiterPE(source, ceiling=-3.0, release=0.1, lookahead=0.01)
        
        assert limiter.ceiling == -3.0
        assert limiter.release == 0.1
        assert limiter.lookahead == 0.01
    
    def test_limiter_caps_output(self, renderer):
        """Limiter should cap output at ceiling."""
        source = ConstantPE(1.0)  # 0dB input
        limiter = LimiterPE(source, ceiling=-6.0)
        
        renderer.set_source(limiter)
        renderer.start()
        
        # Render enough samples for envelope to settle
        snippet = limiter.render(0, 2000)
        
        # Check latter half after envelope settles
        # Output should be approximately at ceiling (-6dB ≈ 0.5)
        latter_half = snippet.data[1000:, 0]
        assert np.all(latter_half < 0.6)  # Allow some tolerance
    
    def test_repr(self):
        """Test string representation."""
        source = SinePE(frequency=440.0)
        limiter = LimiterPE(source, ceiling=-3.0)
        
        repr_str = repr(limiter)
        assert "LimiterPE" in repr_str
        assert "ceiling=-3.0" in repr_str


class TestGatePE:
    """Test GatePE convenience class."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_create_default(self):
        """Test creating GatePE with defaults."""
        source = SinePE(frequency=440.0)
        gate = GatePE(source)
        
        assert gate.threshold == -40.0
        assert gate.attack == 0.001
        assert gate.release == 0.05
        assert gate.gate_range == -80.0

    def test_create_with_params(self):
        """Test creating GatePE with custom params."""
        source = SinePE(frequency=440.0)
        gate = GatePE(source, threshold=-30.0, attack=0.0005, release=0.1, gate_range=-60.0)

        assert gate.threshold == -30.0
        assert gate.attack == 0.0005
        assert gate.release == 0.1
        assert gate.gate_range == -60.0
    
    def test_gate_silences_quiet_signal(self, renderer):
        """Gate should silence signal below threshold."""
        source = ConstantPE(0.001)  # -60dB, below -40dB threshold
        gate = GatePE(source, threshold=-40, gate_range=-80)
        
        renderer.set_source(gate)
        renderer.start()
        
        snippet = gate.render(0, 1000)
        
        # Should be heavily attenuated
        assert np.all(np.abs(snippet.data) < 0.0001)
    
    def test_gate_passes_loud_signal(self, renderer):
        """Gate should pass signal above threshold."""
        source = ConstantPE(0.5)  # -6dB, above -40dB threshold
        gate = GatePE(source, threshold=-40)
        
        renderer.set_source(gate)
        renderer.start()
        
        snippet = gate.render(0, 1000)
        
        # Should pass through
        np.testing.assert_allclose(snippet.data, 0.5, rtol=0.1)
    
    def test_is_not_pure(self):
        """GatePE is not pure due to envelope state."""
        source = SinePE(frequency=440.0)
        gate = GatePE(source)
        
        assert gate.is_pure() is False
    
    def test_repr(self):
        """Test string representation."""
        source = SinePE(frequency=440.0)
        gate = GatePE(source, threshold=-30.0)
        
        repr_str = repr(gate)
        assert "GatePE" in repr_str
        assert "threshold=-30.0" in repr_str


class TestCompressorPEWithSine:
    """Test CompressorPE with actual sine wave signals."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_compressor_on_sine(self, renderer):
        """Compressor should reduce peaks of sine wave."""
        source = SinePE(frequency=100.0, amplitude=1.0)
        comp = CompressorPE(source, threshold=-6, ratio=4, makeup_gain=0)
        
        renderer.set_source(comp)
        renderer.start()
        
        # Render several cycles, allow envelope to settle
        snippet = comp.render(0, 8820)
        
        # Check latter half after envelope settles
        latter_half = snippet.data[4410:, 0]
        
        # Peak amplitude should be reduced
        assert np.max(np.abs(latter_half)) < 0.9
    
    def test_limiter_on_sine(self, renderer):
        """Limiter should prevent sine from exceeding ceiling."""
        source = SinePE(frequency=100.0, amplitude=1.0)
        limiter = LimiterPE(source, ceiling=-3.0)
        
        renderer.set_source(limiter)
        renderer.start()
        
        # Render several cycles (allow envelope to settle)
        snippet = limiter.render(0, 8820)
        
        # After settling, peaks should be near ceiling (-3dB ≈ 0.71)
        # Check latter half after envelope settles
        latter_half = snippet.data[4410:, 0]
        assert np.max(np.abs(latter_half)) < 0.85
