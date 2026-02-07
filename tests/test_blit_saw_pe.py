"""
Tests for BlitSawPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    BlitSawPE,
    ConstantPE,
    CropPE,
    PiecewisePE,
    NullRenderer,
    Extent,
)


class TestBlitSawPEBasics:
    """Test basic BlitSawPE creation and properties."""
    
    def test_create_with_frequency(self):
        """Test BlitSawPE creation with frequency."""
        saw = BlitSawPE(frequency=440.0)
        assert saw.frequency == 440.0
        assert saw.amplitude == 1.0
        assert saw.m is None  # Auto by default
        assert saw.leak == 0.999
        assert saw.channel_count() == 1
    
    def test_create_with_all_params(self):
        """Test BlitSawPE creation with all parameters."""
        saw = BlitSawPE(
            frequency=220.0,
            amplitude=0.5,
            m=15,
            leak=0.995,
            channels=2
        )
        assert saw.frequency == 220.0
        assert saw.amplitude == 0.5
        assert saw.m == 15
        assert saw.leak == 0.995
        assert saw.channel_count() == 2
    
    def test_infinite_extent_constant_params(self):
        """Test that constant params give infinite extent."""
        saw = BlitSawPE(frequency=440.0)
        extent = saw.extent()
        assert extent.start is None
        assert extent.end is None
    
    def test_is_never_pure(self):
        """BlitSawPE is never pure due to integrator state."""
        saw = BlitSawPE(frequency=440.0)
        assert saw.is_pure() is False
        
        # Even with PE inputs, still not pure
        freq_pe = ConstantPE(440.0)
        saw_pe = BlitSawPE(frequency=freq_pe)
        assert saw_pe.is_pure() is False
    
    def test_no_inputs_with_constants(self):
        """inputs() returns empty list with constant params."""
        saw = BlitSawPE(frequency=440.0)
        assert saw.inputs() == []
    
    def test_repr_with_constants(self):
        """Test repr with constant parameters."""
        saw = BlitSawPE(frequency=440.0, amplitude=0.5, m=10)
        repr_str = repr(saw)
        assert "BlitSawPE" in repr_str
        assert "440.0" in repr_str
        assert "0.5" in repr_str
        assert "10" in repr_str
    
    def test_repr_with_auto_m(self):
        """Test repr shows 'auto' for m=None."""
        saw = BlitSawPE(frequency=440.0)
        repr_str = repr(saw)
        assert "m=auto" in repr_str

    def test_extent_with_disjoint_pe_inputs_does_not_crash(self):
        """
        Regression: if PE inputs have disjoint extents, extent() should be
        a well-defined empty extent (start == end), not an exception.
        """
        freq = CropPE(ConstantPE(440.0), Extent(0, 10))
        amp = CropPE(ConstantPE(1.0), Extent(20, 30))  # disjoint from freq
        saw = BlitSawPE(frequency=freq, amplitude=amp)

        extent = saw.extent()
        assert extent.is_empty()


class TestBlitSawPERender:
    """Test BlitSawPE rendering with constant parameters."""
    
    def setup_method(self):
        """Create a renderer for configuring PEs."""
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_render_returns_snippet(self):
        """Test that render returns a properly shaped Snippet."""
        saw = BlitSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 100)
        assert snippet.start == 0
        assert snippet.duration == 100
        assert snippet.channels == 1
        
        self.renderer.stop()
    
    def test_render_stereo(self):
        """Test stereo output."""
        saw = BlitSawPE(frequency=440.0, channels=2)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 100)
        assert snippet.channels == 2
        # Both channels should be identical
        np.testing.assert_array_equal(
            snippet.data[:, 0],
            snippet.data[:, 1]
        )
        
        self.renderer.stop()
    
    def test_render_amplitude(self):
        """Test amplitude scaling."""
        saw = BlitSawPE(frequency=100.0, amplitude=0.5)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # Render several cycles to let integrator stabilize
        snippet = saw.render(0, 44100)  # One second
        
        # After stabilization, max should be close to 0.5
        # Check the latter half where waveform is stable
        stable_data = snippet.data[22050:, 0]
        max_val = np.max(np.abs(stable_data))
        
        # Should be in reasonable range (BLIT amplitude varies slightly)
        assert max_val > 0.3, f"Amplitude too low: {max_val}"
        assert max_val < 0.7, f"Amplitude too high: {max_val}"
        
        self.renderer.stop()
    
    def test_render_frequency_cycles(self):
        """Test that frequency produces approximately correct cycles."""
        sample_rate = 44100
        frequency = 441.0  # 441 cycles per second
        duration = sample_rate  # One second
        
        saw = BlitSawPE(frequency=frequency)
        renderer = NullRenderer(sample_rate=sample_rate)
        renderer.set_source(saw)
        renderer.start()
        
        snippet = saw.render(0, duration)
        data = snippet.data[:, 0]
        
        # Count zero crossings (positive-going)
        # Sawtooth has one positive-going zero crossing per cycle
        zero_crossings = 0
        for i in range(1, len(data)):
            if data[i-1] < 0 and data[i] >= 0:
                zero_crossings += 1
        
        # Should be close to frequency (within a few cycles tolerance)
        assert abs(zero_crossings - frequency) <= 5, \
            f"Expected ~{frequency} crossings, got {zero_crossings}"
        
        renderer.stop()
    
    def test_render_uses_global_sample_rate(self):
        """Render works with globally configured sample rate."""
        saw = BlitSawPE(frequency=440.0)
        snippet = saw.render(0, 100)
        assert snippet.duration == 100
    
    def test_render_continuity(self):
        """Test that consecutive renders are continuous."""
        saw = BlitSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # Render two consecutive chunks
        chunk1 = saw.render(0, 1000)
        chunk2 = saw.render(1000, 1000)
        
        # Should be continuous at boundary (no large jump)
        last_val = chunk1.data[-1, 0]
        first_val = chunk2.data[0, 0]
        
        # Sawtooth is continuous except at wraparound
        # Allow for small discontinuity due to BLIT artifacts
        diff = abs(first_val - last_val)
        assert diff < 0.5, f"Discontinuity too large: {diff}"
        
        self.renderer.stop()
    
    def test_render_negative_start(self):
        """Test rendering with negative start index."""
        saw = BlitSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(-100, 200)
        assert snippet.start == -100
        assert snippet.duration == 200
        
        self.renderer.stop()


class TestBlitSawPEAutoM:
    """Test automatic M computation."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_auto_m_low_frequency(self):
        """Low frequency should have many harmonics."""
        saw = BlitSawPE(frequency=100.0)  # 100 Hz
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # At 100 Hz with 44100 sample rate:
        # M < 44100 / (2 * 100) = 220.5
        # So M should be ~220 (odd)
        snippet = saw.render(0, 1000)
        
        # Just verify it renders without error
        assert snippet.duration == 1000
        
        self.renderer.stop()
    
    def test_auto_m_high_frequency(self):
        """High frequency should have few harmonics."""
        saw = BlitSawPE(frequency=10000.0)  # 10 kHz
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # At 10000 Hz with 44100 sample rate:
        # M < 44100 / (2 * 10000) = 2.205
        # So M should be ~1 or 3 (odd)
        snippet = saw.render(0, 1000)
        
        # Just verify it renders without error
        assert snippet.duration == 1000
        
        self.renderer.stop()
    
    def test_auto_m_near_nyquist(self):
        """Frequency near Nyquist should still work."""
        saw = BlitSawPE(frequency=20000.0)  # Near Nyquist
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # M should be 1 (minimum)
        snippet = saw.render(0, 1000)
        assert snippet.duration == 1000
        
        self.renderer.stop()


class TestBlitSawPEFixedM:
    """Test fixed M parameter."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_fixed_m_value(self):
        """Test that fixed M is used."""
        saw = BlitSawPE(frequency=440.0, m=10)
        assert saw.m == 10
        
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 1000)
        assert snippet.duration == 1000
        
        self.renderer.stop()
    
    def test_fixed_m_minimum_one(self):
        """M should be at least 1 even if set to 0."""
        saw = BlitSawPE(frequency=440.0, m=0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # Should not crash, M clamped to 1
        snippet = saw.render(0, 1000)
        assert snippet.duration == 1000
        
        self.renderer.stop()
    
    def test_fixed_m_affects_spectrum(self):
        """Different M values should produce different spectra."""
        saw_m5 = BlitSawPE(frequency=440.0, m=5)
        saw_m50 = BlitSawPE(frequency=440.0, m=50)
        
        renderer1 = NullRenderer(sample_rate=44100)
        renderer2 = NullRenderer(sample_rate=44100)
        
        renderer1.set_source(saw_m5)
        renderer2.set_source(saw_m50)
        
        renderer1.start()
        renderer2.start()
        
        snippet_m5 = saw_m5.render(0, 4410)
        snippet_m50 = saw_m50.render(0, 4410)
        
        # Waveforms should be different (more harmonics = sharper edges)
        # Compare RMS of difference
        diff = snippet_m5.data[:, 0] - snippet_m50.data[:, 0]
        rms_diff = np.sqrt(np.mean(diff**2))
        
        # Should have noticeable difference
        assert rms_diff > 0.01, "Different M values should produce different waveforms"
        
        renderer1.stop()
        renderer2.stop()


class TestBlitSawPEModulation:
    """Test BlitSawPE with PE inputs."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_frequency_from_pe(self):
        """Test using a PE for frequency input."""
        freq_pe = ConstantPE(440.0)
        saw = BlitSawPE(frequency=freq_pe)
        
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 1000)
        assert snippet.duration == 1000
        
        self.renderer.stop()
    
    def test_amplitude_from_pe(self):
        """Test using a PE for amplitude input."""
        amp_pe = ConstantPE(0.5)
        saw = BlitSawPE(frequency=440.0, amplitude=amp_pe)
        
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 44100)
        
        # Check amplitude is applied
        stable_data = snippet.data[22050:, 0]
        max_val = np.max(np.abs(stable_data))
        assert max_val < 0.7, "Amplitude PE should reduce output"
        
        self.renderer.stop()
    
    def test_m_from_pe(self):
        """Test using a PE for M input."""
        m_pe = ConstantPE(10.0)
        saw = BlitSawPE(frequency=440.0, m=m_pe)
        
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 1000)
        assert snippet.duration == 1000
        
        self.renderer.stop()
    
    def test_inputs_returns_pe_inputs(self):
        """inputs() should return all PE inputs."""
        freq_pe = ConstantPE(440.0)
        amp_pe = ConstantPE(0.5)
        m_pe = ConstantPE(10.0)
        
        saw = BlitSawPE(frequency=freq_pe, amplitude=amp_pe, m=m_pe)
        
        inputs = saw.inputs()
        assert len(inputs) == 3
        assert freq_pe in inputs
        assert amp_pe in inputs
        assert m_pe in inputs
    
    def test_extent_with_pe_inputs(self):
        """Extent should be intersection of PE input extents."""
        # Create a PE with finite extent
        finite_pe = PiecewisePE([(0, 0.0), (1000, 1.0)])
        saw = BlitSawPE(frequency=440.0, amplitude=finite_pe)
        
        extent = saw.extent()
        assert extent.start == 0
        assert extent.end == 1000
    
    def test_repr_with_pe_inputs(self):
        """repr should show PE class names."""
        freq_pe = ConstantPE(440.0)
        saw = BlitSawPE(frequency=freq_pe)
        
        repr_str = repr(saw)
        assert "ConstantPE" in repr_str


class TestBlitSawPEStateManagement:
    """Test state management (on_start, on_stop)."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_on_start_resets_state(self):
        """on_start() should reset phase and integrator."""
        saw = BlitSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # Render some samples
        chunk1 = saw.render(0, 1000)
        
        # Stop and restart
        self.renderer.stop()
        self.renderer.start()
        
        # Should produce same output as before (state reset)
        chunk2 = saw.render(0, 1000)
        np.testing.assert_array_almost_equal(
            chunk1.data, chunk2.data, decimal=5
        )
        
        self.renderer.stop()
    
    def test_state_persists_across_contiguous_renders(self):
        """State should persist for contiguous render calls."""
        saw = BlitSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # Render in two chunks
        chunk1 = saw.render(0, 500)
        chunk2 = saw.render(500, 500)
        
        # Reset and render as one chunk
        self.renderer.stop()
        self.renderer.start()
        full_chunk = saw.render(0, 1000)
        
        # Combined chunks should match single chunk
        combined = np.vstack([chunk1.data, chunk2.data])
        np.testing.assert_array_almost_equal(
            combined, full_chunk.data, decimal=5
        )
        
        self.renderer.stop()
    
    def test_discontinuous_render_raises(self):
        """Non-contiguous render raises for impure PE."""
        saw = BlitSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()

        saw.render(0, 1000)

        # Impure PEs require contiguous requests; skip-ahead is not allowed
        with pytest.raises(ValueError):
            saw.render(5000, 1000)

        self.renderer.stop()


class TestBlitSawPEDifferentSampleRates:
    """Test BlitSawPE at different sample rates."""
    
    def test_sample_rate_48000(self):
        """Test at 48kHz sample rate."""
        import pygmu2 as pg
        pg.set_sample_rate(48000)
        saw = BlitSawPE(frequency=480.0)
        renderer = NullRenderer(sample_rate=48000)
        renderer.set_source(saw)
        renderer.start()
        
        snippet = saw.render(0, 48000)  # One second
        data = snippet.data[:, 0]
        
        # Count zero crossings
        zero_crossings = sum(
            1 for i in range(1, len(data))
            if data[i-1] < 0 and data[i] >= 0
        )
        
        assert abs(zero_crossings - 480) <= 5
        
        renderer.stop()
    
    def test_sample_rate_22050(self):
        """Test at 22.05kHz sample rate."""
        import pygmu2 as pg
        pg.set_sample_rate(22050)
        saw = BlitSawPE(frequency=220.0)
        renderer = NullRenderer(sample_rate=22050)
        renderer.set_source(saw)
        renderer.start()
        
        snippet = saw.render(0, 22050)  # One second
        data = snippet.data[:, 0]
        
        # Count zero crossings
        zero_crossings = sum(
            1 for i in range(1, len(data))
            if data[i-1] < 0 and data[i] >= 0
        )
        
        assert abs(zero_crossings - 220) <= 5
        
        renderer.stop()


class TestBlitSawPELeakParameter:
    """Test leak parameter behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_different_leak_values(self):
        """Different leak values should produce different DC behavior."""
        saw_high_leak = BlitSawPE(frequency=100.0, leak=0.9999)
        saw_low_leak = BlitSawPE(frequency=100.0, leak=0.99)
        
        renderer1 = NullRenderer(sample_rate=44100)
        renderer2 = NullRenderer(sample_rate=44100)
        
        renderer1.set_source(saw_high_leak)
        renderer2.set_source(saw_low_leak)
        
        renderer1.start()
        renderer2.start()
        
        snippet_high = saw_high_leak.render(0, 44100)
        snippet_low = saw_low_leak.render(0, 44100)
        
        # Lower leak should have less DC offset accumulation
        # Check mean of latter half
        mean_high = np.mean(snippet_high.data[22050:, 0])
        mean_low = np.mean(snippet_low.data[22050:, 0])
        
        # Lower leak should have smaller DC offset
        assert abs(mean_low) <= abs(mean_high) + 0.1
        
        renderer1.stop()
        renderer2.stop()


class TestBlitSawPEEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_very_low_frequency(self):
        """Test with very low frequency (long period)."""
        saw = BlitSawPE(frequency=1.0)  # 1 Hz
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 44100)  # One second = one cycle
        assert snippet.duration == 44100
        
        self.renderer.stop()
    
    def test_zero_duration(self):
        """Test rendering zero samples returns empty snippet."""
        saw = BlitSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # Zero duration now allowed at PE level, returns empty snippet
        snippet = saw.render(0, 0)
        assert snippet.duration == 0
        assert snippet.channels == 1
        assert snippet.data.shape == (0, 1)
        
        self.renderer.stop()
    
    def test_single_sample(self):
        """Test rendering single sample."""
        saw = BlitSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 1)
        assert snippet.duration == 1
        assert snippet.data.shape == (1, 1)
        
        self.renderer.stop()
