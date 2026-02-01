"""
Tests for SinePE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    SinePE,
    ConstantPE,
    CropPE,
    PiecewisePE,
    NullRenderer,
    Extent,
    SourcePE,
    ProcessingElement,
    Snippet,
)


class TestSinePEBasics:
    """Test basic SinePE creation and properties."""
    
    def test_create_sine_pe_defaults(self):
        """Test SinePE with all defaults."""
        sine = SinePE()
        assert sine.frequency == 440.0  # Default frequency
        assert sine.amplitude == 1.0
        assert sine.initial_phase == 0.0
        assert sine.channel_count() == 1
    
    def test_create_sine_pe(self):
        sine = SinePE(frequency=440.0)
        assert sine.frequency == 440.0
        assert sine.amplitude == 1.0
        assert sine.initial_phase == 0.0
        assert sine.channel_count() == 1
    
    def test_create_with_all_params(self):
        sine = SinePE(frequency=880.0, amplitude=0.5, phase=np.pi/2, channels=2)
        assert sine.frequency == 880.0
        assert sine.amplitude == 0.5
        assert sine.initial_phase == np.pi/2
        assert sine.channel_count() == 2
    
    def test_infinite_extent_constant_params(self):
        sine = SinePE(frequency=440.0)
        extent = sine.extent()
        assert extent.start is None
        assert extent.end is None
    
    def test_is_pure_with_constants(self):
        sine = SinePE(frequency=440.0)
        assert sine.is_pure() is True
    
    def test_no_inputs_with_constants(self):
        sine = SinePE(frequency=440.0)
        assert sine.inputs() == []
    
    def test_repr_constants(self):
        sine = SinePE(frequency=440.0, amplitude=0.5)
        repr_str = repr(sine)
        assert "SinePE" in repr_str
        assert "440.0" in repr_str
        assert "0.5" in repr_str

    def test_extent_with_disjoint_pe_inputs_does_not_crash(self):
        """
        Regression: if PE inputs have disjoint extents, extent() should be
        a well-defined empty extent (start == end), not an exception.
        """
        freq = CropPE(ConstantPE(440.0), Extent(0, 10))
        amp = CropPE(ConstantPE(1.0), Extent(20, 30))  # disjoint from freq
        sine = SinePE(frequency=freq, amplitude=amp)

        extent = sine.extent()
        assert extent.is_empty()


class TestSinePERender:
    """Test SinePE rendering with constant parameters."""
    
    def setup_method(self):
        """Create a renderer for configuring PEs."""
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_render_returns_snippet(self):
        sine = SinePE(frequency=440.0)
        self.renderer.set_source(sine)
        
        snippet = sine.render(0, 100)
        assert snippet.start == 0
        assert snippet.duration == 100
        assert snippet.channels == 1
    
    def test_render_stereo(self):
        sine = SinePE(frequency=440.0, channels=2)
        self.renderer.set_source(sine)
        
        snippet = sine.render(0, 100)
        assert snippet.channels == 2
        # Both channels should be identical
        np.testing.assert_array_equal(
            snippet.data[:, 0],
            snippet.data[:, 1]
        )
    
    def test_render_amplitude(self):
        sine = SinePE(frequency=440.0, amplitude=0.5)
        self.renderer.set_source(sine)
        
        snippet = sine.render(0, 44100)  # One second
        # Max should be close to 0.5
        assert np.max(np.abs(snippet.data)) <= 0.5 + 1e-6
        assert np.max(np.abs(snippet.data)) >= 0.5 - 1e-3  # Should reach near peak
    
    def test_render_frequency(self):
        """Test that frequency produces correct number of cycles."""
        sample_rate = 44100
        frequency = 441.0  # 441 Hz = 441 cycles per second
        duration = sample_rate  # One second of samples
        
        sine = SinePE(frequency=frequency)
        renderer = NullRenderer(sample_rate=sample_rate)
        renderer.set_source(sine)
        
        snippet = sine.render(0, duration)
        data = snippet.data[:, 0]
        
        # Count zero crossings (positive-going)
        zero_crossings = 0
        for i in range(1, len(data)):
            if data[i-1] < 0 and data[i] >= 0:
                zero_crossings += 1
        
        # Should have approximately 'frequency' zero crossings in one second
        assert abs(zero_crossings - frequency) <= 1
    
    def test_render_phase(self):
        """Test that phase offset works correctly."""
        sine_no_phase = SinePE(frequency=440.0, phase=0.0)
        sine_with_phase = SinePE(frequency=440.0, phase=np.pi/2)
        
        self.renderer.set_source(sine_no_phase)
        snippet_no_phase = sine_no_phase.render(0, 10)
        
        # Need separate renderer for second PE
        renderer2 = NullRenderer(sample_rate=44100)
        renderer2.set_source(sine_with_phase)
        snippet_with_phase = sine_with_phase.render(0, 10)
        
        # At sample 0, sin(0) = 0, sin(pi/2) = 1
        assert abs(snippet_no_phase.data[0, 0] - 0.0) < 1e-6
        assert abs(snippet_with_phase.data[0, 0] - 1.0) < 1e-6
    
    def test_render_continuity_pure(self):
        """Test that consecutive renders are continuous for pure PE."""
        sine = SinePE(frequency=440.0)
        self.renderer.set_source(sine)
        
        # Render two consecutive chunks
        chunk1 = sine.render(0, 1000)
        chunk2 = sine.render(1000, 1000)
        
        # Last sample of chunk1 and first sample of chunk2 should be continuous
        last_val = chunk1.data[-1, 0]
        first_val = chunk2.data[0, 0]
        
        # No large jump
        assert abs(first_val - last_val) < 0.1
    
    def test_render_negative_start(self):
        """Test rendering with negative start index."""
        sine = SinePE(frequency=440.0)
        self.renderer.set_source(sine)
        
        snippet = sine.render(-100, 200)
        assert snippet.start == -100
        assert snippet.duration == 200
    
    def test_render_requires_configuration(self):
        """Test that render fails if not configured."""
        sine = SinePE(frequency=440.0)
        with pytest.raises(RuntimeError, match="sample_rate accessed before configuration"):
            sine.render(0, 100)


class TestSinePEModulation:
    """Test SinePE with PE inputs for modulation."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_frequency_from_pe(self):
        """Test using a PE for frequency input."""
        freq_pe = ConstantPE(440.0)
        sine = SinePE(frequency=freq_pe)
        
        self.renderer.set_source(sine)
        self.renderer.start()
        
        snippet = sine.render(0, 100)
        assert snippet.duration == 100
        
        self.renderer.stop()
    
    def test_amplitude_from_pe(self):
        """Test using a PE for amplitude input (AM synthesis)."""
        amp_pe = ConstantPE(0.5)
        sine = SinePE(frequency=440.0, amplitude=amp_pe)
        
        self.renderer.set_source(sine)
        self.renderer.start()
        
        snippet = sine.render(0, 44100)
        # Max should be close to 0.5
        assert np.max(np.abs(snippet.data)) <= 0.5 + 1e-6
        
        self.renderer.stop()
    
    def test_phase_from_pe(self):
        """Test using a PE for phase modulation (PM synthesis)."""
        # Constant phase offset via PE
        phase_pe = ConstantPE(np.pi/2)
        sine = SinePE(frequency=440.0, phase=phase_pe)
        
        self.renderer.set_source(sine)
        self.renderer.start()
        
        snippet = sine.render(0, 10)
        # First sample should be close to sin(0 + pi/2) = 1
        # But with stateful phase computation, the initial accumulated phase is 0,
        # and phase_mod is added. So first value should be sin(phase_increment[0] + pi/2)
        # which is approximately sin(pi/2) ≈ 1 for small phase_increment
        assert abs(snippet.data[0, 0] - 1.0) < 0.1
        
        self.renderer.stop()
    
    def test_is_not_pure_with_pe_frequency(self):
        """SinePE is non-pure when frequency is a PE."""
        freq_pe = ConstantPE(440.0)
        sine = SinePE(frequency=freq_pe)
        assert sine.is_pure() is False
    
    def test_is_not_pure_with_pe_amplitude(self):
        """SinePE is non-pure when amplitude is a PE."""
        amp_pe = ConstantPE(0.5)
        sine = SinePE(frequency=440.0, amplitude=amp_pe)
        assert sine.is_pure() is False
    
    def test_is_not_pure_with_pe_phase(self):
        """SinePE is non-pure when phase is a PE."""
        phase_pe = ConstantPE(0.0)
        sine = SinePE(frequency=440.0, phase=phase_pe)
        assert sine.is_pure() is False
    
    def test_inputs_returns_pe_inputs(self):
        """inputs() should return all PE inputs."""
        freq_pe = ConstantPE(440.0)
        amp_pe = ConstantPE(0.5)
        sine = SinePE(frequency=freq_pe, amplitude=amp_pe)
        
        inputs = sine.inputs()
        assert len(inputs) == 2
        assert freq_pe in inputs
        assert amp_pe in inputs
    
    def test_extent_with_pe_inputs(self):
        """Extent should be intersection of PE input extents."""
        # Create a PE with finite extent
        finite_pe = PiecewisePE([(0, 0.0), (1000, 1.0)])
        sine = SinePE(frequency=440.0, amplitude=finite_pe)
        
        extent = sine.extent()
        assert extent.start == 0
        assert extent.end == 1000
    
    def test_repr_with_pe_inputs(self):
        """repr should show PE class names for PE inputs."""
        freq_pe = ConstantPE(440.0)
        sine = SinePE(frequency=freq_pe)
        
        repr_str = repr(sine)
        assert "ConstantPE" in repr_str
    
    def test_fm_synthesis_continuity(self):
        """Test FM synthesis produces continuous output across chunks."""
        freq_pe = ConstantPE(440.0)
        sine = SinePE(frequency=freq_pe)
        
        self.renderer.set_source(sine)
        self.renderer.start()
        
        # Render consecutive chunks
        chunk1 = sine.render(0, 1000)
        chunk2 = sine.render(1000, 1000)
        
        # Should be continuous at boundary
        last_val = chunk1.data[-1, 0]
        first_val = chunk2.data[0, 0]
        assert abs(first_val - last_val) < 0.1
        
        self.renderer.stop()
    
    def test_on_start_resets_phase(self):
        """on_start() should reset phase accumulator."""
        freq_pe = ConstantPE(440.0)
        sine = SinePE(frequency=freq_pe)
        
        self.renderer.set_source(sine)
        self.renderer.start()
        
        # Render some samples
        chunk1 = sine.render(0, 1000)
        
        # Stop and restart
        self.renderer.stop()
        self.renderer.start()
        
        # Should produce same output as before
        chunk2 = sine.render(0, 1000)
        np.testing.assert_array_almost_equal(chunk1.data, chunk2.data, decimal=5)
        
        self.renderer.stop()


class TestSinePEDifferentSampleRates:
    """Test SinePE at different sample rates."""
    
    def test_sample_rate_48000(self):
        sine = SinePE(frequency=480.0)  # 480 cycles per second
        renderer = NullRenderer(sample_rate=48000)
        renderer.set_source(sine)
        
        snippet = sine.render(0, 48000)  # One second
        data = snippet.data[:, 0]
        
        # Count zero crossings
        zero_crossings = sum(
            1 for i in range(1, len(data))
            if data[i-1] < 0 and data[i] >= 0
        )
        
        assert abs(zero_crossings - 480) <= 1
    
    def test_sample_rate_22050(self):
        sine = SinePE(frequency=220.5)  # 220.5 cycles per second
        renderer = NullRenderer(sample_rate=22050)
        renderer.set_source(sine)
        
        snippet = sine.render(0, 22050)  # One second
        data = snippet.data[:, 0]
        
        # Count zero crossings
        zero_crossings = sum(
            1 for i in range(1, len(data))
            if data[i-1] < 0 and data[i] >= 0
        )
        
        # Should be close to 220 or 221
        assert abs(zero_crossings - 220.5) <= 1


class TestSinePEVibrato:
    """Test SinePE with time-varying frequency (vibrato/FM)."""
    
    def test_vibrato_basic(self):
        """Test basic vibrato (frequency modulation with LFO)."""
        # Create an LFO that modulates frequency
        # Base frequency 440 Hz, vibrato ±20 Hz at 5 Hz rate
        lfo = SinePE(frequency=5.0, amplitude=20.0)  # 5 Hz, ±20 Hz
        
        # We need a way to add 440 + lfo
        # For now, just test that the PE accepts SinePE as frequency input
        # (Full FM would need an AddPE or similar)
        
        # Create renderer and configure
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(lfo)
        renderer.start()
        
        # Verify LFO produces expected values
        snippet = lfo.render(0, 44100)
        assert np.max(snippet.data) <= 20.0 + 1e-6
        assert np.min(snippet.data) >= -20.0 - 1e-6
        
        renderer.stop()
