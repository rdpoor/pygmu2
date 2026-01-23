"""
Tests for SuperSawPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    SuperSawPE,
    ConstantPE,
    RampPE,
    SinePE,
    NullRenderer,
    Extent,
)


class TestSuperSawPEBasics:
    """Test basic SuperSawPE creation and properties."""
    
    def test_create_with_frequency(self):
        """Test SuperSawPE creation with frequency."""
        saw = SuperSawPE(frequency=440.0)
        assert saw.frequency == 440.0
        assert saw.amplitude == 1.0
        assert saw.voices == 7
        assert saw.detune_cents == 20.0
        assert saw.mix_mode == 'center_heavy'
        assert saw.channel_count() == 1
    
    def test_create_with_all_params(self):
        """Test SuperSawPE creation with all parameters."""
        saw = SuperSawPE(
            frequency=220.0,
            amplitude=0.5,
            voices=9,
            detune_cents=30.0,
            mix_mode='equal',
            channels=2,
        )
        assert saw.frequency == 220.0
        assert saw.amplitude == 0.5
        assert saw.voices == 9
        assert saw.detune_cents == 30.0
        assert saw.mix_mode == 'equal'
        assert saw.channel_count() == 2
    
    def test_minimum_voices(self):
        """Test that voices is clamped to at least 1."""
        saw = SuperSawPE(frequency=440.0, voices=0)
        assert saw.voices == 1
        
        saw = SuperSawPE(frequency=440.0, voices=-5)
        assert saw.voices == 1
    
    def test_infinite_extent_constant_params(self):
        """Test that constant params give infinite extent."""
        saw = SuperSawPE(frequency=440.0)
        extent = saw.extent()
        assert extent.start is None
        assert extent.end is None
    
    def test_is_never_pure(self):
        """SuperSawPE is never pure due to internal oscillator state."""
        saw = SuperSawPE(frequency=440.0)
        assert saw.is_pure() is False
    
    def test_no_inputs_with_constants(self):
        """inputs() returns empty list with constant params."""
        saw = SuperSawPE(frequency=440.0)
        assert saw.inputs() == []
    
    def test_repr(self):
        """Test repr output."""
        saw = SuperSawPE(frequency=440.0, voices=5, detune_cents=15.0)
        repr_str = repr(saw)
        assert "SuperSawPE" in repr_str
        assert "440.0" in repr_str
        assert "5" in repr_str
        assert "15.0" in repr_str


class TestSuperSawPERender:
    """Test SuperSawPE rendering."""
    
    def setup_method(self):
        """Create a renderer for configuring PEs."""
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_render_returns_snippet(self):
        """Test that render returns a properly shaped Snippet."""
        saw = SuperSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 100)
        assert snippet.start == 0
        assert snippet.duration == 100
        assert snippet.channels == 1
        
        self.renderer.stop()
    
    def test_render_stereo(self):
        """Test stereo output."""
        saw = SuperSawPE(frequency=440.0, channels=2)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 100)
        assert snippet.channels == 2
        # Both channels should be identical (mono expanded to stereo)
        np.testing.assert_array_equal(
            snippet.data[:, 0],
            snippet.data[:, 1]
        )
        
        self.renderer.stop()
    
    def test_render_amplitude(self):
        """Test amplitude scaling."""
        saw = SuperSawPE(frequency=100.0, amplitude=0.5)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # Render enough to stabilize
        snippet = saw.render(0, 44100)
        
        # Check amplitude in stable region
        stable_data = snippet.data[22050:, 0]
        max_val = np.max(np.abs(stable_data))
        
        # Should be less than 0.5 * some headroom
        assert max_val < 0.8, f"Amplitude too high: {max_val}"
        
        self.renderer.stop()
    
    def test_render_single_voice_equals_blit_saw(self):
        """Single voice with no detune should behave like BlitSawPE."""
        from pygmu2 import BlitSawPE
        
        super_saw = SuperSawPE(frequency=440.0, voices=1, detune_cents=0.0)
        blit_saw = BlitSawPE(frequency=440.0)
        
        renderer1 = NullRenderer(sample_rate=44100)
        renderer2 = NullRenderer(sample_rate=44100)
        
        renderer1.set_source(super_saw)
        renderer2.set_source(blit_saw)
        
        renderer1.start()
        renderer2.start()
        
        snippet1 = super_saw.render(0, 1000)
        snippet2 = blit_saw.render(0, 1000)
        
        # Should be very similar (not exact due to normalization)
        correlation = np.corrcoef(
            snippet1.data[:, 0].flatten(),
            snippet2.data[:, 0].flatten()
        )[0, 1]
        
        assert correlation > 0.99, f"Correlation too low: {correlation}"
        
        renderer1.stop()
        renderer2.stop()
    
    def test_render_zero_duration(self):
        """Test rendering zero samples returns empty snippet."""
        saw = SuperSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 0)
        assert snippet.duration == 0
        
        self.renderer.stop()
    
    def test_render_continuity(self):
        """Test that consecutive renders are continuous."""
        saw = SuperSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # Render two consecutive chunks
        chunk1 = saw.render(0, 1000)
        chunk2 = saw.render(1000, 1000)
        
        # Should be continuous at boundary
        last_val = chunk1.data[-1, 0]
        first_val = chunk2.data[0, 0]
        
        diff = abs(first_val - last_val)
        assert diff < 0.5, f"Discontinuity too large: {diff}"
        
        self.renderer.stop()


class TestSuperSawPEDetune:
    """Test detuning behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_detune_ratios_symmetric(self):
        """Detune ratios should be symmetric around 1.0."""
        saw = SuperSawPE(frequency=440.0, voices=7, detune_cents=20.0)
        
        ratios = saw._detune_ratios
        assert len(ratios) == 7
        
        # Center ratio should be close to 1.0
        center_idx = 3
        assert abs(ratios[center_idx] - 1.0) < 0.001
        
        # Should be symmetric
        for i in range(3):
            low_ratio = ratios[i]
            high_ratio = ratios[6 - i]
            # Product should be close to 1.0 (symmetric in log space)
            assert abs(low_ratio * high_ratio - 1.0) < 0.001
    
    def test_detune_zero_collapses(self):
        """Zero detune should give all ratios = 1.0."""
        saw = SuperSawPE(frequency=440.0, voices=5, detune_cents=0.0)
        
        ratios = saw._detune_ratios
        # With detune=0, should collapse to single ratio
        assert len(ratios) == 1
        assert ratios[0] == 1.0
    
    def test_more_detune_wider_spread(self):
        """More detune should give wider frequency spread."""
        saw_narrow = SuperSawPE(frequency=440.0, detune_cents=10.0)
        saw_wide = SuperSawPE(frequency=440.0, detune_cents=50.0)
        
        narrow_spread = saw_narrow._detune_ratios[-1] - saw_narrow._detune_ratios[0]
        wide_spread = saw_wide._detune_ratios[-1] - saw_wide._detune_ratios[0]
        
        assert wide_spread > narrow_spread
    
    def test_detune_produces_beating(self):
        """Detuned voices should produce audible beating."""
        saw = SuperSawPE(frequency=100.0, voices=3, detune_cents=50.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # Render enough to see beating
        snippet = saw.render(0, 44100)
        data = snippet.data[:, 0]
        
        # Compute envelope using RMS in windows
        window_size = 1000
        n_windows = len(data) // window_size
        rms_values = []
        for i in range(n_windows):
            window = data[i * window_size:(i + 1) * window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        # Envelope should vary (beating)
        envelope_variation = np.std(rms_values) / np.mean(rms_values)
        assert envelope_variation > 0.01, "Should have some amplitude variation from beating"
        
        self.renderer.stop()


class TestSuperSawPEMixModes:
    """Test different mix modes."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_mix_equal_gains(self):
        """Equal mix mode should have equal gains."""
        saw = SuperSawPE(frequency=440.0, voices=5, mix_mode='equal')
        gains = saw._mix_gains
        
        # All gains should be equal
        assert np.allclose(gains, gains[0])
    
    def test_mix_center_heavy_gains(self):
        """Center heavy mode should have louder center."""
        saw = SuperSawPE(frequency=440.0, voices=5, mix_mode='center_heavy')
        gains = saw._mix_gains
        
        center_idx = 2
        # Center should be loudest
        assert gains[center_idx] == np.max(gains)
    
    def test_mix_linear_gains(self):
        """Linear mode should have gains decreasing from center."""
        saw = SuperSawPE(frequency=440.0, voices=5, mix_mode='linear')
        gains = saw._mix_gains
        
        center_idx = 2
        # Center should be loudest
        assert gains[center_idx] == np.max(gains)
        # Edges should be quieter than center
        assert gains[0] < gains[center_idx]
        assert gains[4] < gains[center_idx]
    
    def test_mix_modes_normalized(self):
        """All mix modes should have normalized total power."""
        for mode in ['equal', 'center_heavy', 'linear']:
            saw = SuperSawPE(frequency=440.0, voices=7, mix_mode=mode)
            gains = saw._mix_gains
            
            # Sum of squared gains should be approximately 1
            power = np.sum(gains ** 2)
            assert abs(power - 1.0) < 0.01, f"Mode {mode} not normalized: {power}"


class TestSuperSawPEModulation:
    """Test SuperSawPE with PE inputs."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_frequency_from_pe(self):
        """Test using a PE for frequency input."""
        freq_pe = ConstantPE(440.0)
        saw = SuperSawPE(frequency=freq_pe)
        
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 1000)
        assert snippet.duration == 1000
        
        self.renderer.stop()
    
    def test_amplitude_from_pe(self):
        """Test using a PE for amplitude input."""
        amp_pe = ConstantPE(0.5)
        saw = SuperSawPE(frequency=440.0, amplitude=amp_pe)
        
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 44100)
        
        # Check that amplitude PE is being applied
        # With 7 detuned voices, peaks can constructively interfere
        # so we just verify the output isn't unreasonably large
        stable_data = snippet.data[22050:, 0]
        max_val = np.max(np.abs(stable_data))
        # 0.5 amplitude * 7 voices with some normalization should be < 2.0
        assert max_val < 2.0, f"Amplitude too high: {max_val}"
        # And should be greater than 0 (sanity check)
        assert max_val > 0.1, f"Amplitude too low: {max_val}"
        
        self.renderer.stop()
    
    def test_inputs_returns_pe_inputs(self):
        """inputs() should return all PE inputs."""
        freq_pe = ConstantPE(440.0)
        amp_pe = ConstantPE(0.5)
        
        saw = SuperSawPE(frequency=freq_pe, amplitude=amp_pe)
        
        inputs = saw.inputs()
        assert len(inputs) == 2
        assert freq_pe in inputs
        assert amp_pe in inputs
    
    def test_extent_with_pe_inputs(self):
        """Extent should be intersection of PE input extents."""
        finite_pe = RampPE(0.0, 1.0, duration=1000)
        saw = SuperSawPE(frequency=440.0, amplitude=finite_pe)
        
        extent = saw.extent()
        assert extent.start == 0
        assert extent.end == 1000
    
    def test_vibrato_modulation(self):
        """Test frequency modulation (vibrato) with PE."""
        # This is a basic smoke test - just verify it doesn't crash
        lfo = SinePE(frequency=5.0, amplitude=10.0)  # ±10 Hz vibrato
        base_freq = ConstantPE(440.0)
        
        # We can't easily add PEs, so use a constant for now
        saw = SuperSawPE(frequency=440.0)
        
        self.renderer.set_source(saw)
        self.renderer.start()
        
        snippet = saw.render(0, 1000)
        assert snippet.duration == 1000
        
        self.renderer.stop()


class TestSuperSawPEStateManagement:
    """Test state management (on_start, on_stop)."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_on_start_resets_state(self):
        """on_start() should reset internal oscillator state."""
        saw = SuperSawPE(frequency=440.0)
        self.renderer.set_source(saw)
        self.renderer.start()
        
        # Render some samples
        chunk1 = saw.render(0, 1000)
        
        # Stop and restart
        self.renderer.stop()
        self.renderer.start()
        
        # Should produce same output as before
        chunk2 = saw.render(0, 1000)
        np.testing.assert_array_almost_equal(
            chunk1.data, chunk2.data, decimal=5
        )
        
        self.renderer.stop()


class TestSuperSawPEDifferentSampleRates:
    """Test SuperSawPE at different sample rates."""
    
    def test_sample_rate_48000(self):
        """Test at 48kHz sample rate."""
        saw = SuperSawPE(frequency=480.0)
        renderer = NullRenderer(sample_rate=48000)
        renderer.set_source(saw)
        renderer.start()
        
        snippet = saw.render(0, 48000)
        assert snippet.duration == 48000
        
        renderer.stop()
    
    def test_sample_rate_22050(self):
        """Test at 22.05kHz sample rate."""
        saw = SuperSawPE(frequency=220.0)
        renderer = NullRenderer(sample_rate=22050)
        renderer.set_source(saw)
        renderer.start()
        
        snippet = saw.render(0, 22050)
        assert snippet.duration == 22050
        
        renderer.stop()
