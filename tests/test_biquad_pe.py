"""
Tests for BiquadPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    BiquadPE,
    BiquadMode,
    ConstantPE,
    PiecewisePE,
    SinePE,
    DiracPE,
    NullRenderer,
    Extent,
)


class TestBiquadPEBasics:
    """Test basic BiquadPE creation and properties."""
    
    def test_create_lowpass(self):
        source = ConstantPE(1.0)
        bq = BiquadPE(source, frequency=1000.0, q=0.707, mode=BiquadMode.LOWPASS)
        
        assert bq.source is source
        assert bq.frequency == 1000.0
        assert bq.q == 0.707
        assert bq.mode == BiquadMode.LOWPASS
    
    def test_create_all_modes(self):
        """Test that all filter modes can be created."""
        source = ConstantPE(1.0)
        
        for mode in BiquadMode:
            bq = BiquadPE(source, frequency=1000.0, q=1.0, mode=mode)
            assert bq.mode == mode
    
    def test_create_with_pe_frequency(self):
        source = ConstantPE(1.0)
        freq_pe = PiecewisePE([(0, 100.0), (44100, 5000.0)])
        
        bq = BiquadPE(source, frequency=freq_pe, q=1.0, mode=BiquadMode.LOWPASS)
        
        assert bq.frequency is freq_pe
    
    def test_create_with_pe_q(self):
        source = ConstantPE(1.0)
        q_pe = PiecewisePE([(0, 0.5), (44100, 10.0)])
        
        bq = BiquadPE(source, frequency=1000.0, q=q_pe, mode=BiquadMode.LOWPASS)
        
        assert bq.q is q_pe
    
    def test_create_with_gain_db(self):
        source = ConstantPE(1.0)
        bq = BiquadPE(
            source, frequency=1000.0, q=1.0,
            mode=BiquadMode.PEAKING, gain_db=6.0
        )
        
        assert bq.gain_db == 6.0
    
    def test_inputs_constant_params(self):
        source = ConstantPE(1.0)
        bq = BiquadPE(source, frequency=1000.0, q=1.0, mode=BiquadMode.LOWPASS)
        
        assert bq.inputs() == [source]
    
    def test_inputs_with_pe_frequency(self):
        source = ConstantPE(1.0)
        freq_pe = PiecewisePE([(0, 100.0), (44100, 5000.0)])
        
        bq = BiquadPE(source, frequency=freq_pe, q=1.0, mode=BiquadMode.LOWPASS)
        
        inputs = bq.inputs()
        assert len(inputs) == 2
        assert source in inputs
        assert freq_pe in inputs
    
    def test_inputs_with_pe_q(self):
        source = ConstantPE(1.0)
        q_pe = PiecewisePE([(0, 0.5), (44100, 10.0)])
        
        bq = BiquadPE(source, frequency=1000.0, q=q_pe, mode=BiquadMode.LOWPASS)
        
        inputs = bq.inputs()
        assert len(inputs) == 2
        assert source in inputs
        assert q_pe in inputs
    
    def test_inputs_with_both_pe(self):
        source = ConstantPE(1.0)
        freq_pe = PiecewisePE([(0, 100.0), (44100, 5000.0)])
        q_pe = PiecewisePE([(0, 0.5), (44100, 10.0)])
        
        bq = BiquadPE(source, frequency=freq_pe, q=q_pe, mode=BiquadMode.LOWPASS)
        
        inputs = bq.inputs()
        assert len(inputs) == 3
        assert source in inputs
        assert freq_pe in inputs
        assert q_pe in inputs
    
    def test_is_not_pure(self):
        """BiquadPE maintains state, so is_pure() should return False."""
        source = ConstantPE(1.0)
        bq = BiquadPE(source, frequency=1000.0, q=1.0, mode=BiquadMode.LOWPASS)
        
        assert bq.is_pure() is False
    
    def test_channel_count_passthrough(self):
        source = ConstantPE(1.0, channels=2)
        bq = BiquadPE(source, frequency=1000.0, q=1.0, mode=BiquadMode.LOWPASS)
        
        assert bq.channel_count() == 2
    
    def test_extent_from_source(self):
        source = PiecewisePE([(0, 0.0), (1000, 1.0)])
        bq = BiquadPE(source, frequency=1000.0, q=1.0, mode=BiquadMode.LOWPASS)
        
        extent = bq.extent()
        assert extent.start == 0
        assert extent.end == 1000
    
    def test_repr(self):
        source = ConstantPE(1.0)
        bq = BiquadPE(source, frequency=1000.0, q=0.707, mode=BiquadMode.LOWPASS)
        
        repr_str = repr(bq)
        assert "BiquadPE" in repr_str
        assert "ConstantPE" in repr_str
        assert "1000.0" in repr_str
        assert "0.707" in repr_str
        assert "lowpass" in repr_str


class TestBiquadPELowpass:
    """Test lowpass filter behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_lowpass_passes_dc(self):
        """DC signal should pass through lowpass filter unchanged."""
        source = ConstantPE(1.0)
        bq = BiquadPE(source, frequency=1000.0, q=0.707, mode=BiquadMode.LOWPASS)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            # Let filter settle
            _ = bq.render(0, 1000)
            
            # Check steady state
            snippet = bq.render(1000, 100)
            
            # DC should pass through (approximately 1.0)
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 1.0, dtype=np.float32),
                decimal=2
            )
    
    def test_lowpass_attenuates_high_freq(self):
        """High frequency signal should be attenuated by lowpass filter."""
        # Source: high frequency sine (10kHz)
        source = SinePE(frequency=10000.0, amplitude=1.0)
        
        # Lowpass at 1kHz
        bq = BiquadPE(source, frequency=1000.0, q=0.707, mode=BiquadMode.LOWPASS)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            # Let filter settle
            _ = bq.render(0, 1000)
            
            # Check that output is significantly attenuated
            snippet = bq.render(1000, 1000)
            
            # RMS of output should be much less than input RMS (1/sqrt(2) ≈ 0.707)
            output_rms = np.sqrt(np.mean(snippet.data ** 2))
            assert output_rms < 0.1  # Significant attenuation


class TestBiquadPEHighpass:
    """Test highpass filter behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_highpass_blocks_dc(self):
        """DC signal should be blocked by highpass filter."""
        source = ConstantPE(1.0)
        bq = BiquadPE(source, frequency=1000.0, q=0.707, mode=BiquadMode.HIGHPASS)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            # Let filter settle
            _ = bq.render(0, 1000)
            
            # Check steady state
            snippet = bq.render(1000, 100)
            
            # DC should be blocked (approximately 0.0)
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.zeros((100, 1), dtype=np.float32),
                decimal=2
            )


class TestBiquadPEBandpass:
    """Test bandpass filter behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_bandpass_blocks_dc(self):
        """DC signal should be blocked by bandpass filter."""
        source = ConstantPE(1.0)
        bq = BiquadPE(source, frequency=1000.0, q=5.0, mode=BiquadMode.BANDPASS)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            _ = bq.render(0, 1000)
            snippet = bq.render(1000, 100)
            
            # DC should be blocked
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.zeros((100, 1), dtype=np.float32),
                decimal=2
            )


class TestBiquadPENotch:
    """Test notch (band-reject) filter behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_notch_passes_dc(self):
        """DC signal should pass through notch filter."""
        source = ConstantPE(1.0)
        bq = BiquadPE(source, frequency=1000.0, q=5.0, mode=BiquadMode.NOTCH)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            _ = bq.render(0, 1000)
            snippet = bq.render(1000, 100)
            
            # DC should pass through
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 1.0, dtype=np.float32),
                decimal=2
            )


class TestBiquadPEAllpass:
    """Test allpass filter behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_allpass_preserves_magnitude(self):
        """Allpass filter should preserve signal magnitude."""
        source = ConstantPE(1.0)
        bq = BiquadPE(source, frequency=1000.0, q=0.707, mode=BiquadMode.ALLPASS)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            _ = bq.render(0, 1000)
            snippet = bq.render(1000, 100)
            
            # Magnitude should be preserved (≈1.0)
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 1.0, dtype=np.float32),
                decimal=2
            )


class TestBiquadPEImpulseResponse:
    """Test filter impulse response."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_impulse_response_decays(self):
        """Filter impulse response should decay over time."""
        source = DiracPE()
        bq = BiquadPE(source, frequency=1000.0, q=2.0, mode=BiquadMode.LOWPASS)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            # Render impulse response
            snippet = bq.render(0, 1000)
            
            # Initial response should be non-zero
            assert abs(snippet.data[0, 0]) > 0.001
            
            # Peak of response (may not be at sample 0 for resonant filters)
            peak = np.max(np.abs(snippet.data[:100]))
            assert peak > 0.01
            
            # Response should decay toward zero over time
            late_response = np.max(np.abs(snippet.data[-100:]))
            assert late_response < peak


class TestBiquadPETimeVarying:
    """Test time-varying filter parameters."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_frequency_sweep(self):
        """Test filter with sweeping frequency."""
        source = ConstantPE(1.0)
        freq_sweep = PiecewisePE([(0, 100.0), (1000, 10000.0)])
        
        bq = BiquadPE(source, frequency=freq_sweep, q=0.707, mode=BiquadMode.LOWPASS)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            # Should render without error
            snippet = bq.render(0, 1000)
            
            assert snippet.duration == 1000
            assert snippet.channels == 1
    
    def test_q_modulation(self):
        """Test filter with modulated Q."""
        source = ConstantPE(1.0)
        q_mod = PiecewisePE([(0, 0.5), (1000, 10.0)])
        
        bq = BiquadPE(source, frequency=1000.0, q=q_mod, mode=BiquadMode.LOWPASS)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            snippet = bq.render(0, 1000)
            
            assert snippet.duration == 1000


class TestBiquadPEStateManagement:
    """Test filter state management."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_state_persists_across_renders(self):
        """Filter state should persist between render calls."""
        source = DiracPE()
        bq = BiquadPE(source, frequency=1000.0, q=5.0, mode=BiquadMode.LOWPASS)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            # First render - captures impulse
            snippet1 = bq.render(0, 100)
            
            # Second render - should have decaying response
            snippet2 = bq.render(100, 100)
            
            # The response should continue from where it left off
            # (not restart from zero)
            # Check that the start of snippet2 is close to end of snippet1
            # in terms of decay envelope
            assert snippet2.data[0, 0] != 0.0  # State carried over
    
    def test_state_resets_on_start(self):
        """Filter state should reset when on_start() is called."""
        source = DiracPE()
        bq = BiquadPE(source, frequency=1000.0, q=5.0, mode=BiquadMode.LOWPASS)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            # First pass
            snippet1 = bq.render(0, 100)
            
            # Restart
            self.renderer.stop()
            self.renderer.start()
            
            # Second pass - should match first
            snippet2 = bq.render(0, 100)
            
            # Should be identical (state was reset)
            np.testing.assert_array_almost_equal(
                snippet1.data, snippet2.data, decimal=5
            )


class TestBiquadPEStereo:
    """Test stereo signal handling."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_stereo_filtering(self):
        """Filter should process stereo signals independently."""
        source = ConstantPE(1.0, channels=2)
        bq = BiquadPE(source, frequency=1000.0, q=0.707, mode=BiquadMode.LOWPASS)
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            _ = bq.render(0, 500)
            snippet = bq.render(500, 100)
            
            assert snippet.channels == 2
            # Both channels should be approximately 1.0
            np.testing.assert_array_almost_equal(
                snippet.data[:, 0],
                snippet.data[:, 1],
                decimal=5
            )


class TestBiquadPEPeakingEQ:
    """Test peaking EQ filter."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_peaking_boost(self):
        """Peaking filter with positive gain should boost around center frequency."""
        source = SinePE(frequency=1000.0, amplitude=1.0)
        bq = BiquadPE(
            source, frequency=1000.0, q=2.0,
            mode=BiquadMode.PEAKING, gain_db=6.0  # 6dB boost
        )
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            _ = bq.render(0, 1000)
            snippet = bq.render(1000, 1000)
            
            # Output should be louder than input
            output_rms = np.sqrt(np.mean(snippet.data ** 2))
            # 6dB boost ≈ 2x amplitude ≈ 1.41 RMS (from original 0.707)
            assert output_rms > 1.0
    
    def test_peaking_cut(self):
        """Peaking filter with negative gain should cut around center frequency."""
        source = SinePE(frequency=1000.0, amplitude=1.0)
        bq = BiquadPE(
            source, frequency=1000.0, q=2.0,
            mode=BiquadMode.PEAKING, gain_db=-6.0  # 6dB cut
        )
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            _ = bq.render(0, 1000)
            snippet = bq.render(1000, 1000)
            
            # Output should be quieter than input
            output_rms = np.sqrt(np.mean(snippet.data ** 2))
            assert output_rms < 0.707  # Less than original RMS


class TestBiquadPEShelfFilters:
    """Test shelf filters."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_lowshelf_boost_dc(self):
        """Low shelf with boost should increase DC level."""
        source = ConstantPE(1.0)
        bq = BiquadPE(
            source, frequency=1000.0, q=0.707,
            mode=BiquadMode.LOWSHELF, gain_db=6.0
        )
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            _ = bq.render(0, 1000)
            snippet = bq.render(1000, 100)
            
            # DC should be boosted (≈2x)
            mean_output = np.mean(snippet.data)
            assert mean_output > 1.5
    
    def test_highshelf_passes_dc(self):
        """High shelf should not affect DC (below shelf frequency)."""
        source = ConstantPE(1.0)
        bq = BiquadPE(
            source, frequency=1000.0, q=0.707,
            mode=BiquadMode.HIGHSHELF, gain_db=6.0
        )
        
        self.renderer.set_source(bq)
        
        with self.renderer:
            self.renderer.start()
            
            _ = bq.render(0, 1000)
            snippet = bq.render(1000, 100)
            
            # DC should be approximately unchanged
            mean_output = np.mean(snippet.data)
            assert 0.9 < mean_output < 1.1
