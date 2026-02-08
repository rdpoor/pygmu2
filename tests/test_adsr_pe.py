"""
Tests for AdsrPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    AdsrPE,
    ArrayPE,
    ConstantPE,
    CropPE,
    Extent,
    NullRenderer,
)


class TestAdsrPEBasics:
    """Test basic AdsrPE creation and properties."""
    
    def setup_method(self):
        import pygmu2 as pg
        pg.set_sample_rate(1000)
        self.renderer = NullRenderer(sample_rate=1000)  # 1kHz for easy math
    
    def test_create_default(self):
        """Test creation with default parameters."""
        gate = ConstantPE(1.0)
        adsr = AdsrPE(gate)
        self.renderer.set_source(adsr)
        
        assert adsr.gate is gate
        # Defaults are specified in seconds and resolved at construction time.
        assert adsr.attack == 10    # 10ms at 1kHz
        assert adsr.decay == 100    # 100ms at 1kHz
        assert adsr.sustain_level == 0.7
        assert adsr.release == 200  # 200ms at 1kHz
    
    def test_create_with_params(self):
        """Test creation with custom parameters."""
        gate = ConstantPE(1.0)
        adsr = AdsrPE(
            gate,
            attack_samples=50,
            decay_samples=200,
            sustain_level=0.5,
            release_samples=300,
        )
        self.renderer.set_source(adsr)
        
        assert adsr.attack == 50
        assert adsr.decay == 200
        assert adsr.sustain_level == 0.5
        assert adsr.release == 300
    
    def test_is_pure(self):
        """AdsrPE is not pure (maintains state)."""
        gate = ConstantPE(1.0)
        adsr = AdsrPE(gate)
        assert adsr.is_pure() is False
    
    def test_channel_count(self):
        """AdsrPE outputs mono control signal."""
        gate = ConstantPE(1.0)
        adsr = AdsrPE(gate)
        assert adsr.channel_count() == 1

    def test_extent_finite_gate_includes_release(self):
        gate = CropPE(ConstantPE(1.0), 0, 100)
        adsr = AdsrPE(gate, release_samples=20)
        extent = adsr.extent()
        assert extent.start == 0
        assert extent.end == 120

    def test_extent_infinite_gate_is_infinite(self):
        gate = ConstantPE(1.0)
        adsr = AdsrPE(gate, release_samples=20)
        extent = adsr.extent()
        assert extent.start is None
        assert extent.end is None
    
    def test_state_reset_on_start(self):
        """Test that state resets on on_start()."""
        gate = ArrayPE([0.0, 1.0, 1.0, 0.0])
        adsr = AdsrPE(gate, attack_samples=10, decay_samples=10, sustain_level=0.5, release_samples=10)
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        # After start, should be in IDLE state
        snippet = adsr.render(0, 1)
        assert snippet.data[0, 0] == 0.0
    
    def test_repr(self):
        """Test string representation."""
        gate = ConstantPE(1.0)
        adsr = AdsrPE(gate, attack_samples=10, decay_samples=100, sustain_level=0.7, release_samples=200)
        repr_str = repr(adsr)
        assert "AdsrPE" in repr_str
        assert "attack_samples=10" in repr_str


class TestAdsrPERender:
    """Test AdsrPE rendering behavior."""
    
    def setup_method(self):
        import pygmu2 as pg
        pg.set_sample_rate(1000)
        self.renderer = NullRenderer(sample_rate=1000)  # 1kHz for easy math
    
    def test_complete_adsr_cycle(self):
        """Test complete ADSR cycle: gate high -> attack -> decay -> sustain -> gate low -> release."""
        # Gate: high for 500 samples, then low
        gate_data = np.concatenate([
            np.ones(500, dtype=np.float32),
            np.zeros(500, dtype=np.float32),
        ])
        gate = ArrayPE(gate_data)
        
        # ADSR with 10 samples attack, 20 samples decay, 0.5 sustain, 30 samples release
        adsr = AdsrPE(
            gate,
            attack_samples=10,
            decay_samples=20,
            sustain_level=0.5,
            release_samples=30,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        # Render full cycle
        snippet = adsr.render(0, 1000)
        output = snippet.data[:, 0]
        
        # Attack phase (0-10): ramp from 0 to 1
        # At sample 5: should be ~0.5
        assert abs(output[5] - 0.5) < 0.1
        # At sample 10: should be ~1.0
        assert abs(output[10] - 1.0) < 0.1
        
        # Decay phase (10-30): ramp from 1.0 to 0.5
        # At sample 20: should be ~0.75 (halfway between 1.0 and 0.5)
        assert abs(output[20] - 0.75) < 0.1
        # At sample 30: should be ~0.5
        assert abs(output[30] - 0.5) < 0.1
        
        # Sustain phase (30-500): hold at 0.5
        assert abs(output[100] - 0.5) < 0.01
        assert abs(output[400] - 0.5) < 0.01
        
        # Gate goes low at sample 500 -> Release phase
        # At sample 515: should be ramping down from 0.5
        # Release is 30 samples, so at 15 samples into release: 0.5 * (1 - 15/30) = 0.25
        assert abs(output[515] - 0.25) < 0.1
        
        # At sample 530: should be ~0.0 (release complete)
        assert abs(output[530]) < 0.1
        
        # After release: should stay at 0 (IDLE)
        assert abs(output[600]) < 0.01
        assert abs(output[999]) < 0.01
    
    def test_early_release_during_attack(self):
        """Test early release: gate falls during attack phase."""
        # Gate: high for 5 samples (during attack), then low
        gate_data = np.concatenate([
            np.ones(5, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
        ])
        gate = ArrayPE(gate_data)
        
        # ADSR with 10 samples attack, 20 samples release
        adsr = AdsrPE(
            gate,
            attack_samples=10,
            decay_samples=20,
            sustain_level=0.5,
            release_samples=20,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 105)
        output = snippet.data[:, 0]
        
        # At sample 5: attack phase, should be ~0.5 (halfway through 10-sample attack)
        assert abs(output[5] - 0.5) < 0.1
        
        # Gate falls at sample 5 -> enters release from current value (~0.5)
        # At sample 15: 10 samples into release, should be ~0.25 (halfway to 0)
        assert abs(output[15] - 0.25) < 0.1
        
        # At sample 25: release complete, should be ~0
        assert abs(output[25]) < 0.1
    
    def test_early_release_during_decay(self):
        """Test early release: gate falls during decay phase."""
        # Gate: high for 15 samples (attack complete, in decay), then low
        gate_data = np.concatenate([
            np.ones(15, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
        ])
        gate = ArrayPE(gate_data)
        
        # ADSR with 10 samples attack, 20 samples decay, 20 samples release
        adsr = AdsrPE(
            gate,
            attack_samples=10,
            decay_samples=20,
            sustain_level=0.5,
            release_samples=20,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 115)
        output = snippet.data[:, 0]
        
        # At sample 10: attack complete, should be ~1.0
        assert abs(output[10] - 1.0) < 0.1
        
        # At sample 15: 5 samples into decay (1.0 -> 0.5 over 20 samples)
        # Should be: 1.0 - (5/20) * (1.0 - 0.5) = 1.0 - 0.125 = 0.875
        assert abs(output[15] - 0.875) < 0.1
        
        # Gate falls at sample 15 -> enters release from current value (~0.875)
        # At sample 25: 10 samples into release, should be ~0.4375
        assert abs(output[25] - 0.4375) < 0.1
        
        # At sample 35: release complete, should be ~0
        assert abs(output[35]) < 0.1
    
    def test_retrigger_during_release(self):
        """Test re-triggering: gate rises again during release phase."""
        # Gate: high (0-50), low (50-65), high again (65-115)
        # Re-trigger happens 15 samples into release (before release completes)
        gate_data = np.concatenate([
            np.ones(50, dtype=np.float32),
            np.zeros(15, dtype=np.float32),
            np.ones(50, dtype=np.float32),
        ])
        gate = ArrayPE(gate_data)
        
        # ADSR with 10 samples attack, 20 samples decay, 30 samples release
        adsr = AdsrPE(
            gate,
            attack_samples=10,
            decay_samples=20,
            sustain_level=0.5,
            release_samples=30,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 115)
        output = snippet.data[:, 0]
        
        # First cycle: attack -> decay -> sustain -> release
        # At sample 50: should be at sustain (0.5)
        assert abs(output[50] - 0.5) < 0.1
        
        # Gate falls at 50 -> release starts from 0.5
        # At sample 65: 15 samples into release (30 total)
        # Should be: 0.5 * (1 - 15/30) = 0.25
        assert abs(output[65] - 0.25) < 0.1
        
        # Gate rises again at sample 65 -> re-trigger during release
        # Should enter attack from current value (0.25)
        # At sample 70: 5 samples into attack from 0.25 to 1.0
        # Progress: 5/10 = 0.5, so value = 0.25 + (1.0 - 0.25) * 0.5 = 0.625
        assert abs(output[70] - 0.625) < 0.1
        
        # At sample 75: attack complete, should be ~1.0
        assert abs(output[75] - 1.0) < 0.1
    
    def test_idle_state(self):
        """Test that output is zero when gate is low and ADSR is idle."""
        gate = ArrayPE([0.0] * 100)
        adsr = AdsrPE(gate, attack_samples=10, decay_samples=20, sustain_level=0.5, release_samples=30)
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 100)
        output = snippet.data[:, 0]
        
        # All should be zero (gate never goes high)
        np.testing.assert_array_almost_equal(output, np.zeros(100), decimal=5)
    
    def test_sustain_holds_until_gate_falls(self):
        """Test that sustain phase holds until gate signal falls."""
        # Gate: high for 200 samples
        gate = ArrayPE([1.0] * 200)
        adsr = AdsrPE(
            gate,
            attack_samples=10,
            decay_samples=20,
            sustain_level=0.6,
            release_samples=30,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 200)
        output = snippet.data[:, 0]
        
        # After attack + decay (30 samples), should hold at sustain
        # At sample 50: should be at sustain (0.6)
        assert abs(output[50] - 0.6) < 0.01
        # At sample 100: should still be at sustain
        assert abs(output[100] - 0.6) < 0.01
        # At sample 199: should still be at sustain
        assert abs(output[199] - 0.6) < 0.01


class TestAdsrPESampleAccurate:
    """Test sample-accurate behavior of AdsrPE."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=1000)  # 1kHz for easy math
    
    def test_precise_attack_ramp(self):
        """Test precise attack ramp values."""
        # Gate: high for 20 samples
        gate = ArrayPE([1.0] * 20)
        adsr = AdsrPE(
            gate,
            attack_samples=10,
            decay_samples=20,
            sustain_level=0.5,
            release_samples=30,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 20)
        output = snippet.data[:, 0]
        
        # Attack phase: linear ramp from 0 to 1 over 10 samples
        # Sample 0: 0.0
        assert abs(output[0]) < 0.01
        # Sample 5: 0.5 (halfway)
        assert abs(output[5] - 0.5) < 0.01
        # Sample 10: 1.0 (attack complete)
        assert abs(output[10] - 1.0) < 0.01
    
    def test_precise_decay_ramp(self):
        """Test precise decay ramp values."""
        # Gate: high for 40 samples (covers attack + decay)
        gate = ArrayPE([1.0] * 40)
        adsr = AdsrPE(
            gate,
            attack_samples=10,
            decay_samples=20,
            sustain_level=0.5,
            release_samples=30,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 40)
        output = snippet.data[:, 0]
        
        # Decay phase: linear ramp from 1.0 to 0.5 over 20 samples (samples 10-30)
        # Sample 10: 1.0 (decay starts)
        assert abs(output[10] - 1.0) < 0.01
        # Sample 20: 0.75 (halfway: 1.0 - 0.5 * 0.5 = 0.75)
        assert abs(output[20] - 0.75) < 0.01
        # Sample 30: 0.5 (decay complete, sustain starts)
        assert abs(output[30] - 0.5) < 0.01
    
    def test_precise_release_ramp(self):
        """Test precise release ramp values."""
        # Gate: high for 50 samples, then low
        gate_data = np.concatenate([
            np.ones(50, dtype=np.float32),
            np.zeros(50, dtype=np.float32),
        ])
        gate = ArrayPE(gate_data)
        adsr = AdsrPE(
            gate,
            attack_samples=10,
            decay_samples=20,
            sustain_level=0.5,
            release_samples=30,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 100)
        output = snippet.data[:, 0]
        
        # Release phase: linear ramp from 0.5 to 0.0 over 30 samples (samples 50-80)
        # Sample 50: 0.5 (release starts)
        assert abs(output[50] - 0.5) < 0.01
        # Sample 65: 0.25 (halfway: 0.5 * (1 - 15/30) = 0.25)
        assert abs(output[65] - 0.25) < 0.01
        # Sample 80: 0.0 (release complete)
        assert abs(output[80]) < 0.01


class TestAdsrPEEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=1000)
    
    def test_zero_sustain_level(self):
        """Test ADSR with zero sustain level (no sustain phase)."""
        gate = ArrayPE([1.0] * 50)
        adsr = AdsrPE(
            gate,
            attack_samples=10,
            decay_samples=20,
            sustain_level=0.0,  # Zero sustain
            release_samples=30,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 50)
        output = snippet.data[:, 0]
        
        # After attack + decay, should be at 0.0 (sustain level)
        assert abs(output[30]) < 0.01
        assert abs(output[49]) < 0.01
    
    def test_unit_sustain_level(self):
        """Test ADSR with sustain level of 1.0 (no decay)."""
        gate = ArrayPE([1.0] * 50)
        adsr = AdsrPE(
            gate,
            attack_samples=10,
            decay_samples=20,
            sustain_level=1.0,  # Unit sustain
            release_samples=30,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 50)
        output = snippet.data[:, 0]
        
        # After attack, should go to 1.0, then decay to 1.0 (no change), then sustain at 1.0
        assert abs(output[10] - 1.0) < 0.01
        assert abs(output[30] - 1.0) < 0.01
        assert abs(output[49] - 1.0) < 0.01
    
    def test_very_short_times(self):
        """Test ADSR with very short time parameters."""
        gate = ArrayPE([1.0] * 10)
        adsr = AdsrPE(
            gate,
            attack_samples=1,
            decay_samples=1,
            sustain_level=0.5,
            release_samples=1,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 10)
        output = snippet.data[:, 0]
        
        # Should still function (minimum 1 sample per phase)
        # Sample 0: attack starts
        # Sample 1: attack complete (1.0), decay starts
        # Sample 2: decay complete (0.5), sustain starts
        assert abs(output[2] - 0.5) < 0.1
    
    def test_rapid_gate_changes(self):
        """Test rapid gate on/off changes."""
        # Gate: rapid on/off pattern
        gate_data = np.array([
            1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0
        ], dtype=np.float32)
        gate = ArrayPE(gate_data)
        adsr = AdsrPE(
            gate,
            attack_samples=10,
            decay_samples=20,
            sustain_level=0.5,
            release_samples=10,
        )
        
        self.renderer.set_source(adsr)
        self.renderer.start()
        
        snippet = adsr.render(0, 10)
        output = snippet.data[:, 0]
        
        # Should handle rapid changes without crashing
        # Values should be in valid range [0, 1]
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)
