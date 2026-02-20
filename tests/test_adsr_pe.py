"""
Tests for AdsrGatedPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    AdsrGatedPE,
    ArrayPE,
    ConstantPE,
    CropPE,
    Extent,
    NullRenderer,
)


class TestAdsrGatedPEBasics:
    """Test basic AdsrGatedPE creation and properties."""

    def setup_method(self):
        import pygmu2 as pg
        pg.set_sample_rate(1000)
        self.renderer = NullRenderer(sample_rate=1000)  # 1kHz for easy math

    def test_create_default(self):
        """Test creation with default parameters."""
        gate = ConstantPE(1.0)
        adsr = AdsrGatedPE(gate)
        self.renderer.set_source(adsr)
        assert adsr is not None

    def test_create_with_params(self):
        """Test creation with custom parameters."""
        gate = ConstantPE(1.0)
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.050,   # 50 samples at 1kHz
            decay_time=0.200,    # 200 samples at 1kHz
            sustain_level=0.5,
            release_time=0.300,  # 300 samples at 1kHz
        )
        self.renderer.set_source(adsr)
        assert adsr is not None

    def test_is_pure(self):
        """AdsrGatedPE is not pure (maintains state)."""
        gate = ConstantPE(1.0)
        adsr = AdsrGatedPE(gate)
        assert adsr.is_pure() is False

    def test_channel_count(self):
        """AdsrGatedPE outputs mono control signal."""
        gate = ConstantPE(1.0)
        adsr = AdsrGatedPE(gate)
        assert adsr.channel_count() == 1

    def test_extent_matches_gate(self):
        """ADSR extent matches gate extent (no release tail is added)."""
        gate = CropPE(ConstantPE(1.0), 0, 100)
        adsr = AdsrGatedPE(gate, release_time=0.020)
        extent = adsr.extent()
        assert extent.start == 0
        assert extent.end == 100

    def test_extent_infinite_gate_is_infinite(self):
        gate = ConstantPE(1.0)
        adsr = AdsrGatedPE(gate, release_time=0.020)
        extent = adsr.extent()
        assert extent.start is None
        assert extent.end is None

    def test_state_reset_on_start(self):
        """Test that state resets on _on_start()."""
        gate = ArrayPE([0.0, 1.0, 1.0, 0.0])
        adsr = AdsrGatedPE(gate, attack_time=0.010, decay_time=0.010,
                           sustain_level=0.5, release_time=0.010)
        self.renderer.set_source(adsr)
        self.renderer.start()

        # After start, should be in IDLE state (gate is 0 at sample 0)
        snippet = adsr.render(0, 1)
        assert snippet.data[0, 0] == 0.0

    def test_repr(self):
        """Test string representation contains class name."""
        gate = ConstantPE(1.0)
        adsr = AdsrGatedPE(gate, attack_time=0.010, decay_time=0.100,
                           sustain_level=0.7, release_time=0.200)
        assert "AdsrGatedPE" in repr(adsr)


class TestAdsrGatedPERender:
    """Test AdsrGatedPE rendering behavior."""

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

        # ADSR: 10-sample attack, 20-sample decay, 0.5 sustain, 30-sample release
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.5,
            release_time=0.030,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 1000)
        output = snippet.data[:, 0]

        # Attack phase (samples 0-10): ramp from 0 to 1
        assert abs(output[5] - 0.5) < 0.1    # midpoint
        assert abs(output[10] - 1.0) < 0.1   # peak

        # Decay phase (samples 10-30): ramp from 1.0 to 0.5
        assert abs(output[20] - 0.75) < 0.1  # halfway
        assert abs(output[30] - 0.5) < 0.1   # sustain reached

        # Sustain phase (samples 30-500): hold at 0.5
        assert abs(output[100] - 0.5) < 0.01
        assert abs(output[400] - 0.5) < 0.01

        # Gate falls at sample 500 -> Release
        # At sample 515: 15 samples into 30-sample release from 0.5 → 0.25
        assert abs(output[515] - 0.25) < 0.1
        # At sample 530: release complete → 0.0
        assert abs(output[530]) < 0.1

        # IDLE after release
        assert abs(output[600]) < 0.01
        assert abs(output[999]) < 0.01

    def test_early_release_during_attack(self):
        """Test early release: gate falls during attack phase."""
        # Gate: high for 5 samples (mid-attack), then low
        gate_data = np.concatenate([
            np.ones(5, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
        ])
        gate = ArrayPE(gate_data)

        # 10-sample attack, 20-sample release, sustain_level=0.5
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.5,
            release_time=0.020,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 105)
        output = snippet.data[:, 0]

        # At sample 5: ~0.5 (halfway through 10-sample attack)
        assert abs(output[5] - 0.5) < 0.1

        # Gate falls at sample 5 → release from ~0.5.
        # release_dvdt = -sustain_level / release_samples = -0.5/20 = -0.025
        # Releasing from 0.5 (== sustain_level), so standard timing applies.
        assert abs(output[15] - 0.25) < 0.1  # 10 samples into release
        assert abs(output[25]) < 0.1          # 20 samples into release → 0

    def test_early_release_during_decay(self):
        """Test early release: gate falls during decay phase.

        release_dvdt is fixed at -(sustain_level / release_samples) regardless
        of the actual envelope value when release starts.
        """
        # Gate: high for 15 samples (attack done, mid-decay), then low
        gate_data = np.concatenate([
            np.ones(15, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
        ])
        gate = ArrayPE(gate_data)

        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.5,
            release_time=0.020,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 115)
        output = snippet.data[:, 0]

        # At sample 10: attack complete → 1.0
        assert abs(output[10] - 1.0) < 0.1

        # At sample 15: 5 samples into decay (1.0→0.5 over 20 samples)
        # 1.0 - (5/20)*(1.0-0.5) = 0.875
        assert abs(output[15] - 0.875) < 0.1

        # Gate falls at sample 15 → release starts from ~0.875.
        # release_dvdt = -0.5/20 = -0.025 (fixed, based on sustain_level)
        # At sample 25: 0.875 + 10*(-0.025) = 0.625
        assert abs(output[25] - 0.625) < 0.1

        # Release reaches 0 after 0.875/0.025 = 35 samples → sample 50
        assert abs(output[50]) < 0.1

    def test_retrigger_during_release(self):
        """Test re-triggering: gate rises again during release phase.

        attack_dvdt is fixed at 1/attack_samples per sample, regardless
        of the envelope value at the re-trigger point.
        """
        # Gate: high (0-50), low (50-65), high again (65-115)
        gate_data = np.concatenate([
            np.ones(50, dtype=np.float32),
            np.zeros(15, dtype=np.float32),
            np.ones(50, dtype=np.float32),
        ])
        gate = ArrayPE(gate_data)

        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.5,
            release_time=0.030,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 115)
        output = snippet.data[:, 0]

        # At sample 50: sustain at 0.5
        assert abs(output[50] - 0.5) < 0.1

        # Gate falls at 50 → release from 0.5 (release_dvdt = -0.5/30 ≈ -0.01667)
        # At sample 65: 15 samples into release → 0.5*(1-15/30) = 0.25
        assert abs(output[65] - 0.25) < 0.1

        # Gate rises at 65 → re-trigger attack from 0.25.
        # attack_dvdt = 0.1/sample (fixed, not proportional to distance from 1.0)
        # At sample 70: 0.25 + 5*0.1 = 0.75
        assert abs(output[70] - 0.75) < 0.1

        # At sample 75: 0.25 + 10*0.1 = 1.25 → clamped to 1.0
        assert abs(output[75] - 1.0) < 0.1

    def test_idle_state(self):
        """Test that output is zero when gate is always low."""
        gate = ArrayPE([0.0] * 100)
        adsr = AdsrGatedPE(gate, attack_time=0.010, decay_time=0.020,
                           sustain_level=0.5, release_time=0.030)

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 100)
        output = snippet.data[:, 0]

        np.testing.assert_array_almost_equal(output, np.zeros(100), decimal=5)

    def test_sustain_holds_until_gate_falls(self):
        """Test that sustain phase holds until gate signal falls."""
        gate = ArrayPE([1.0] * 200)
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.6,
            release_time=0.030,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 200)
        output = snippet.data[:, 0]

        # After attack+decay (30 samples), sustain should hold at 0.6
        assert abs(output[50] - 0.6) < 0.01
        assert abs(output[100] - 0.6) < 0.01
        assert abs(output[199] - 0.6) < 0.01


class TestAdsrGatedPESampleAccurate:
    """Test sample-accurate behavior of AdsrGatedPE."""

    def setup_method(self):
        import pygmu2 as pg
        pg.set_sample_rate(1000)
        self.renderer = NullRenderer(sample_rate=1000)  # 1kHz for easy math

    def test_precise_attack_ramp(self):
        """Test precise attack ramp values."""
        gate = ArrayPE([1.0] * 20)
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.5,
            release_time=0.030,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 20)
        output = snippet.data[:, 0]

        # Linear ramp from 0 to 1 over 10 samples (dvdt = 0.1/sample)
        assert abs(output[0]) < 0.01         # starts at 0
        assert abs(output[5] - 0.5) < 0.01  # midpoint
        assert abs(output[10] - 1.0) < 0.01 # peak

    def test_precise_decay_ramp(self):
        """Test precise decay ramp values."""
        gate = ArrayPE([1.0] * 40)
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.5,
            release_time=0.030,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 40)
        output = snippet.data[:, 0]

        # Attack completes at cursor 10 (emits ~1.0), decay runs from cursor 11.
        # Due to float accumulation, attack takes 11 steps (not 10).
        assert abs(output[10] - 1.0) < 0.01  # near-1.0 pre-clamp value
        assert abs(output[11] - 1.0) < 0.01  # exact 1.0 after clamp
        assert abs(output[21] - 0.75) < 0.01 # midpoint (decay step 9 → cursor 21)
        assert abs(output[31] - 0.5) < 0.01  # sustain reached (decay step 19 → cursor 31)

    def test_precise_release_ramp(self):
        """Test precise release ramp values."""
        gate_data = np.concatenate([
            np.ones(50, dtype=np.float32),
            np.zeros(50, dtype=np.float32),
        ])
        gate = ArrayPE(gate_data)
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.5,
            release_time=0.030,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 100)
        output = snippet.data[:, 0]

        # Release: 0.5 → 0.0 over 30 samples starting at sample 50
        assert abs(output[50] - 0.5) < 0.01  # release starts
        assert abs(output[65] - 0.25) < 0.01 # midpoint (15/30)
        assert abs(output[80]) < 0.01         # release complete


class TestAdsrGatedPEEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        import pygmu2 as pg
        pg.set_sample_rate(1000)
        self.renderer = NullRenderer(sample_rate=1000)

    def test_zero_sustain_level(self):
        """Test ADSR with zero sustain level."""
        gate = ArrayPE([1.0] * 50)
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.0,
            release_time=0.030,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 50)
        output = snippet.data[:, 0]

        # Attack takes 11 slots (output-then-update + float accumulation),
        # decay 20 more → sustain level first emitted at cursor 31.
        assert abs(output[31]) < 0.01
        assert abs(output[49]) < 0.01

    def test_unit_sustain_level(self):
        """Test ADSR with sustain level of 1.0 (decay is a no-op)."""
        gate = ArrayPE([1.0] * 50)
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=1.0,
            release_time=0.030,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 50)
        output = snippet.data[:, 0]

        assert abs(output[10] - 1.0) < 0.01
        assert abs(output[30] - 1.0) < 0.01
        assert abs(output[49] - 1.0) < 0.01

    def test_very_short_times(self):
        """Test ADSR with 1ms phases (= 1 sample at 1kHz)."""
        gate = ArrayPE([1.0] * 10)
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.001,
            decay_time=0.001,
            sustain_level=0.5,
            release_time=0.001,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 10)
        output = snippet.data[:, 0]

        # Sample 0: attack starts (emits 0)
        # Sample 1: emit 1.0 (attack peak), decay completes in same step
        # Sample 2: emit 0.5 (sustain)
        assert abs(output[2] - 0.5) < 0.1

    def test_rapid_gate_changes(self):
        """Test rapid gate on/off changes don't crash and stay in [0,1]."""
        gate_data = np.array([
            1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0
        ], dtype=np.float32)
        gate = ArrayPE(gate_data)
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.5,
            release_time=0.010,
        )

        self.renderer.set_source(adsr)
        self.renderer.start()

        snippet = adsr.render(0, 10)
        output = snippet.data[:, 0]

        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)
