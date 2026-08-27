"""
Tests for AdsrGatedPE, AdsrTriggeredPE, and the _generate_ramp helper.

At 1 kHz (SR=1000), 1 ms == 1 sample — convenient for exact arithmetic.

Timing reference for most tests
  (attack=10ms, decay=20ms, sustain_level=0.5, release=30ms):

  AdsrGatedPE (gate high for N samples, then low):
    dvdt_attack  =  0.1 / sample
    dvdt_decay   = -0.025 / sample   (1.0 → 0.5 in 20 samples)
    dvdt_release = -1/60  / sample   (0.5 → 0.0 in 30 samples)
    out[k]       =  k * 0.1          for k = 0..9   (attack)
    out[10+k]    =  1.0 - k * 0.025  for k = 0..19  (decay)
    out[30..]    =  0.5              while gate high  (sustain)
    out[G+k]     =  0.5 - k / 60    for k = 0..29   (release, gate fell at G)
    out[G+30+..] =  0.0                              (idle)

  AdsrTriggeredPE (trigger at sample 0,
                    attack=10ms, decay=20ms, sustain_time=10ms, release=30ms):
    attack  : samples  0-9
    decay   : samples 10-29
    sustain : samples 30-39  (10 samples, timed)
    release : samples 40-69  (30 samples)
    idle    : sample 70+

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np
import pytest
import pygmu2 as pg
from pygmu2 import ArrayPE, ConstantPE, CropPE, NullRenderer
from pygmu2.adsr_pe import AdsrGatedPE, AdsrTriggeredPE, _generate_ramp

SR = 1000  # 1 kHz — 1 ms == 1 sample


def setup_renderer():
    pg.set_sample_rate(SR)
    return NullRenderer(sample_rate=SR)


def gate_array(*segments):
    """Build a 1-D float32 gate from (value, count) pairs."""
    return np.concatenate([np.full(n, v, dtype=np.float32) for v, n in segments])


def trigger_array(length, *positions):
    """Build a 1-D float32 trigger array with impulses at the given positions."""
    arr = np.zeros(length, dtype=np.float32)
    for p in positions:
        arr[p] = 1.0
    return arr


# ---------------------------------------------------------------------------
# _generate_ramp  (module-level helper)
# ---------------------------------------------------------------------------


class TestGenerateRamp:

    def test_flat_ramp(self):
        out = np.zeros(10, dtype=np.float32)
        cursor, env = _generate_ramp(out, 0.5, 0.0, 0, 10)
        np.testing.assert_allclose(out, 0.5, atol=1e-6)
        assert cursor == 10
        assert abs(env - 0.5) < 1e-6

    def test_rising_ramp(self):
        out = np.zeros(5, dtype=np.float32)
        _generate_ramp(out, 0.0, 0.1, 0, 5)
        np.testing.assert_allclose(out, [0.0, 0.1, 0.2, 0.3, 0.4], atol=1e-6)

    def test_falling_ramp(self):
        out = np.zeros(5, dtype=np.float32)
        _generate_ramp(out, 1.0, -0.1, 0, 5)
        np.testing.assert_allclose(out, [1.0, 0.9, 0.8, 0.7, 0.6], atol=1e-5)

    def test_offset_write(self):
        """Only writes to output[offset:offset+length]."""
        out = np.zeros(10, dtype=np.float32)
        _generate_ramp(out, 0.5, 0.1, 3, 4)
        np.testing.assert_array_equal(out[:3], 0.0)
        np.testing.assert_array_equal(out[7:], 0.0)
        np.testing.assert_allclose(out[3:7], [0.5, 0.6, 0.7, 0.8], atol=1e-5)

    def test_returns_next_cursor_and_env(self):
        out = np.zeros(10, dtype=np.float32)
        cursor, env = _generate_ramp(out, 1.0, -0.1, 2, 5)
        assert cursor == 7
        assert abs(env - 0.5) < 1e-6  # 1.0 + 5 * (-0.1)


# ---------------------------------------------------------------------------
# AdsrGatedPE — properties
# ---------------------------------------------------------------------------


class TestAdsrGatedPEBasics:

    def setup_method(self):
        self.renderer = setup_renderer()

    def test_stateful(self):
        adsr = AdsrGatedPE(ConstantPE(1.0))
        assert adsr.stateful

    def test_channel_count(self):
        adsr = AdsrGatedPE(ConstantPE(1.0))
        assert adsr.channel_count() == 1

    def test_extent_matches_gate(self):
        gate = CropPE(ConstantPE(1.0), 0, 100)
        adsr = AdsrGatedPE(gate)
        ext = adsr.extent()
        assert ext.start == 0
        assert ext.end == 100

    def test_extent_infinite_gate_is_infinite(self):
        adsr = AdsrGatedPE(ConstantPE(1.0))
        ext = adsr.extent()
        assert ext.start is None and ext.end is None

    def test_inputs_contains_gate(self):
        gate = ConstantPE(1.0)
        adsr = AdsrGatedPE(gate)
        assert gate in adsr.inputs()


# ---------------------------------------------------------------------------
# AdsrGatedPE — envelope shape (loose tolerance)
# ---------------------------------------------------------------------------


class TestAdsrGatedPERender:

    def setup_method(self):
        self.renderer = setup_renderer()

    def _make_adsr(self, gate_data):
        gate = ArrayPE(gate_data)
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,  # 10 samples
            decay_time=0.020,  # 20 samples
            sustain_level=0.5,
            release_time=0.030,  # 30 samples
        )
        self.renderer.set_source(adsr)
        self.renderer.start()
        return adsr

    def _out(self, adsr, start, duration):
        return adsr.render(start, duration).data.ravel()

    def test_idle_all_zeros(self):
        """Gate always low → output is all zeros."""
        adsr = self._make_adsr(gate_array((0, 100)))
        np.testing.assert_array_equal(self._out(adsr, 0, 100), 0.0)

    def test_attack_rises(self):
        """Attack phase ramps from 0 toward 1."""
        adsr = self._make_adsr(gate_array((1, 100)))
        out = self._out(adsr, 0, 100)
        assert out[0] == 0.0
        assert out[5] > 0.3
        assert abs(out[10] - 1.0) < 0.01

    def test_sustain_holds(self):
        """Sustain phase holds at sustain_level while gate is high."""
        adsr = self._make_adsr(gate_array((1, 200)))
        out = self._out(adsr, 0, 200)
        assert abs(out[50] - 0.5) < 0.01
        assert abs(out[100] - 0.5) < 0.01
        assert abs(out[199] - 0.5) < 0.01

    def test_release_falls_to_zero(self):
        """Release phase falls from sustain_level to zero."""
        adsr = self._make_adsr(gate_array((1, 50), (0, 50)))
        out = self._out(adsr, 0, 100)
        assert abs(out[50] - 0.5) < 0.01  # release start
        assert abs(out[65] - 0.25) < 0.05  # midpoint
        assert abs(out[80]) < 0.01  # complete

    def test_idle_after_release(self):
        """Envelope stays at zero after release completes."""
        adsr = self._make_adsr(gate_array((1, 50), (0, 100)))
        out = self._out(adsr, 0, 150)
        assert abs(out[90]) < 0.01
        assert abs(out[149]) < 0.01

    def test_complete_adsr_cycle(self):
        """Full ADSR cycle stays in [0, 1] and hits expected waypoints."""
        adsr = self._make_adsr(gate_array((1, 50), (0, 80)))
        out = self._out(adsr, 0, 130)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
        assert abs(out[10] - 1.0) < 0.01  # attack peak → decay starts
        assert abs(out[30] - 0.5) < 0.01  # sustain starts
        assert abs(out[50] - 0.5) < 0.01  # gate falls here
        assert abs(out[65] - 0.25) < 0.05  # release midpoint
        assert out[85] < 0.01  # idle

    def test_early_release_during_attack(self):
        """Gate falls during attack; release starts from partial attack level."""
        adsr = self._make_adsr(gate_array((1, 5), (0, 60)))
        out = self._out(adsr, 0, 65)
        # 5 samples into 10-sample attack → env ≈ 0.5
        assert abs(out[5] - 0.5) < 0.1
        # release_dvdt = -0.5/30; 10 samples in → ~0.25
        assert abs(out[15] - 0.25) < 0.1
        assert out[35] < 0.05

    def test_early_release_during_decay(self):
        """Gate falls during decay; release starts from partial decay level."""
        adsr = self._make_adsr(gate_array((1, 15), (0, 80)))
        out = self._out(adsr, 0, 95)
        # 5 samples into decay from 1.0 → 1.0 - 5*0.025 = 0.875
        assert abs(out[15] - 0.875) < 0.05
        # release from 0.875, dvdt=-0.5/30; 10 samples in → 0.875 - 10/60 ≈ 0.708
        assert abs(out[25] - 0.708) < 0.05
        # release from 0.875 takes ceil(52.5)=53 samples → idle at sample 68
        assert out[70] < 0.05

    def test_retrigger_during_release(self):
        """Gate rises again during release; attack restarts from current env."""
        adsr = self._make_adsr(gate_array((1, 50), (0, 15), (1, 60)))
        out = self._out(adsr, 0, 125)
        # 15 samples into 30-sample release from 0.5 → ~0.25
        assert abs(out[65] - 0.25) < 0.1
        # attack rises at dvdt=0.1 from 0.25; 5 samples later → 0.75
        assert abs(out[70] - 0.75) < 0.1
        # attack from 0.25 takes ceil(7.5)=8 samples; peak at cursor 73
        assert abs(out[73] - 1.0) < 0.01

    def test_output_always_in_range(self):
        """Output never leaves [0, 1] under rapid gate changes."""
        gate = ArrayPE(np.tile([1.0, 0.0], 25).astype(np.float32))
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.5,
            release_time=0.010,
        )
        self.renderer.set_source(adsr)
        self.renderer.start()
        out = adsr.render(0, 50).data
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_state_persists_across_buffers(self):
        """Envelope state carries correctly across multiple render calls."""
        # gate high for 100 samples, then low for 50
        adsr = self._make_adsr(gate_array((1, 100), (0, 50)))

        out_a = self._out(adsr, 0, 50)  # attack + decay + sustain start
        out_b = self._out(adsr, 50, 50)  # sustain continues (gate still high)
        out_c = self._out(adsr, 100, 30)  # gate falls → release

        assert abs(out_a[30] - 0.5) < 0.01  # sustain reached
        assert abs(out_b[0] - 0.5) < 0.01  # still sustaining
        assert abs(out_b[49] - 0.5) < 0.01  # still sustaining at buffer end
        # release starts at sample 100 (out_c cursor 0)
        assert abs(out_c[0] - 0.5) < 0.01  # release just started
        assert abs(out_c[15] - 0.25) < 0.05  # midpoint


# ---------------------------------------------------------------------------
# AdsrGatedPE — sample-accurate values (arange-based semantics)
# ---------------------------------------------------------------------------


class TestAdsrGatedPESampleAccurate:
    """
    Pin exact output values using the arange formula:
        out[offset + i] = env0 + i * dvdt   for i = 0..n-1

    At 1 kHz, attack=10ms, decay=20ms, sustain=0.5, release=30ms:
        dvdt_attack  =  0.1       →  T=10, n=10, attack samples 0-9
        dvdt_decay   = -0.025     →  T=20, n=20, decay samples 10-29
        dvdt_release = -1/60      →  T=30, n=30, release samples G..G+29
    """

    def setup_method(self):
        self.renderer = setup_renderer()

    def _adsr(self, gate_data):
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
        return adsr

    def test_attack_sample_values(self):
        """out[k] = k * 0.1  for k = 0..9."""
        out = self._adsr(gate_array((1, 15))).render(0, 15).data.ravel()
        for k in range(10):
            assert abs(out[k] - k * 0.1) < 1e-5, f"out[{k}]={out[k]}, expected {k*0.1}"

    def test_first_decay_sample_is_one(self):
        """Attack completes exactly at sample 10; out[10] = 1.0."""
        out = self._adsr(gate_array((1, 15))).render(0, 15).data.ravel()
        assert abs(out[10] - 1.0) < 1e-5

    def test_decay_sample_values(self):
        """out[10+k] = 1.0 - k * 0.025  for k = 0..19."""
        out = self._adsr(gate_array((1, 35))).render(0, 35).data.ravel()
        for k in range(20):
            expected = 1.0 - k * 0.025
            assert (
                abs(out[10 + k] - expected) < 1e-5
            ), f"out[{10+k}]={out[10+k]}, expected {expected}"

    def test_first_sustain_sample(self):
        """Decay completes at sample 30; out[30] = 0.5 exactly."""
        out = self._adsr(gate_array((1, 35))).render(0, 35).data.ravel()
        assert abs(out[30] - 0.5) < 1e-5

    def test_sustain_fills_remaining_gate_segment(self):
        """Sustain holds exactly at 0.5 for every sample while gate is high."""
        out = self._adsr(gate_array((1, 80))).render(0, 80).data.ravel()
        np.testing.assert_allclose(out[30:80], 0.5, atol=1e-5)

    def test_release_sample_values(self):
        """out[50+k] = 0.5 - k * (0.5/30)  for k = 0..29  (gate fell at 50)."""
        out = self._adsr(gate_array((1, 50), (0, 50))).render(0, 100).data.ravel()
        dvdt = 0.5 / 30
        for k in range(30):
            expected = 0.5 - k * dvdt
            assert (
                abs(out[50 + k] - expected) < 1e-5
            ), f"out[{50+k}]={out[50+k]}, expected {expected}"

    def test_idle_starts_at_sample_80(self):
        """Release completes at sample 80; output is 0.0 from there on."""
        out = self._adsr(gate_array((1, 50), (0, 60))).render(0, 110).data.ravel()
        assert abs(out[80]) < 1e-5
        assert abs(out[109]) < 1e-5


# ---------------------------------------------------------------------------
# AdsrGatedPE — edge cases
# ---------------------------------------------------------------------------


class TestAdsrGatedPEEdgeCases:

    def setup_method(self):
        self.renderer = setup_renderer()

    def test_zero_sustain_level(self):
        """sustain_level=0: decay ramps all the way to zero."""
        gate = ArrayPE(gate_array((1, 50)))
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.0,
            release_time=0.030,
        )
        self.renderer.set_source(adsr)
        self.renderer.start()
        out = adsr.render(0, 50).data.ravel()
        assert abs(out[30]) < 0.01  # decay complete → 0
        assert abs(out[49]) < 0.01

    def test_unit_sustain_level(self):
        """sustain_level=1.0: decay is a no-op; output stays at 1.0."""
        gate = ArrayPE(gate_array((1, 50)))
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=1.0,
            release_time=0.030,
        )
        self.renderer.set_source(adsr)
        self.renderer.start()
        out = adsr.render(0, 50).data.ravel()
        assert abs(out[10] - 1.0) < 0.01
        assert abs(out[30] - 1.0) < 0.01
        assert abs(out[49] - 1.0) < 0.01

    def test_single_sample_gate_high(self):
        """Gate high for exactly one sample; no crash, output in [0, 1]."""
        gate = ArrayPE(gate_array((1, 1), (0, 60)))
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.5,
            release_time=0.030,
        )
        self.renderer.set_source(adsr)
        self.renderer.start()
        out = adsr.render(0, 61).data
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_rapid_gate_changes(self):
        """Rapid alternating gate doesn't crash; output stays in [0, 1]."""
        gate = ArrayPE(np.tile([1.0, 0.0], 25).astype(np.float32))
        adsr = AdsrGatedPE(
            gate,
            attack_time=0.010,
            decay_time=0.020,
            sustain_level=0.5,
            release_time=0.010,
        )
        self.renderer.set_source(adsr)
        self.renderer.start()
        out = adsr.render(0, 50).data
        assert np.all(out >= 0.0) and np.all(out <= 1.0)


# ---------------------------------------------------------------------------
# AdsrTriggeredPE — properties
# ---------------------------------------------------------------------------


class TestAdsrTriggeredPEBasics:

    def setup_method(self):
        self.renderer = setup_renderer()

    def test_stateful(self):
        adsr = AdsrTriggeredPE(ArrayPE(np.zeros(10, dtype=np.float32)))
        assert adsr.stateful

    def test_channel_count(self):
        adsr = AdsrTriggeredPE(ArrayPE(np.zeros(10, dtype=np.float32)))
        assert adsr.channel_count() == 1

    def test_extent_matches_trigger(self):
        trigger = CropPE(ConstantPE(0.0), 0, 100)
        adsr = AdsrTriggeredPE(trigger)
        ext = adsr.extent()
        assert ext.start == 0
        assert ext.end == 100

    def test_inputs_contains_trigger(self):
        trigger = ArrayPE(np.zeros(10, dtype=np.float32))
        adsr = AdsrTriggeredPE(trigger)
        assert trigger in adsr.inputs()


# ---------------------------------------------------------------------------
# AdsrTriggeredPE — envelope shape (loose tolerance)
# ---------------------------------------------------------------------------


class TestAdsrTriggeredPERender:

    def setup_method(self):
        self.renderer = setup_renderer()

    def _make_adsr(self, trig_data, sustain_time=0.010):
        trigger = ArrayPE(trig_data)
        adsr = AdsrTriggeredPE(
            trigger,
            attack_time=0.010,  # 10 samples
            decay_time=0.020,  # 20 samples
            sustain_time=sustain_time,
            sustain_level=0.5,
            release_time=0.030,  # 30 samples
        )
        self.renderer.set_source(adsr)
        self.renderer.start()
        return adsr

    def _out(self, adsr, start, duration):
        return adsr.render(start, duration).data.ravel()

    def test_idle_no_trigger(self):
        """No trigger → output is all zeros."""
        adsr = self._make_adsr(trigger_array(100))
        np.testing.assert_array_equal(self._out(adsr, 0, 100), 0.0)

    def test_complete_cycle(self):
        """Single trigger: attack→decay→sustain→release→idle; stays in [0,1]."""
        adsr = self._make_adsr(trigger_array(120, 0))
        out = self._out(adsr, 0, 120)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
        assert abs(out[10] - 1.0) < 0.01  # decay starts
        assert abs(out[30] - 0.5) < 0.01  # sustain starts
        assert abs(out[39] - 0.5) < 0.01  # sustain still held
        assert abs(out[40] - 0.5) < 0.01  # release starts (env₀=0.5)
        assert abs(out[55] - 0.25) < 0.05  # release midpoint
        assert out[70] < 0.01  # idle

    def test_sustain_is_timed(self):
        """Sustain expires automatically (no gate required)."""
        # sustain_time=20ms → sustain samples 30-49, release at 50
        adsr = self._make_adsr(trigger_array(100, 0), sustain_time=0.020)
        out = self._out(adsr, 0, 100)
        assert abs(out[35] - 0.5) < 0.01  # mid-sustain
        assert abs(out[50] - 0.5) < 0.01  # release starts
        assert abs(out[65] - 0.25) < 0.05  # release midpoint
        assert out[80] < 0.01  # idle

    def test_retrigger_during_release(self):
        """Trigger during release restarts attack from current env level."""
        # First trigger at 0; second at 60 (20 samples into 30-sample release)
        adsr = self._make_adsr(trigger_array(150, 0, 60))
        out = self._out(adsr, 0, 150)
        assert out[60] < 0.5  # was releasing
        # after retrigger, attack rises: expect a peak ≈ 1.0 within 10 more samples
        assert np.max(out[60:75]) > 0.9

    def test_retrigger_during_attack(self):
        """Trigger during attack restarts attack from current env level."""
        # env=0.5 at sample 5; retrigger → attack from 0.5
        adsr = self._make_adsr(trigger_array(60, 0, 5))
        out = self._out(adsr, 0, 60)
        # attack from 0.5 at dvdt=0.1: T=5, completes at sample 10
        assert abs(out[10] - 1.0) < 0.05
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_retrigger_during_sustain(self):
        """Trigger during sustain restarts the cycle."""
        # sustain runs 30-39; retrigger at 35 → attack from env=0.5
        adsr = self._make_adsr(trigger_array(150, 0, 35))
        out = self._out(adsr, 0, 150)
        assert abs(out[35] - 0.5) < 0.05  # env at retrigger
        # attack from 0.5, T=5 → decay starts at sample 40
        assert abs(out[40] - 1.0) < 0.1
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_output_always_in_range(self):
        """Multiple rapid triggers; output never leaves [0, 1]."""
        adsr = self._make_adsr(trigger_array(200, 0, 25, 60, 100, 150))
        out = adsr.render(0, 200).data
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_state_persists_across_buffers(self):
        """Envelope state carries correctly across multiple render calls."""
        adsr = self._make_adsr(trigger_array(200, 0))
        out_a = self._out(adsr, 0, 50)  # attack + decay + sustain start
        out_b = self._out(adsr, 50, 50)  # release (sustain ends at 40)

        assert abs(out_a[10] - 1.0) < 0.01  # attack peak
        assert abs(out_a[30] - 0.5) < 0.01  # sustain starts
        # Buffer B starts at sample 50, which is 10 samples into release
        # env = 0.5 - 10*(0.5/30) ≈ 0.333
        assert 0.2 < out_b[0] < 0.5


# ---------------------------------------------------------------------------
# AdsrTriggeredPE — sample-accurate values
# ---------------------------------------------------------------------------


class TestAdsrTriggeredPESampleAccurate:
    """
    Pin exact output at 1kHz, trigger at sample 0:
        attack=10ms, decay=20ms, sustain_time=10ms, sustain_level=0.5, release=30ms

        attack  (0- 9): out[k]    = k * 0.1
        decay  (10-29): out[10+k] = 1.0 - k * 0.025
        sustain(30-39): out[30..39] == 0.5
        release(40-69): out[40+k] = 0.5 - k * (0.5/30)
        idle   (70+  ): out == 0.0
    """

    def setup_method(self):
        self.renderer = setup_renderer()

    def _adsr(self, trig_data):
        trigger = ArrayPE(trig_data)
        adsr = AdsrTriggeredPE(
            trigger,
            attack_time=0.010,
            decay_time=0.020,
            sustain_time=0.010,
            sustain_level=0.5,
            release_time=0.030,
        )
        self.renderer.set_source(adsr)
        self.renderer.start()
        return adsr

    def test_attack_sample_values(self):
        out = self._adsr(trigger_array(15, 0)).render(0, 15).data.ravel()
        for k in range(10):
            assert abs(out[k] - k * 0.1) < 1e-5, f"out[{k}]={out[k]}, expected {k*0.1}"

    def test_first_decay_sample(self):
        out = self._adsr(trigger_array(15, 0)).render(0, 15).data.ravel()
        assert abs(out[10] - 1.0) < 1e-5

    def test_decay_sample_values(self):
        out = self._adsr(trigger_array(35, 0)).render(0, 35).data.ravel()
        for k in range(20):
            expected = 1.0 - k * 0.025
            assert (
                abs(out[10 + k] - expected) < 1e-5
            ), f"out[{10+k}]={out[10+k]}, expected {expected}"

    def test_sustain_sample_values(self):
        out = self._adsr(trigger_array(45, 0)).render(0, 45).data.ravel()
        np.testing.assert_allclose(out[30:40], 0.5, atol=1e-5)

    def test_release_sample_values(self):
        out = self._adsr(trigger_array(75, 0)).render(0, 75).data.ravel()
        dvdt = 0.5 / 30
        for k in range(30):
            expected = 0.5 - k * dvdt
            assert (
                abs(out[40 + k] - expected) < 1e-5
            ), f"out[{40+k}]={out[40+k]}, expected {expected}"

    def test_idle_starts_at_sample_70(self):
        out = self._adsr(trigger_array(100, 0)).render(0, 100).data.ravel()
        assert abs(out[70]) < 1e-5
        assert abs(out[99]) < 1e-5


# ---------------------------------------------------------------------------
# AdsrTriggeredPE — edge cases
# ---------------------------------------------------------------------------


class TestAdsrTriggeredPEEdgeCases:

    def setup_method(self):
        self.renderer = setup_renderer()

    def test_zero_sustain_time(self):
        """sustain_time=0: sustain phase is skipped; release follows decay."""
        trigger = ArrayPE(trigger_array(80, 0))
        adsr = AdsrTriggeredPE(
            trigger,
            attack_time=0.010,
            decay_time=0.020,
            sustain_time=0.0,
            sustain_level=0.5,
            release_time=0.030,
        )
        self.renderer.set_source(adsr)
        self.renderer.start()
        out = adsr.render(0, 80).data
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
        # release should complete by sample 30+30=60
        assert out.ravel()[65] < 0.1

    def test_rapid_triggers(self):
        """Triggers every 5 samples; no crash, output stays in [0, 1]."""
        trig = trigger_array(50, *range(0, 50, 5))
        trigger = ArrayPE(trig)
        adsr = AdsrTriggeredPE(
            trigger,
            attack_time=0.010,
            decay_time=0.020,
            sustain_time=0.010,
            sustain_level=0.5,
            release_time=0.010,
        )
        self.renderer.set_source(adsr)
        self.renderer.start()
        out = adsr.render(0, 50).data
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_trigger_at_last_sample(self):
        """Trigger at the last sample of a buffer doesn't crash."""
        trig = trigger_array(50, 49)
        trigger = ArrayPE(trig)
        adsr = AdsrTriggeredPE(
            trigger,
            attack_time=0.010,
            decay_time=0.020,
            sustain_time=0.010,
            sustain_level=0.5,
            release_time=0.010,
        )
        self.renderer.set_source(adsr)
        self.renderer.start()
        out = adsr.render(0, 50).data
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
