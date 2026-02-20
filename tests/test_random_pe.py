"""Tests for RandomPE."""

import numpy as np
import pytest

pytest.importorskip("pygmu2.random_pe", reason="random_pe module not yet implemented")

from pygmu2 import NullRenderer, SinePE, ConstantPE, GainPE, MixPE
from pygmu2.random_pe import RandomPE, RandomMode
from pygmu2.extent import Extent


class TestRandomPEBasics:
    """Test creation and properties."""
    
    def test_create_default(self):
        pe = RandomPE()
        assert pe.rate == 1.0
        assert pe.min_value == 0.0
        assert pe.max_value == 1.0
        assert pe.mode == RandomMode.SAMPLE_HOLD
        assert pe.seed is None
        assert pe.trigger is None
    
    def test_create_custom(self):
        pe = RandomPE(
            rate=4.0,
            min_value=-1.0,
            max_value=1.0,
            mode=RandomMode.SMOOTH,
            seed=12345,
            slew=0.05,
        )
        assert pe.rate == 4.0
        assert pe.min_value == -1.0
        assert pe.max_value == 1.0
        assert pe.mode == RandomMode.SMOOTH
        assert pe.seed == 12345
        assert pe.slew == 0.05
    
    def test_create_with_trigger(self):
        trigger = SinePE(frequency=2.0)
        pe = RandomPE(trigger=trigger)
        assert pe.trigger is trigger
        assert trigger in pe.inputs()
    
    def test_inputs_without_trigger(self):
        pe = RandomPE()
        assert pe.inputs() == []
    
    def test_inputs_with_trigger(self):
        trigger = SinePE(frequency=2.0)
        pe = RandomPE(trigger=trigger)
        assert pe.inputs() == [trigger]
    
    def test_is_pure(self):
        pe = RandomPE()
        assert pe.is_pure() is False
    
    def test_channel_count(self):
        pe = RandomPE()
        assert pe.channel_count() == 1
    
    def test_extent_is_infinite(self):
        pe = RandomPE()
        extent = pe.extent()
        assert extent.start is None
        assert extent.end is None
    
    def test_repr(self):
        pe = RandomPE(rate=2.0, min_value=0, max_value=10, mode=RandomMode.LINEAR)
        repr_str = repr(pe)
        assert "RandomPE" in repr_str
        assert "linear" in repr_str
        assert "rate=2.0" in repr_str


class TestRandomPERender:
    """Test rendering behavior."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_render_returns_snippet(self, renderer):
        pe = RandomPE(seed=42)
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 1000)
        
        assert snippet.start == 0
        assert snippet.data.shape == (1000, 1)
        assert snippet.data.dtype == np.float32
        renderer.stop()
    
    def test_render_zero_duration(self, renderer):
        pe = RandomPE(seed=42)
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 0)
        
        assert snippet.data.shape == (0, 1)
        renderer.stop()
    
    def test_values_in_range(self, renderer):
        pe = RandomPE(rate=100, min_value=0.5, max_value=0.8, seed=42)
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 10000)
        
        assert np.all(snippet.data >= 0.5)
        assert np.all(snippet.data <= 0.8)
        renderer.stop()
    
    def test_seed_reproducibility(self, renderer):
        """Same seed should produce identical output."""
        pe1 = RandomPE(rate=10, seed=12345)
        pe2 = RandomPE(rate=10, seed=12345)
        
        renderer.set_source(pe1)
        renderer.start()
        snippet1 = pe1.render(0, 1000)
        renderer.stop()
        
        renderer.set_source(pe2)
        renderer.start()
        snippet2 = pe2.render(0, 1000)
        renderer.stop()
        
        np.testing.assert_array_equal(snippet1.data, snippet2.data)
    
    def test_different_seeds_differ(self, renderer):
        """Different seeds should produce different output."""
        pe1 = RandomPE(rate=10, seed=111)
        pe2 = RandomPE(rate=10, seed=222)
        
        renderer.set_source(pe1)
        renderer.start()
        snippet1 = pe1.render(0, 1000)
        renderer.stop()
        
        renderer.set_source(pe2)
        renderer.start()
        snippet2 = pe2.render(0, 1000)
        renderer.stop()
        
        assert not np.allclose(snippet1.data, snippet2.data)


class TestRandomModes:
    """Test different interpolation modes."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_sample_hold_has_steps(self, renderer):
        """Sample-hold should have discrete jumps."""
        pe = RandomPE(rate=10, mode=RandomMode.SAMPLE_HOLD, seed=42)
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 44100)  # 1 second at 10 Hz = ~10 values
        data = snippet.data[:, 0]
        
        # Count unique values (should be relatively few for S&H)
        unique_values = np.unique(data)
        # At 10 Hz for 1 second, expect around 10-11 unique values
        assert len(unique_values) <= 15
        renderer.stop()
    
    def test_linear_is_continuous(self, renderer):
        """Linear mode should have smooth ramps."""
        pe = RandomPE(rate=2, mode=RandomMode.LINEAR, seed=42)
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 44100)
        data = snippet.data[:, 0]
        
        # Check that consecutive samples don't have huge jumps
        # (except at period boundaries)
        diffs = np.abs(np.diff(data))
        # Most diffs should be small (linear interpolation)
        small_diffs = diffs[diffs < 0.01]
        assert len(small_diffs) > len(diffs) * 0.9  # 90% should be small
        renderer.stop()
    
    def test_smooth_is_continuous(self, renderer):
        """Smooth mode should be continuous with smooth transitions."""
        pe = RandomPE(rate=2, mode=RandomMode.SMOOTH, seed=42)
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 44100)
        data = snippet.data[:, 0]
        
        # Check continuity
        diffs = np.abs(np.diff(data))
        max_diff = np.max(diffs)
        # Smooth mode shouldn't have huge jumps
        assert max_diff < 0.1
        renderer.stop()
    
    def test_walk_stays_in_bounds(self, renderer):
        """Walk mode should stay within min/max bounds."""
        pe = RandomPE(
            rate=100,
            min_value=-0.5,
            max_value=0.5,
            mode=RandomMode.WALK,
            slew=0.1,  # Large slew to stress test bounds
            seed=42
        )
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 44100)
        data = snippet.data[:, 0]
        
        assert np.all(data >= -0.5)
        assert np.all(data <= 0.5)
        renderer.stop()
    
    def test_walk_has_small_steps(self, renderer):
        """Walk mode should take small steps."""
        pe = RandomPE(
            min_value=0,
            max_value=1,
            mode=RandomMode.WALK,
            slew=0.001,  # Very small steps
            seed=42
        )
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 1000)
        data = snippet.data[:, 0]
        
        # Check that steps are bounded by slew
        diffs = np.abs(np.diff(data))
        max_step = 1.0 * 0.001  # range * slew
        assert np.all(diffs <= max_step * 1.01)  # Small tolerance
        renderer.stop()


class TestRandomPETrigger:
    """Test trigger-based random generation."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_trigger_changes_value(self, renderer):
        """Value should change on rising edge of trigger."""
        # Create a trigger that goes positive periodically
        # Using a slow sine that we'll detect rising edges on
        trigger = SinePE(frequency=10.0)  # 10 Hz trigger
        
        pe = RandomPE(
            min_value=0,
            max_value=100,
            mode=RandomMode.SAMPLE_HOLD,
            trigger=trigger,
            seed=42
        )
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 44100)  # 1 second
        data = snippet.data[:, 0]
        
        # Should have multiple distinct values from trigger edges
        unique_values = np.unique(data)
        # 10 Hz for 1 second = ~10 rising edges = ~10-11 values
        assert len(unique_values) >= 5
        assert len(unique_values) <= 15
        renderer.stop()
    
    def test_trigger_respects_seed(self, renderer):
        """Triggered random with seed should be reproducible."""
        trigger1 = SinePE(frequency=5.0)
        trigger2 = SinePE(frequency=5.0)
        
        pe1 = RandomPE(trigger=trigger1, seed=999)
        pe2 = RandomPE(trigger=trigger2, seed=999)
        
        renderer.set_source(pe1)
        renderer.start()
        snippet1 = pe1.render(0, 10000)
        renderer.stop()
        
        renderer.set_source(pe2)
        renderer.start()
        snippet2 = pe2.render(0, 10000)
        renderer.stop()
        
        np.testing.assert_array_equal(snippet1.data, snippet2.data)
    
    def test_trigger_walk_resets(self, renderer):
        """Walk mode with trigger should reset on rising edge."""
        # Create trigger that pulses
        trigger = SinePE(frequency=2.0)
        
        pe = RandomPE(
            min_value=0,
            max_value=1,
            mode=RandomMode.WALK,
            trigger=trigger,
            seed=42
        )
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 44100)
        data = snippet.data[:, 0]
        
        # Values should stay in range
        assert np.all(data >= 0)
        assert np.all(data <= 1)
        
        # Should see jumps at trigger points
        diffs = np.abs(np.diff(data))
        # Some diffs should be larger (the resets)
        large_diffs = diffs[diffs > 0.1]
        assert len(large_diffs) > 0  # Should have some resets
        renderer.stop()


class TestRandomPEContinuity:
    """Test state continuity across render calls."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_continuous_across_renders(self, renderer):
        """Output should be continuous across multiple render calls."""
        pe = RandomPE(rate=2, mode=RandomMode.LINEAR, seed=42)
        renderer.set_source(pe)
        renderer.start()
        
        # Render in chunks
        chunk1 = pe.render(0, 1000)
        chunk2 = pe.render(1000, 1000)
        
        # Last sample of chunk1 should connect smoothly to first of chunk2
        last_val = chunk1.data[-1, 0]
        first_val = chunk2.data[0, 0]
        
        # In linear mode, they should be very close
        assert abs(last_val - first_val) < 0.01
        renderer.stop()
    
    def test_walk_continuous_across_renders(self, renderer):
        """Walk mode should be continuous across render calls."""
        pe = RandomPE(mode=RandomMode.WALK, slew=0.001, seed=42)
        renderer.set_source(pe)
        renderer.start()
        
        chunk1 = pe.render(0, 1000)
        chunk2 = pe.render(1000, 1000)
        
        last_val = chunk1.data[-1, 0]
        first_val = chunk2.data[0, 0]
        
        # Walk step should be bounded by slew
        assert abs(last_val - first_val) <= 1.0 * 0.001 * 1.01
        renderer.stop()


class TestRandomPEEdgeCases:
    """Test edge cases and special scenarios."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_very_high_rate(self, renderer):
        """Very high rate should still work."""
        pe = RandomPE(rate=22050, mode=RandomMode.SAMPLE_HOLD, seed=42)
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 1000)
        
        assert snippet.data.shape == (1000, 1)
        # Many unique values expected
        unique = np.unique(snippet.data)
        assert len(unique) > 100
        renderer.stop()
    
    def test_very_low_rate(self, renderer):
        """Very low rate should hold value for long time."""
        pe = RandomPE(rate=0.1, mode=RandomMode.SAMPLE_HOLD, seed=42)  # 10 sec period
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 44100)  # 1 second
        
        # Should have very few unique values
        unique = np.unique(snippet.data)
        assert len(unique) <= 3
        renderer.stop()
    
    def test_min_equals_max(self, renderer):
        """When min equals max, output should be constant."""
        pe = RandomPE(rate=10, min_value=0.5, max_value=0.5, seed=42)
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 1000)
        
        np.testing.assert_allclose(snippet.data, 0.5, rtol=1e-6)
        renderer.stop()
    
    def test_negative_range(self, renderer):
        """Should work with negative value ranges."""
        pe = RandomPE(rate=10, min_value=-10, max_value=-5, seed=42)
        renderer.set_source(pe)
        renderer.start()
        
        snippet = pe.render(0, 10000)
        
        assert np.all(snippet.data >= -10)
        assert np.all(snippet.data <= -5)
        renderer.stop()
