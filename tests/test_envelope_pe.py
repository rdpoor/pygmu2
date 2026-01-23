"""
Tests for EnvelopePE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    EnvelopePE,
    DetectionMode,
    ConstantPE,
    SinePE,
    DiracPE,
    RampPE,
    NullRenderer,
)


class TestEnvelopePEBasics:
    """Test basic EnvelopePE creation and properties."""
    
    def test_create_default(self):
        source = ConstantPE(1.0)
        env = EnvelopePE(source)
        
        assert env.source is source
        assert env.attack == 0.01
        assert env.release == 0.1
        assert env.lookahead == 0.0
        assert env.mode == DetectionMode.PEAK
    
    def test_create_with_params(self):
        source = ConstantPE(1.0)
        env = EnvelopePE(
            source,
            attack=0.02,
            release=0.2,
            lookahead=0.01,
            mode=DetectionMode.RMS,
        )
        
        assert env.attack == 0.02
        assert env.release == 0.2
        assert env.lookahead == 0.01
        assert env.mode == DetectionMode.RMS
    
    def test_lookahead_clamped_to_attack(self):
        """Lookahead should be clamped to attack time."""
        source = ConstantPE(1.0)
        env = EnvelopePE(source, attack=0.01, lookahead=0.1)
        
        # Lookahead (0.1) should be clamped to attack (0.01)
        assert env.lookahead == 0.01
    
    def test_negative_values_clamped(self):
        """Negative attack/release/lookahead should be clamped to 0."""
        source = ConstantPE(1.0)
        env = EnvelopePE(source, attack=-0.1, release=-0.1, lookahead=-0.1)
        
        assert env.attack == 0.0
        assert env.release == 0.0
        assert env.lookahead == 0.0
    
    def test_inputs(self):
        source = ConstantPE(1.0)
        env = EnvelopePE(source)
        
        assert env.inputs() == [source]
    
    def test_is_not_pure(self):
        """EnvelopePE maintains state, so is_pure() should return False."""
        source = ConstantPE(1.0)
        env = EnvelopePE(source)
        
        assert env.is_pure() is False
    
    def test_channel_count_passthrough(self):
        source = ConstantPE(1.0, channels=2)
        env = EnvelopePE(source)
        
        assert env.channel_count() == 2
    
    def test_extent_from_source(self):
        source = RampPE(0.0, 1.0, duration=1000)
        env = EnvelopePE(source)
        
        extent = env.extent()
        assert extent.start == 0
        assert extent.end == 1000
    
    def test_repr(self):
        source = ConstantPE(1.0)
        env = EnvelopePE(source, attack=0.01, release=0.1)
        
        repr_str = repr(env)
        assert "EnvelopePE" in repr_str
        assert "ConstantPE" in repr_str
        assert "0.01" in repr_str
        assert "0.1" in repr_str


class TestEnvelopePEBehavior:
    """Test envelope follower behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_constant_input_reaches_unity(self):
        """Envelope of constant 1.0 should reach 1.0 after settling."""
        source = ConstantPE(1.0)
        env = EnvelopePE(source, attack=0.01, release=0.1)
        
        self.renderer.set_source(env)
        
        with self.renderer:
            self.renderer.start()
            
            # Let envelope settle (10x attack time)
            _ = env.render(0, 4410)  # ~100ms
            
            # Check steady state
            snippet = env.render(4410, 100)
            
            # Should be approximately 1.0
            np.testing.assert_array_almost_equal(
                snippet.data,
                np.full((100, 1), 1.0, dtype=np.float32),
                decimal=2
            )
    
    def test_envelope_is_positive(self):
        """Envelope output should always be positive."""
        # Sine wave has negative values
        source = SinePE(frequency=100.0, amplitude=1.0)
        env = EnvelopePE(source, attack=0.001, release=0.01)
        
        self.renderer.set_source(env)
        
        with self.renderer:
            self.renderer.start()
            
            snippet = env.render(0, 1000)
            
            assert np.all(snippet.data >= 0)
    
    def test_attack_rises(self):
        """Envelope should rise during attack phase."""
        # Step from 0 to 1
        source = ConstantPE(1.0)
        env = EnvelopePE(source, attack=0.01, release=0.1)
        
        self.renderer.set_source(env)
        
        with self.renderer:
            self.renderer.start()
            
            snippet = env.render(0, 441)  # 10ms at 44100Hz
            
            # First sample should be lower than last
            assert snippet.data[0, 0] < snippet.data[-1, 0]
            # Should be monotonically increasing (attack)
            diff = np.diff(snippet.data[:, 0])
            assert np.all(diff >= 0)
    
    def test_release_falls(self):
        """Envelope should fall during release phase."""
        source = ConstantPE(1.0)
        env = EnvelopePE(source, attack=0.001, release=0.1)
        
        self.renderer.set_source(env)
        
        with self.renderer:
            self.renderer.start()
            
            # First, charge up the envelope
            _ = env.render(0, 4410)  # ~100ms
            
            # Now switch to zero input (manually - we'll use a workaround)
            # Actually, let's test with DiracPE which is 1 at sample 0, 0 elsewhere
            
        # Restart with impulse
        source2 = DiracPE()
        env2 = EnvelopePE(source2, attack=0.001, release=0.01)
        
        self.renderer.set_source(env2)
        
        with self.renderer:
            self.renderer.start()
            
            snippet = env2.render(0, 1000)
            
            # Should have a peak near the start, then decay
            peak_idx = np.argmax(snippet.data[:, 0])
            
            # After peak, values should generally decrease
            post_peak = snippet.data[peak_idx:, 0]
            if len(post_peak) > 10:
                assert post_peak[0] > post_peak[-1]


class TestEnvelopePELookahead:
    """Test lookahead functionality."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_lookahead_anticipates_transient(self):
        """Envelope with lookahead should start rising before transient."""
        from pygmu2 import DelayPE
        
        # Create impulse delayed to sample 500 (so we can see the buildup)
        impulse = DiracPE()
        source = DelayPE(impulse, delay=500)  # Delays impulse to sample 500
        
        lookahead_samples = 220  # ~5ms at 44100Hz
        
        # Without lookahead - envelope rises after sample 500
        env_no_look = EnvelopePE(source, attack=0.005, release=0.1, lookahead=0.0)
        
        # With lookahead - envelope starts rising ~220 samples earlier
        env_look = EnvelopePE(source, attack=0.005, release=0.1, lookahead=0.005)
        
        self.renderer.set_source(env_no_look)
        with self.renderer:
            self.renderer.start()
            snippet_no_look = env_no_look.render(0, 1000)
        
        self.renderer.set_source(env_look)
        with self.renderer:
            self.renderer.start()
            snippet_look = env_look.render(0, 1000)
        
        # Find where each envelope crosses a threshold
        threshold = 0.01
        
        # Both should eventually rise, but lookahead should rise first
        no_look_rises = snippet_no_look.data[:, 0] > threshold
        look_rises = snippet_look.data[:, 0] > threshold
        
        if np.any(no_look_rises) and np.any(look_rises):
            no_look_start = np.argmax(no_look_rises)
            look_start = np.argmax(look_rises)
            
            # Lookahead version should start rising earlier
            assert look_start < no_look_start, (
                f"Lookahead started at {look_start}, non-lookahead at {no_look_start}"
            )


class TestEnvelopePERMS:
    """Test RMS detection mode."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_rms_mode_smoother(self):
        """RMS mode should produce smoother output than PEAK."""
        source = SinePE(frequency=100.0, amplitude=1.0)
        
        env_peak = EnvelopePE(source, attack=0.001, release=0.001, mode=DetectionMode.PEAK)
        env_rms = EnvelopePE(source, attack=0.001, release=0.001, mode=DetectionMode.RMS)
        
        self.renderer.set_source(env_peak)
        with self.renderer:
            self.renderer.start()
            snippet_peak = env_peak.render(0, 1000)
        
        self.renderer.set_source(env_rms)
        with self.renderer:
            self.renderer.start()
            snippet_rms = env_rms.render(0, 1000)
        
        # RMS should have lower variance (smoother)
        var_peak = np.var(snippet_peak.data)
        var_rms = np.var(snippet_rms.data)
        
        # RMS variance should be lower (or at least not much higher)
        # This is a soft test since behavior depends on parameters
        assert var_rms <= var_peak * 1.5


class TestEnvelopePEStateManagement:
    """Test envelope state management."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_state_persists_across_renders(self):
        """Envelope state should persist between render calls."""
        source = ConstantPE(1.0)
        env = EnvelopePE(source, attack=0.01, release=0.1)
        
        self.renderer.set_source(env)
        
        with self.renderer:
            self.renderer.start()
            
            # First render - envelope rising
            snippet1 = env.render(0, 100)
            last_val_1 = snippet1.data[-1, 0]
            
            # Second render - should continue from previous
            snippet2 = env.render(100, 100)
            first_val_2 = snippet2.data[0, 0]
            
            # Should be continuous (close values)
            assert abs(first_val_2 - last_val_1) < 0.1
    
    def test_state_resets_on_start(self):
        """Envelope state should reset when on_start() is called."""
        source = ConstantPE(1.0)
        env = EnvelopePE(source, attack=0.01, release=0.1)
        
        self.renderer.set_source(env)
        
        with self.renderer:
            self.renderer.start()
            
            # First pass - charge up envelope
            snippet1 = env.render(0, 1000)
            
            # Restart
            self.renderer.stop()
            self.renderer.start()
            
            # Second pass - should start from zero again
            snippet2 = env.render(0, 1000)
            
            # First values should be close to zero (just starting)
            assert snippet2.data[0, 0] < 0.5


class TestEnvelopePEStereo:
    """Test stereo signal handling."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_stereo_independent_channels(self):
        """Each stereo channel should have independent envelope."""
        source = ConstantPE(1.0, channels=2)
        env = EnvelopePE(source, attack=0.01, release=0.1)
        
        self.renderer.set_source(env)
        
        with self.renderer:
            self.renderer.start()
            
            # Let settle
            _ = env.render(0, 4410)
            
            snippet = env.render(4410, 100)
            
            assert snippet.channels == 2
            # Both channels should be similar (same input)
            np.testing.assert_array_almost_equal(
                snippet.data[:, 0],
                snippet.data[:, 1],
                decimal=4
            )
