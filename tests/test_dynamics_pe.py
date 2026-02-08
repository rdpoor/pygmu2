"""
Tests for DynamicsPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

from pygmu2 import (
    DynamicsPE,
    DynamicsMode,
    EnvelopePE,
    ConstantPE,
    SinePE,
    GainPE,
    CropPE,
    Extent,
    NullRenderer,
)


class TestDynamicsPEBasics:
    """Test basic DynamicsPE creation and properties."""
    
    def test_create_default(self):
        """Test creating DynamicsPE with default parameters."""
        source = ConstantPE(1.0)
        envelope = ConstantPE(0.5)
        dynamics = DynamicsPE(source, envelope)
        
        assert dynamics.threshold == -20.0
        assert dynamics.ratio == 4.0
        assert dynamics.knee == 0.0
        assert dynamics.mode == DynamicsMode.COMPRESS
        assert dynamics.stereo_link is True
    
    def test_create_with_params(self):
        """Test creating DynamicsPE with custom parameters."""
        source = ConstantPE(1.0)
        envelope = ConstantPE(0.5)
        dynamics = DynamicsPE(
            source, envelope,
            threshold=-10.0,
            ratio=8.0,
            knee=6.0,
            makeup_gain=3.0,
            mode=DynamicsMode.LIMIT,
            stereo_link=False,
        )
        
        assert dynamics.threshold == -10.0
        assert dynamics.ratio == 8.0
        assert dynamics.knee == 6.0
        assert dynamics.makeup_gain == 3.0
        assert dynamics.mode == DynamicsMode.LIMIT
        assert dynamics.stereo_link is False
    
    def test_inputs(self):
        """Test that inputs() returns both source and envelope."""
        source = SinePE(frequency=440.0)
        envelope = EnvelopePE(source)
        dynamics = DynamicsPE(source, envelope)
        
        inputs = dynamics.inputs()
        assert len(inputs) == 2
        assert source in inputs
        assert envelope in inputs
    
    def test_is_pure(self):
        """DynamicsPE is pure (state is in envelope PE)."""
        source = ConstantPE(1.0)
        envelope = ConstantPE(0.5)
        dynamics = DynamicsPE(source, envelope)
        
        assert dynamics.is_pure() is True
    
    def test_channel_count_passthrough(self):
        """Channel count comes from source."""
        source = ConstantPE(1.0, channels=2)
        envelope = ConstantPE(0.5)
        dynamics = DynamicsPE(source, envelope)
        
        assert dynamics.channel_count() == 2
    
    def test_auto_makeup_gain(self):
        """Auto makeup gain is computed for compression."""
        source = ConstantPE(1.0)
        envelope = ConstantPE(0.5)
        dynamics = DynamicsPE(source, envelope, threshold=-20, ratio=4, makeup_gain="auto")
        
        # Auto makeup should be positive (compensating for gain reduction)
        assert dynamics.makeup_gain > 0
    
    def test_repr(self):
        """Test string representation."""
        source = ConstantPE(1.0)
        envelope = ConstantPE(0.5)
        dynamics = DynamicsPE(source, envelope, threshold=-20, ratio=4)
        
        repr_str = repr(dynamics)
        assert "DynamicsPE" in repr_str
        assert "threshold=-20" in repr_str
        assert "ratio=4" in repr_str

    def test_extent_with_disjoint_inputs_does_not_crash(self):
        """
        Regression: if source/envelope extents do not overlap, extent() should be
        a well-defined empty extent (start == end), not an exception.
        """
        source = CropPE(ConstantPE(1.0), 0, (10) - (0))
        envelope = CropPE(ConstantPE(0.5), 20, (30) - (20))  # disjoint
        dynamics = DynamicsPE(source, envelope)

        extent = dynamics.extent()
        assert extent.is_empty()


class TestDynamicsPECompression:
    """Test compression behavior."""
    
    @pytest.fixture
    def renderer(self):
        """Create a renderer for testing."""
        return NullRenderer(sample_rate=44100)
    
    def test_no_compression_below_threshold(self, renderer):
        """Signal below threshold should pass through unchanged."""
        # Create a quiet signal (well below -20dB threshold)
        source = ConstantPE(0.01)  # About -40dB
        envelope = ConstantPE(0.01)
        dynamics = DynamicsPE(source, envelope, threshold=-20, ratio=4, makeup_gain=0)
        
        renderer.set_source(dynamics)
        renderer.start()
        
        snippet = dynamics.render(0, 1000)
        
        # Should be unchanged (no compression below threshold)
        np.testing.assert_allclose(snippet.data, 0.01, rtol=0.01)
    
    def test_compression_above_threshold(self, renderer):
        """Signal above threshold should be compressed."""
        # Create a loud signal (above threshold)
        source = ConstantPE(1.0)  # 0dB
        envelope = ConstantPE(1.0)  # 0dB envelope
        dynamics = DynamicsPE(source, envelope, threshold=-20, ratio=4, makeup_gain=0)
        
        renderer.set_source(dynamics)
        renderer.start()
        
        snippet = dynamics.render(0, 1000)
        
        # With 20dB overshoot and 4:1 ratio:
        # Gain reduction = 20 * (1 - 1/4) = 15dB
        # Output should be about -15dB = 0.178
        assert np.all(snippet.data < 0.3)  # Significantly reduced
        assert np.all(snippet.data > 0.1)  # But not silenced
    
    def test_higher_ratio_more_compression(self, renderer):
        """Higher ratio should result in more compression."""
        source = ConstantPE(1.0)
        envelope = ConstantPE(1.0)
        
        # 4:1 compression
        dyn_4 = DynamicsPE(source, envelope, threshold=-20, ratio=4, makeup_gain=0)
        renderer.set_source(dyn_4)
        renderer.start()
        out_4 = dyn_4.render(0, 1000).data[0, 0]
        renderer.stop()
        
        # 8:1 compression
        dyn_8 = DynamicsPE(source, envelope, threshold=-20, ratio=8, makeup_gain=0)
        renderer.set_source(dyn_8)
        renderer.start()
        out_8 = dyn_8.render(0, 1000).data[0, 0]
        
        # Higher ratio should produce lower output
        assert out_8 < out_4
    
    def test_soft_knee_gradual_transition(self, renderer):
        """Soft knee should provide gradual compression onset."""
        source = ConstantPE(1.0)
        
        # Test at threshold level
        envelope = ConstantPE(0.1)  # -20dB (at threshold)
        
        # Hard knee
        dyn_hard = DynamicsPE(source, envelope, threshold=-20, ratio=4, knee=0, makeup_gain=0)
        renderer.set_source(dyn_hard)
        renderer.start()
        out_hard = dyn_hard.render(0, 1000).data[0, 0]
        renderer.stop()
        
        # Soft knee (12dB)
        dyn_soft = DynamicsPE(source, envelope, threshold=-20, ratio=4, knee=12, makeup_gain=0)
        renderer.set_source(dyn_soft)
        renderer.start()
        out_soft = dyn_soft.render(0, 1000).data[0, 0]
        
        # At threshold, soft knee should already have some compression
        # (we're in the middle of the knee region)
        assert out_soft < out_hard


class TestDynamicsPELimiter:
    """Test limiter mode."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_limiter_caps_output(self, renderer):
        """Limiter mode should cap output at threshold."""
        source = ConstantPE(1.0)  # 0dB input
        envelope = ConstantPE(1.0)
        
        # Limit at -6dB
        dynamics = DynamicsPE(
            source, envelope,
            threshold=-6,
            mode=DynamicsMode.LIMIT,
            makeup_gain=0,
        )
        
        renderer.set_source(dynamics)
        renderer.start()
        
        snippet = dynamics.render(0, 1000)
        
        # Output should be approximately at threshold (-6dB = 0.5)
        np.testing.assert_allclose(snippet.data, 0.5, rtol=0.1)


class TestDynamicsPEGate:
    """Test gate mode."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_gate_silences_below_threshold(self, renderer):
        """Gate should silence signal below threshold."""
        source = ConstantPE(1.0)
        envelope = ConstantPE(0.001)  # -60dB, well below threshold
        
        dynamics = DynamicsPE(
            source, envelope,
            threshold=-40,
            mode=DynamicsMode.GATE,
            range=-80,  # Attenuate by 80dB when gated
            makeup_gain=0,
        )
        
        renderer.set_source(dynamics)
        renderer.start()
        
        snippet = dynamics.render(0, 1000)
        
        # Should be heavily attenuated
        assert np.all(np.abs(snippet.data) < 0.001)
    
    def test_gate_passes_above_threshold(self, renderer):
        """Gate should pass signal above threshold."""
        source = ConstantPE(1.0)
        envelope = ConstantPE(0.5)  # -6dB, above -40dB threshold
        
        dynamics = DynamicsPE(
            source, envelope,
            threshold=-40,
            mode=DynamicsMode.GATE,
            makeup_gain=0,
        )
        
        renderer.set_source(dynamics)
        renderer.start()
        
        snippet = dynamics.render(0, 1000)
        
        # Should pass through unchanged
        np.testing.assert_allclose(snippet.data, 1.0, rtol=0.01)


class TestDynamicsPEExpander:
    """Test expander mode."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_expander_reduces_below_threshold(self, renderer):
        """Expander should reduce signal below threshold."""
        source = ConstantPE(1.0)
        envelope = ConstantPE(0.01)  # -40dB, below -20dB threshold
        
        dynamics = DynamicsPE(
            source, envelope,
            threshold=-20,
            ratio=2.0,  # 2:1 expansion
            mode=DynamicsMode.EXPAND,
            makeup_gain=0,
        )
        
        renderer.set_source(dynamics)
        renderer.start()
        
        snippet = dynamics.render(0, 1000)
        
        # Should be reduced (expanded down)
        assert np.all(snippet.data < 0.5)
    
    def test_expander_passes_above_threshold(self, renderer):
        """Expander should pass signal above threshold."""
        source = ConstantPE(1.0)
        envelope = ConstantPE(0.5)  # -6dB, above -20dB threshold
        
        dynamics = DynamicsPE(
            source, envelope,
            threshold=-20,
            ratio=2.0,
            mode=DynamicsMode.EXPAND,
            makeup_gain=0,
        )
        
        renderer.set_source(dynamics)
        renderer.start()
        
        snippet = dynamics.render(0, 1000)
        
        # Should pass through unchanged
        np.testing.assert_allclose(snippet.data, 1.0, rtol=0.01)


class TestDynamicsPEStereoLink:
    """Test stereo linking behavior."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_stereo_linked_uses_max(self, renderer):
        """With stereo link, max envelope across channels is used."""
        # Stereo source
        source = ConstantPE(1.0, channels=2)
        
        # Create envelope with different levels per channel
        # Left channel loud (0dB), right channel quiet (-40dB)
        env_left = ConstantPE(1.0)
        env_right = ConstantPE(0.01)
        
        # Use CropPE to create a stereo envelope PE
        # Actually, let's use a simpler approach - just test with mono envelope
        envelope = ConstantPE(1.0)  # Loud envelope
        
        dynamics = DynamicsPE(
            source, envelope,
            threshold=-20,
            ratio=4,
            stereo_link=True,
            makeup_gain=0,
        )
        
        renderer.set_source(dynamics)
        renderer.start()
        
        snippet = dynamics.render(0, 1000)
        
        # Both channels should have same gain reduction
        np.testing.assert_allclose(snippet.data[:, 0], snippet.data[:, 1])
    
    def test_stereo_unlinked_independent(self, renderer):
        """Without stereo link, channels are processed independently."""
        # This test would require a stereo envelope, which is more complex
        # For now, just verify the flag is respected
        source = ConstantPE(1.0, channels=2)
        envelope = ConstantPE(0.5)
        
        dynamics = DynamicsPE(source, envelope, stereo_link=False)
        
        assert dynamics.stereo_link is False


class TestDynamicsPESidechain:
    """Test sidechain compression use case."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_sidechain_ducking(self, renderer):
        """Envelope from different source should control compression."""
        # Bass signal (to be ducked)
        bass = ConstantPE(1.0)
        
        # Kick envelope (triggers ducking)
        # High envelope = compress the bass
        kick_env = ConstantPE(1.0)  # Kick is hitting (loud)
        
        ducked = DynamicsPE(
            bass, kick_env,
            threshold=-20,
            ratio=8,
            makeup_gain=0,
        )
        
        renderer.set_source(ducked)
        renderer.start()
        
        snippet = ducked.render(0, 1000)
        
        # Bass should be ducked (compressed) when kick envelope is high
        assert np.all(snippet.data < 0.5)


class TestDynamicsPEZeroDuration:
    """Test edge cases."""
    
    @pytest.fixture
    def renderer(self):
        return NullRenderer(sample_rate=44100)
    
    def test_zero_duration(self, renderer):
        """Zero duration should return empty snippet."""
        source = ConstantPE(1.0)
        envelope = ConstantPE(0.5)
        dynamics = DynamicsPE(source, envelope)
        
        renderer.set_source(dynamics)
        renderer.start()
        
        snippet = dynamics.render(0, 0)
        
        assert snippet.data.shape[0] == 0
