"""
Tests for LoopPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    LoopPE,
    ConstantPE,
    PiecewisePE,
    DiracPE,
    NullRenderer,
)


class TestLoopPEBasics:
    """Test basic LoopPE creation and properties."""
    
    def test_create_default(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        loop = LoopPE(source)
        
        assert loop.source is source
        assert loop.loop_start is None
        assert loop.loop_end is None
        assert loop.count is None
        assert loop.crossfade_seconds == 0.0
    
    def test_create_with_params(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        loop = LoopPE(
            source,
            loop_start=10,
            loop_end=50,
            count=4,
            crossfade_seconds=0.01,
        )
        
        assert loop.loop_start == 10
        assert loop.loop_end == 50
        assert loop.count == 4
        assert loop.crossfade_seconds == 0.01
    
    def test_negative_crossfade_clamped(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        with pytest.raises(ValueError):
            LoopPE(source, crossfade_seconds=-0.1)
    
    def test_inputs(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        loop = LoopPE(source)
        
        assert loop.inputs() == [source]
    
    def test_is_pure(self):
        """LoopPE is stateless, so is_pure() should return True."""
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        loop = LoopPE(source)
        
        assert loop.is_pure() is True
    
    def test_channel_count_passthrough(self):
        source = ConstantPE(1.0, channels=2)
        loop = LoopPE(source, loop_end=100)
        
        assert loop.channel_count() == 2
    
    def test_repr(self):
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        loop = LoopPE(source, loop_start=10, loop_end=50, count=4)
        
        repr_str = repr(loop)
        assert "LoopPE" in repr_str
        assert "PiecewisePE" in repr_str
        assert "10" in repr_str
        assert "50" in repr_str
        assert "count=4" in repr_str


class TestLoopPEExtent:
    """Test extent calculation."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_infinite_loop_extent(self):
        """Infinite loop should have infinite extent."""
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        loop = LoopPE(source)
        
        self.renderer.set_source(loop)
        
        extent = loop.extent()
        assert extent.start == 0
        assert extent.end is None
    
    def test_finite_loop_extent(self):
        """Finite loop should have bounded extent."""
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        loop = LoopPE(source, count=3)
        
        self.renderer.set_source(loop)
        
        extent = loop.extent()
        assert extent.start == 0
        assert extent.end == 300  # 3 * 100
    
    def test_custom_region_extent(self):
        """Loop with custom region should calculate extent correctly."""
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        loop = LoopPE(source, loop_start=20, loop_end=60, count=5)
        
        self.renderer.set_source(loop)
        
        extent = loop.extent()
        assert extent.start == 0
        assert extent.end == 200  # 5 * (60 - 20)


class TestLoopPEBasicLooping:
    """Test basic looping behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_single_loop_matches_source(self):
        """First iteration should match source."""
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        loop = LoopPE(source)
        
        self.renderer.set_source(loop)
        
        with self.renderer:
            self.renderer.start()
            
            source_snippet = source.render(0, 100)
            loop_snippet = loop.render(0, 100)
            
            np.testing.assert_array_almost_equal(
                source_snippet.data,
                loop_snippet.data,
                decimal=5
            )
    
    def test_second_iteration_repeats(self):
        """Second iteration should repeat first."""
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        loop = LoopPE(source)
        
        self.renderer.set_source(loop)
        
        with self.renderer:
            self.renderer.start()
            
            first = loop.render(0, 100)
            second = loop.render(100, 100)
            
            np.testing.assert_array_almost_equal(
                first.data,
                second.data,
                decimal=5
            )
    
    def test_loop_wraps_correctly(self):
        """Values should wrap at loop boundary."""
        # Ramp [0,5): values 0, 0.8, 1.6, 2.4, 3.2 at samples 0-4
        source = PiecewisePE([(0, 0.0), (5, 4.0)])
        loop = LoopPE(source)
        
        self.renderer.set_source(loop)
        
        with self.renderer:
            self.renderer.start()
            
            # Render across loop boundary: positions 3,4,0,1 → source 2.4, 3.2, 0, 0.8
            snippet = loop.render(3, 4)
            
            assert snippet.data[0, 0] == pytest.approx(2.4, abs=0.1)
            assert snippet.data[1, 0] == pytest.approx(3.2, abs=0.1)
            assert snippet.data[2, 0] == pytest.approx(0.0, abs=0.1)
            assert snippet.data[3, 0] == pytest.approx(0.8, abs=0.1)


class TestLoopPECustomRegion:
    """Test looping with custom start/end."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_custom_loop_region(self):
        """Should loop only the specified region."""
        # Ramp [0,10): at sample i value = 0.9*i → 0, 0.9, 1.8, ..., 8.1
        source = PiecewisePE([(0, 0.0), (10, 9.0)])
        # Loop only samples 2-5 (indices 2,3,4 → values 1.8, 2.7, 3.6)
        loop = LoopPE(source, loop_start=2, loop_end=5)
        
        self.renderer.set_source(loop)
        
        with self.renderer:
            self.renderer.start()
            
            snippet = loop.render(0, 9)  # 3 iterations
            
            # Each iteration: 1.8, 2.7, 3.6
            assert snippet.data[0, 0] == pytest.approx(1.8, abs=0.1)
            assert snippet.data[1, 0] == pytest.approx(2.7, abs=0.1)
            assert snippet.data[2, 0] == pytest.approx(3.6, abs=0.1)
            
            assert snippet.data[3, 0] == pytest.approx(1.8, abs=0.1)
            assert snippet.data[4, 0] == pytest.approx(2.7, abs=0.1)
            assert snippet.data[5, 0] == pytest.approx(3.6, abs=0.1)
            
            assert snippet.data[6, 0] == pytest.approx(1.8, abs=0.1)
            assert snippet.data[7, 0] == pytest.approx(2.7, abs=0.1)
            assert snippet.data[8, 0] == pytest.approx(3.6, abs=0.1)


class TestLoopPEFiniteCount:
    """Test finite loop count."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_stops_after_count(self):
        """Should output silence after count iterations."""
        source = ConstantPE(1.0)
        source_cropped = PiecewisePE([(0, 1.0), (10, 1.0)])  # 10 samples of 1.0
        loop = LoopPE(source_cropped, count=2)
        
        self.renderer.set_source(loop)
        
        with self.renderer:
            self.renderer.start()
            
            # Within loop range
            snippet1 = loop.render(0, 10)
            assert np.all(snippet1.data == pytest.approx(1.0, abs=0.01))
            
            snippet2 = loop.render(10, 10)
            assert np.all(snippet2.data == pytest.approx(1.0, abs=0.01))
            
            # Past loop range - should be silence
            snippet3 = loop.render(20, 10)
            assert np.all(snippet3.data == 0.0)
    
    def test_partial_final_render(self):
        """Render spanning end of loop should be partial."""
        source = PiecewisePE([(0, 1.0), (10, 1.0)])
        loop = LoopPE(source, count=2)  # Total 20 samples
        
        self.renderer.set_source(loop)
        
        with self.renderer:
            self.renderer.start()
            
            # Render from 15 to 25 (5 valid, 5 silence)
            snippet = loop.render(15, 10)
            
            # First 5 should be non-zero
            assert np.all(snippet.data[:5, 0] != 0.0)
            # Last 5 should be zero
            assert np.all(snippet.data[5:, 0] == 0.0)


class TestLoopPECrossfade:
    """Test crossfade behavior."""
    
    def setup_method(self):
        import pygmu2 as pg
        pg.set_sample_rate(1000)
        self.renderer = NullRenderer(sample_rate=1000)  # 1kHz for easy math
    
    def test_crossfade_smooths_transition(self):
        """Crossfade should smooth the loop boundary."""
        # Create a step function: 0 for first half, 1 for second half
        # Without crossfade, loop point would have a hard edge
        source = PiecewisePE([(0, 0.0), (50, 0.0)])  # First 50 samples = 0
        # Actually let's use a simpler test - constant values
        
        # Create source with different start and end values
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        
        # With crossfade
        loop_xfade = LoopPE(source, crossfade_seconds=0.02)  # 20 samples at 1kHz
        
        self.renderer.set_source(loop_xfade)
        
        with self.renderer:
            self.renderer.start()
            
            # Render around the loop point
            snippet = loop_xfade.render(80, 40)  # 80-119, spans loop at 100
            
            # The crossfade region should have intermediate values
            # At sample 90 (10 into xfade), should be blending
            # At sample 99 (last before wrap), should be mostly from start
            
            # Values should transition smoothly, not jump
            diffs = np.abs(np.diff(snippet.data[:, 0]))
            max_diff = np.max(diffs)
            
            # Maximum difference between adjacent samples should be smaller
            # than the discontinuity without crossfade (which is ~1.0)
            assert max_diff < 0.3  # Smooth transition, not a hard jump
    
    def test_no_crossfade_has_discontinuity(self):
        """Without crossfade, there should be a discontinuity."""
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        loop_no_xfade = LoopPE(source, crossfade_seconds=0.0)
        
        self.renderer.set_source(loop_no_xfade)
        
        with self.renderer:
            self.renderer.start()
            
            # Render exactly at the loop point
            snippet = loop_no_xfade.render(99, 2)  # Sample 99 and 100
            
            # Sample 99 should be near 1.0, sample 100 (which wraps) near 0.0
            assert snippet.data[0, 0] > 0.9
            assert snippet.data[1, 0] < 0.1


class TestLoopPEStereo:
    """Test stereo signal handling."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_stereo_looping(self):
        """Stereo source should loop both channels."""
        source = ConstantPE(0.5, channels=2)
        # Need a finite source for looping
        from pygmu2 import CropPE, Extent
        cropped = CropPE(source, Extent(0, 100))
        loop = LoopPE(cropped)
        
        self.renderer.set_source(loop)
        
        with self.renderer:
            self.renderer.start()
            
            snippet = loop.render(0, 200)
            
            assert snippet.channels == 2
            np.testing.assert_array_almost_equal(
                snippet.data[:, 0],
                snippet.data[:, 1],
                decimal=5
            )


class TestLoopPEErrors:
    """Test error handling."""
    
    def setup_method(self):
        import pygmu2 as pg
        pg.set_sample_rate(44100)
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_invalid_loop_length(self):
        """Should raise error if loop_end <= loop_start."""
        source = PiecewisePE([(0, 0.0), (100, 1.0)])
        with pytest.raises(ValueError, match="positive"):
            LoopPE(source, loop_start=50, loop_end=50)
    
    def test_infinite_source_without_end(self):
        """Should raise error if source is infinite without explicit loop_end."""
        source = ConstantPE(1.0)  # Infinite extent
        with pytest.raises(ValueError, match="infinite"):
            LoopPE(source)
    
    def test_infinite_source_with_explicit_end_ok(self):
        """Infinite source should work with explicit loop_end."""
        source = ConstantPE(1.0)
        loop = LoopPE(source, loop_start=0, loop_end=100)
        
        # Should not raise
        self.renderer.set_source(loop)
        
        with self.renderer:
            self.renderer.start()
            snippet = loop.render(0, 50)
            assert np.all(snippet.data == pytest.approx(1.0, abs=0.01))
