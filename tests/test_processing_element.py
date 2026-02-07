"""
Tests for ProcessingElement and SourcePE classes.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import ProcessingElement, SourcePE, Extent, Snippet


# Concrete test implementations

class ConstantPE(SourcePE):
    """A simple source that outputs a constant value."""
    
    def __init__(self, value: float, duration: int, channels: int = 1):
        self._value = value
        self._duration = duration
        self._channels = channels
    
    def _render(self, start: int, duration: int) -> Snippet:
        my_extent = self.extent()
        data = np.zeros((duration, self._channels))
        
        # Fill in the portion that overlaps with our extent
        for i in range(duration):
            sample_idx = start + i
            if my_extent.contains(sample_idx):
                data[i, :] = self._value
        
        return Snippet(start, data)
    
    def extent(self) -> Extent:
        return Extent(0, self._duration)
    
    def channel_count(self) -> int:
        return self._channels


class GainPE(ProcessingElement):
    """A simple processor that applies gain to input."""
    
    def __init__(self, source: ProcessingElement, gain: float):
        self._source = source
        self._gain = gain
    
    def _render(self, start: int, duration: int) -> Snippet:
        snippet = self._source.render(start, duration)
        return Snippet(start, snippet.data * self._gain)
    
    def extent(self) -> Extent:
        return self._source.extent()
    
    def inputs(self) -> list[ProcessingElement]:
        return [self._source]


class StatefulPE(ProcessingElement):
    """A processor with internal state (non-pure)."""
    
    def __init__(self, source: ProcessingElement):
        self._source = source
        self._call_count = 0
    
    def _render(self, start: int, duration: int) -> Snippet:
        self._call_count += 1
        return self._source.render(start, duration)
    
    def extent(self) -> Extent:
        return self._source.extent()
    
    def inputs(self) -> list[ProcessingElement]:
        return [self._source]
    
    def is_pure(self) -> bool:
        return False  # Has state


class MixPE(ProcessingElement):
    """A processor that mixes multiple inputs."""
    
    def __init__(self, sources: list[ProcessingElement]):
        self._sources = sources
    
    def _render(self, start: int, duration: int) -> Snippet:
        if not self._sources:
            return Snippet.from_zeros(start, duration, 1)

        # Render all inputs (contiguous request per source)
        snippets = [source.render(start, duration) for source in self._sources]
        channels = snippets[0].channels
        result = snippets[0].data.copy()
        for snippet in snippets[1:]:
            result += snippet.data
        return Snippet(start, result)
    
    def extent(self) -> Extent:
        if not self._sources:
            return Extent(0, 0)
        result = self._sources[0].extent()
        for source in self._sources[1:]:
            result = result.union(source.extent())
        return result
    
    def inputs(self) -> list[ProcessingElement]:
        return self._sources


class TestSourcePE:
    """Test SourcePE base class."""
    
    def test_source_has_no_inputs(self):
        """Test that sources have empty inputs list."""
        source = ConstantPE(1.0, 100)
        assert source.inputs() == []
    
    def test_source_is_pure_by_default(self):
        """Test that sources are pure by default (arbitrary render times, multi-sink OK)."""
        source = ConstantPE(1.0, 100)
        assert source.is_pure() is True

    def test_impure_pe_requires_contiguous_requests(self):
        """Test that impure PEs raise when given non-contiguous render requests."""
        source = ConstantPE(1.0, 100)
        stateful = StatefulPE(source)  # StatefulPE is impure
        stateful.render(0, 10)  # First request: ok
        with pytest.raises(ValueError, match="contiguous"):
            stateful.render(20, 10)  # Non-contiguous: expected start=10, got start=20

    def test_source_must_declare_channels(self):
        """Test that source channel_count returns int."""
        source = ConstantPE(1.0, 100, channels=2)
        assert source.channel_count() == 2
    
    def test_source_render(self):
        """Test source render returns correct snippet."""
        source = ConstantPE(0.5, 100, channels=1)
        snippet = source.render(0, 50)
        assert snippet.start == 0
        assert snippet.duration == 50
        assert snippet.channels == 1
        assert np.allclose(snippet.data, 0.5)
    
    def test_source_render_outside_extent(self):
        """Test that rendering outside extent returns zeros."""
        source = ConstantPE(1.0, 100)
        snippet = source.render(100, 50)  # Beyond extent
        assert np.all(snippet.data == 0)
    
    def test_source_render_partial_overlap(self):
        """Test rendering with partial overlap."""
        source = ConstantPE(1.0, 100)
        snippet = source.render(50, 100)  # 50-99 has data, 100-149 is zeros
        
        # First 50 samples should be 1.0
        assert np.allclose(snippet.data[:50], 1.0)
        # Last 50 samples should be 0.0
        assert np.all(snippet.data[50:] == 0)


class TestProcessingElement:
    """Test ProcessingElement base class."""
    
    def test_processor_has_inputs(self):
        """Test that processors have inputs."""
        source = ConstantPE(1.0, 100)
        gain = GainPE(source, 0.5)
        assert gain.inputs() == [source]
    
    def test_processor_default_not_pure(self):
        """Test that processors are not pure by default."""
        source = ConstantPE(1.0, 100)
        gain = GainPE(source, 0.5)
        # GainPE doesn't override is_pure, so it uses default False
        assert gain.is_pure() is False
    
    def test_processor_passthrough_channels(self):
        """Test that processors pass through channels by default."""
        source = ConstantPE(1.0, 100, channels=2)
        gain = GainPE(source, 0.5)
        # channel_count() returns None (pass-through)
        assert gain.channel_count() is None
    
    def test_processor_render(self):
        """Test processor render applies transformation."""
        source = ConstantPE(1.0, 100)
        gain = GainPE(source, 0.5)
        snippet = gain.render(0, 50)
        assert np.allclose(snippet.data, 0.5)
    
    def test_processor_chain(self):
        """Test chaining multiple processors."""
        source = ConstantPE(1.0, 100)
        gain1 = GainPE(source, 0.5)
        gain2 = GainPE(gain1, 0.5)
        snippet = gain2.render(0, 50)
        assert np.allclose(snippet.data, 0.25)

    def test_render_negative_duration_raises(self):
        """Regression: negative durations should raise ValueError."""
        source = ConstantPE(1.0, 100)
        with pytest.raises(ValueError):
            source.render(0, -1)

    def test_scalar_or_pe_values_scalar_and_pe(self):
        """_scalar_or_pe_values() returns 1D by default and supports multi-channel opt-in."""

        class HarnessPE(ProcessingElement):
            def _render(self, start: int, duration: int) -> Snippet:
                return Snippet.from_zeros(start, duration, 1)
            def inputs(self) -> list[ProcessingElement]:
                return []

        h = HarnessPE()

        # Scalar -> 1D
        v = h._scalar_or_pe_values(2.5, 0, 4)
        assert v.shape == (4,)
        assert np.allclose(v, 2.5)

        # PE mono -> 1D (channel 0)
        pe_mono = ConstantPE(7.0, 10, channels=1)
        v2 = h._scalar_or_pe_values(pe_mono, 0, 4)
        assert v2.shape == (4,)
        assert np.allclose(v2, 7.0)

        # Scalar -> 2D if allow_multichannel
        v3 = h._scalar_or_pe_values(1.0, 0, 3, allow_multichannel=True, channels=2)
        assert v3.shape == (3, 2)
        assert np.allclose(v3, 1.0)

        # PE stereo -> 2D if allow_multichannel
        pe_stereo = ConstantPE(0.25, 10, channels=2)
        v4 = h._scalar_or_pe_values(pe_stereo, 0, 3, allow_multichannel=True)
        assert v4.shape == (3, 2)
        assert np.allclose(v4, 0.25)


class TestMixPE:
    """Test MixPE multi-input processor."""
    
    def test_mix_two_sources(self):
        """Test mixing two sources."""
        source1 = ConstantPE(0.3, 100)
        source2 = ConstantPE(0.2, 100)
        mix = MixPE([source1, source2])
        snippet = mix.render(0, 50)
        assert np.allclose(snippet.data, 0.5)
    
    def test_mix_extent_union(self):
        """Test that mix extent is union of inputs."""
        source1 = ConstantPE(1.0, 100)  # Extent [0, 100)
        source2 = ConstantPE(1.0, 200)  # Extent [0, 200)
        mix = MixPE([source1, source2])
        assert mix.extent() == Extent(0, 200)
    
    def test_mix_inputs(self):
        """Test mix returns all inputs."""
        source1 = ConstantPE(1.0, 100)
        source2 = ConstantPE(1.0, 100)
        mix = MixPE([source1, source2])
        assert mix.inputs() == [source1, source2]


class TestResolveOutputChannels:
    """Test resolve_channel_count behavior."""
    
    def test_default_resolution(self):
        """Test default channel resolution uses first input."""
        source = ConstantPE(1.0, 100, channels=2)
        gain = GainPE(source, 0.5)
        # Default resolution returns first input's channels
        assert gain.resolve_channel_count([2]) == 2
        assert gain.resolve_channel_count([2, 1, 4]) == 2


class TestSampleRate:
    """Test global sample rate behavior."""

    def test_sample_rate_available_on_construction(self):
        """Sample rate is set at construction time from global value."""
        source = ConstantPE(1.0, 100)
        assert source.sample_rate == 44100

    def test_missing_sample_rate_raises_on_construction(self):
        """Constructing a PE without a global sample rate is an error."""
        import pygmu2.config as cfg
        prev = cfg.get_sample_rate()
        cfg._SAMPLE_RATE = None  # test-only: clear global sample rate
        try:
            with pytest.raises(RuntimeError, match="Global sample_rate is required"):
                ConstantPE(1.0, 100)
        finally:
            cfg._SAMPLE_RATE = prev
