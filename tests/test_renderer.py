"""
Tests for Renderer class.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import Renderer, NullRenderer, ProcessingElement, SourcePE, Extent, Snippet

# Concrete test implementations


class ConstantPE(SourcePE):
    """A simple source that outputs a constant value."""

    def __init__(self, value: float, duration: int, channels: int = 1):
        self._value = value
        self._duration = duration
        self._channels = channels

    def _render(self, start: int, duration: int) -> Snippet:
        data = np.full((duration, self._channels), self._value)
        return Snippet(start, data)

    def extent(self) -> Extent:
        return Extent(0, self._duration)

    def channel_count(self) -> int:
        return self._channels


class GainPE(ProcessingElement):
    """A simple pass-through processor."""

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
    """A stateful processor (not pure)."""

    def __init__(self, source: ProcessingElement):
        self._source = source

    def _render(self, start: int, duration: int) -> Snippet:
        return self._source.render(start, duration)

    def extent(self) -> Extent:
        return self._source.extent()

    def inputs(self) -> list[ProcessingElement]:
        return [self._source]

    stateful = True


class MixPE(ProcessingElement):
    """A multi-input mixer."""

    def __init__(self, sources: list[ProcessingElement]):
        self._sources = sources

    def _render(self, start: int, duration: int) -> Snippet:
        if not self._sources:
            return Snippet.from_zeros(start, duration, 1)
        result = self._sources[0].render(start, duration).data.copy()
        for source in self._sources[1:]:
            result += source.render(start, duration).data
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


class MockRenderer(Renderer):
    """A concrete Renderer for testing."""

    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self.output_snippets: list[Snippet] = []

    def _output(self, snippet: Snippet) -> None:
        self.output_snippets.append(snippet)


class TestRendererBasics:
    """Test basic Renderer functionality."""

    def test_create_renderer(self):
        """Test creating a renderer."""
        renderer = MockRenderer(sample_rate=48000)
        assert renderer.sample_rate == 48000
        assert renderer.source is None
        assert renderer.started is False

    def test_default_sample_rate(self):
        """Test default sample rate is 44100."""
        renderer = MockRenderer()
        assert renderer.sample_rate == 44100

    def test_set_source(self):
        """Test setting a source."""
        renderer = MockRenderer()
        source = ConstantPE(1.0, 100)
        renderer.set_source(source)
        assert renderer.source is source

    def test_render_without_source_raises(self):
        """Test that render without source raises error."""
        renderer = MockRenderer()
        with pytest.raises(RuntimeError, match="No source set"):
            renderer.render(0, 100)

    def test_render_without_start_raises(self):
        """Test that render without start raises error."""
        renderer = MockRenderer()
        source = ConstantPE(0.5, 100)
        renderer.set_source(source)
        with pytest.raises(RuntimeError, match="Not started"):
            renderer.render(0, 100)

    def test_render_zero_duration_raises(self):
        """Test that render with zero duration raises error to prevent infinite loops."""
        renderer = MockRenderer()
        source = ConstantPE(0.5, 100)
        renderer.set_source(source)
        renderer.start()
        with pytest.raises(ValueError, match="duration >= 1"):
            renderer.render(0, 0)
        renderer.stop()

    def test_render_outputs_snippet(self):
        """Test that render outputs snippet."""
        renderer = MockRenderer()
        source = ConstantPE(0.5, 100)
        renderer.set_source(source)
        renderer.start()
        renderer.render(0, 50)

        assert len(renderer.output_snippets) == 1
        assert renderer.output_snippets[0].duration == 50
        renderer.stop()


class TestGraphAttachment:
    """set_source() attaches the graph without validation (PD-2):
    correctness is proven in CI; anything wrong surfaces at render."""

    def test_simple_graph_renders(self):
        renderer = MockRenderer()
        gain = GainPE(ConstantPE(1.0, 100), 0.5)
        renderer.set_source(gain)
        renderer.start()
        renderer.render(0, 50)
        assert renderer.output_snippets[0].duration == 50
        renderer.stop()

    def test_multi_sink_attaches_without_walking(self):
        """No purity walk: a fan-out graph attaches; correctness of the
        fan-out is the contiguity check's job at render time."""
        renderer = MockRenderer()
        source = ConstantPE(1.0, 100)
        mix = MixPE([GainPE(source, 0.5), GainPE(source, 0.3)])
        renderer.set_source(mix)
        assert renderer.source is mix

    def test_diamond_graph_renders(self):
        renderer = MockRenderer()
        #     source
        #     /    \
        #  gain1  gain2
        #     \    /
        #      mix
        source = ConstantPE(1.0, 100)
        mix = MixPE([GainPE(source, 0.5), GainPE(source, 0.3)])
        renderer.set_source(mix)
        renderer.start()
        renderer.render(0, 50)
        assert renderer.output_snippets[0].duration == 50
        renderer.stop()

    def test_multi_level_reuse_renders(self):
        renderer = MockRenderer()
        source = ConstantPE(1.0, 100)
        gain1 = GainPE(source, 0.5)
        mix = MixPE([GainPE(gain1, 0.3), GainPE(gain1, 0.2)])
        renderer.set_source(mix)
        renderer.start()
        renderer.render(0, 50)
        renderer.stop()


class TestRendererConfiguration:
    """Test renderer behavior with global sample rate."""

    def test_set_source_does_not_override_global_sample_rate(self):
        """Renderer should not override the global sample rate."""
        import pygmu2 as pg

        pg.set_sample_rate(48000)

        renderer = MockRenderer(sample_rate=96000)
        source = ConstantPE(1.0, 100)

        renderer.set_source(source)

        assert source.sample_rate == 48000

    def test_sample_rate_applies_at_construction(self):
        """PEs capture the global sample rate at construction time."""
        import pygmu2 as pg

        pg.set_sample_rate(44100)
        source1 = ConstantPE(1.0, 100)

        pg.set_sample_rate(48000)
        source2 = ConstantPE(1.0, 100)

        assert source1.sample_rate == 44100
        assert source2.sample_rate == 48000


class LifecycleTrackingPE(SourcePE):
    """A PE that tracks on_start/on_stop calls for testing."""

    def __init__(self):
        self.start_count = 0
        self.stop_count = 0

    def _render(self, start: int, duration: int) -> Snippet:
        return Snippet.from_zeros(start, duration, 1)

    def extent(self) -> Extent:
        return Extent(0, 1000)

    def channel_count(self) -> int:
        return 1

    def on_start(self) -> None:
        self.start_count += 1

    def on_stop(self) -> None:
        self.stop_count += 1


class LifecycleTrackingProcessorPE(ProcessingElement):
    """A processor PE that tracks on_start/on_stop calls."""

    def __init__(self, source: ProcessingElement):
        self._source = source
        self.start_count = 0
        self.stop_count = 0

    def _render(self, start: int, duration: int) -> Snippet:
        return self._source.render(start, duration)

    def extent(self) -> Extent:
        return self._source.extent()

    def inputs(self) -> list[ProcessingElement]:
        return [self._source]

    def on_start(self) -> None:
        self.start_count += 1

    def on_stop(self) -> None:
        self.stop_count += 1


class TestRendererLifecycle:
    """Test Renderer lifecycle (start/stop)."""

    def test_start_requires_source(self):
        """Test that start() requires set_source() first."""
        renderer = MockRenderer()
        with pytest.raises(RuntimeError, match="No source set"):
            renderer.start()

    def test_start_sets_started_flag(self):
        """Test that start() sets the started flag."""
        renderer = MockRenderer()
        source = ConstantPE(1.0, 100)
        renderer.set_source(source)

        assert renderer.started is False
        renderer.start()
        assert renderer.started is True
        renderer.stop()

    def test_double_start_raises(self):
        """Test that calling start() twice raises error."""
        renderer = MockRenderer()
        source = ConstantPE(1.0, 100)
        renderer.set_source(source)
        renderer.start()

        with pytest.raises(RuntimeError, match="Already started"):
            renderer.start()
        renderer.stop()

    def test_stop_clears_started_flag(self):
        """Test that stop() clears the started flag."""
        renderer = MockRenderer()
        source = ConstantPE(1.0, 100)
        renderer.set_source(source)
        renderer.start()
        renderer.stop()

        assert renderer.started is False

    def test_stop_is_idempotent(self):
        """Test that stop() can be called multiple times."""
        renderer = MockRenderer()
        source = ConstantPE(1.0, 100)
        renderer.set_source(source)
        renderer.start()
        renderer.stop()
        renderer.stop()  # Should not raise
        renderer.stop()  # Should not raise

    def test_set_source_while_started_raises(self):
        """Test that set_source() while started raises error."""
        renderer = MockRenderer()
        source1 = ConstantPE(1.0, 100)
        source2 = ConstantPE(1.0, 100)

        renderer.set_source(source1)
        renderer.start()

        with pytest.raises(RuntimeError, match="Cannot set source while started"):
            renderer.set_source(source2)
        renderer.stop()

    def test_on_start_called(self):
        """Test that on_start() is called on PEs."""
        renderer = MockRenderer()
        source = LifecycleTrackingPE()
        renderer.set_source(source)

        assert source.start_count == 0
        renderer.start()
        assert source.start_count == 1
        renderer.stop()

    def test_on_stop_called(self):
        """Test that on_stop() is called on PEs."""
        renderer = MockRenderer()
        source = LifecycleTrackingPE()
        renderer.set_source(source)
        renderer.start()

        assert source.stop_count == 0
        renderer.stop()
        assert source.stop_count == 1

    def test_on_start_bottom_up_order(self):
        """Test that on_start() is called bottom-up (inputs first)."""
        renderer = MockRenderer()
        source = LifecycleTrackingPE()
        processor = LifecycleTrackingProcessorPE(source)
        renderer.set_source(processor)

        # Track call order
        call_order = []
        source.on_start = lambda: call_order.append("source")
        processor.on_start = lambda: call_order.append("processor")

        renderer.start()
        assert call_order == ["source", "processor"]
        renderer.stop()

    def test_on_stop_top_down_order(self):
        """Test that on_stop() is called top-down (outputs first)."""
        renderer = MockRenderer()
        source = LifecycleTrackingPE()
        processor = LifecycleTrackingProcessorPE(source)
        renderer.set_source(processor)
        renderer.start()

        # Track call order
        call_order = []
        source.on_stop = lambda: call_order.append("source")
        processor.on_stop = lambda: call_order.append("processor")

        renderer.stop()
        assert call_order == ["processor", "source"]

    def test_diamond_graph_on_start_called_once(self):
        """Test that on_start() is called once per PE in diamond graph."""
        renderer = MockRenderer()
        source = LifecycleTrackingPE()
        proc1 = LifecycleTrackingProcessorPE(source)
        proc2 = LifecycleTrackingProcessorPE(source)
        mix = MixPE([proc1, proc2])
        renderer.set_source(mix)

        renderer.start()
        assert source.start_count == 1  # Only once, not twice
        renderer.stop()

    def test_context_manager(self):
        """Test context manager calls stop() on exit."""
        source = LifecycleTrackingPE()

        with MockRenderer() as renderer:
            renderer.set_source(source)
            renderer.start()
            assert source.stop_count == 0

        assert source.stop_count == 1

    def test_context_manager_calls_stop_on_exception(self):
        """Test context manager calls stop() even on exception."""
        source = LifecycleTrackingPE()

        try:
            with MockRenderer() as renderer:
                renderer.set_source(source)
                renderer.start()
                raise ValueError("test error")
        except ValueError:
            pass

        assert source.stop_count == 1

    def test_restart_after_stop(self):
        """Test that graph can be restarted after stop."""
        renderer = MockRenderer()
        source = LifecycleTrackingPE()
        renderer.set_source(source)

        renderer.start()
        renderer.stop()

        renderer.start()  # Should work
        assert source.start_count == 2
        renderer.stop()
        assert source.stop_count == 2


class TestNullRenderer:
    """Test NullRenderer concrete implementation."""

    def test_null_renderer_creation(self):
        """Test creating a NullRenderer."""
        renderer = NullRenderer(sample_rate=48000)
        assert renderer.sample_rate == 48000

    def test_null_renderer_renders_silently(self):
        """Test that NullRenderer renders without error."""
        renderer = NullRenderer()
        source = ConstantPE(1.0, 1000)
        renderer.set_source(source)
        renderer.start()

        # Should complete without error
        renderer.render(0, 100)
        renderer.render(100, 100)
        renderer.stop()

    def test_null_renderer_with_context_manager(self):
        """Test NullRenderer with context manager."""
        source = LifecycleTrackingPE()

        with NullRenderer() as renderer:
            renderer.set_source(source)
            renderer.start()
            renderer.render(0, 100)

        assert source.stop_count == 1

    def test_null_renderer_drives_side_effects(self):
        """Test that NullRenderer can drive side-effect PEs."""
        # Track render calls
        render_count = [0]

        class CountingPE(SourcePE):
            def _render(self, start: int, duration: int) -> Snippet:
                render_count[0] += 1
                return Snippet.from_zeros(start, duration, 1)

            def extent(self) -> Extent:
                return Extent(0, 10000)

            def channel_count(self) -> int:
                return 1

        renderer = NullRenderer()
        source = CountingPE()
        renderer.set_source(source)
        renderer.start()

        renderer.render(0, 100)
        renderer.render(100, 100)
        renderer.render(200, 100)

        assert render_count[0] == 3
        renderer.stop()
