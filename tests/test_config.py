"""
Tests for the config module (global sample rate) and plain-raise error
behaviour. The configurable ErrorMode was deleted (DESIGN_PHILOSOPHY.md
R3): errors are plain exceptions, always.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import pygmu2.config as cfg
from pygmu2 import NullRenderer, ConstantPE, set_sample_rate, get_sample_rate


class TestSampleRate:
    """The global sample rate is the single source of truth."""

    def test_set_and_get(self):
        set_sample_rate(48000)
        assert get_sample_rate() == 48000
        set_sample_rate(44100)  # restore for other tests

    def test_coerced_to_int(self):
        set_sample_rate(44100.0)
        assert get_sample_rate() == 44100
        assert isinstance(get_sample_rate(), int)

    def test_missing_sample_rate_raises_on_construction(self):
        prev = cfg.get_sample_rate()
        cfg._SAMPLE_RATE = None  # test-only: clear global sample rate
        try:
            with pytest.raises(RuntimeError, match="Global sample_rate is required"):
                ConstantPE(1.0, channels=1)
        finally:
            cfg._SAMPLE_RATE = prev

    def test_error_mode_machinery_is_gone(self):
        """The deleted names must not quietly return (R3 gate)."""
        import pygmu2

        for name in ("ErrorMode", "handle_error", "set_error_mode", "get_error_mode"):
            with pytest.raises(AttributeError):
                getattr(pygmu2, name)


class TestRendererErrors:
    """Renderer misuse raises plain exceptions — no lenient mode."""

    def test_double_start_raises(self):
        renderer = NullRenderer()
        renderer.set_source(ConstantPE(1.0, channels=1))
        renderer.start()
        with pytest.raises(RuntimeError, match="Already started"):
            renderer.start()
        renderer.stop()

    def test_set_source_while_started_raises(self):
        renderer = NullRenderer()
        renderer.set_source(ConstantPE(1.0, channels=1))
        renderer.start()
        with pytest.raises(RuntimeError, match="Cannot set source while started"):
            renderer.set_source(ConstantPE(2.0, channels=1))
        renderer.stop()

    def test_render_without_source_raises(self):
        with pytest.raises(RuntimeError, match="No source set"):
            NullRenderer().render(0, 100)

    def test_render_without_start_raises(self):
        renderer = NullRenderer()
        renderer.set_source(ConstantPE(1.0, channels=1))
        with pytest.raises(RuntimeError, match="Not started"):
            renderer.render(0, 100)

    def test_start_without_source_raises(self):
        with pytest.raises(RuntimeError, match="No source set"):
            NullRenderer().start()
