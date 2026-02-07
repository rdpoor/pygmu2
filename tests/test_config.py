"""
Tests for config module and error handling utilities.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import pygmu2
from pygmu2.config import (
    ErrorMode,
    DEFAULT_ERROR_MODE,
    set_error_mode,
    get_error_mode,
    handle_error,
)
from pygmu2 import NullRenderer, ConstantPE


class TestErrorMode:
    """Test ErrorMode enum."""
    
    def test_strict_mode_value(self):
        assert ErrorMode.STRICT.value == "strict"
    
    def test_lenient_mode_value(self):
        assert ErrorMode.LENIENT.value == "lenient"


class TestGetSetErrorMode:
    """Test get/set error mode functions."""
    
    def setup_method(self):
        """Save original mode before each test."""
        self._original_mode = get_error_mode()
    
    def teardown_method(self):
        """Restore original mode after each test."""
        set_error_mode(self._original_mode)
    
    def test_default_is_strict(self):
        # Reset to default
        set_error_mode(ErrorMode.STRICT)
        assert get_error_mode() == ErrorMode.STRICT
    
    def test_set_lenient(self):
        set_error_mode(ErrorMode.LENIENT)
        assert get_error_mode() == ErrorMode.LENIENT
    
    def test_set_strict(self):
        set_error_mode(ErrorMode.LENIENT)
        set_error_mode(ErrorMode.STRICT)
        assert get_error_mode() == ErrorMode.STRICT


class TestHandleError:
    """Test handle_error utility function."""
    
    def setup_method(self):
        """Save original mode before each test."""
        self._original_mode = get_error_mode()
    
    def teardown_method(self):
        """Restore original mode after each test."""
        set_error_mode(self._original_mode)
    
    def test_strict_mode_raises(self):
        set_error_mode(ErrorMode.STRICT)
        with pytest.raises(RuntimeError, match="test error"):
            handle_error("test error")
    
    def test_lenient_mode_warns(self, caplog):
        set_error_mode(ErrorMode.LENIENT)
        result = handle_error("test warning")
        assert result is True
        assert "test warning" in caplog.text
    
    def test_fatal_always_raises_in_strict(self):
        set_error_mode(ErrorMode.STRICT)
        with pytest.raises(RuntimeError, match="fatal error"):
            handle_error("fatal error", fatal=True)
    
    def test_fatal_always_raises_in_lenient(self):
        set_error_mode(ErrorMode.LENIENT)
        with pytest.raises(RuntimeError, match="fatal error"):
            handle_error("fatal error", fatal=True)
    
    def test_custom_exception_class(self):
        set_error_mode(ErrorMode.STRICT)
        with pytest.raises(ValueError, match="value error"):
            handle_error("value error", exception_class=ValueError)
    
    def test_override_mode_parameter(self):
        set_error_mode(ErrorMode.STRICT)
        # Even in strict mode, passing lenient should warn
        result = handle_error("override test", error_mode=ErrorMode.LENIENT)
        assert result is True
    
    def test_override_to_strict_raises(self):
        set_error_mode(ErrorMode.LENIENT)
        # Even in lenient mode, passing strict should raise
        with pytest.raises(RuntimeError, match="override test"):
            handle_error("override test", error_mode=ErrorMode.STRICT)


class TestRendererErrorHandling:
    """Test Renderer error handling with different modes."""
    
    def setup_method(self):
        """Save original mode before each test."""
        self._original_mode = get_error_mode()
    
    def teardown_method(self):
        """Restore original mode after each test."""
        set_error_mode(self._original_mode)
    
    def test_double_start_strict_raises(self):
        set_error_mode(ErrorMode.STRICT)
        renderer = NullRenderer()
        source = ConstantPE(1.0, channels=1)
        renderer.set_source(source)
        renderer.start()
        with pytest.raises(RuntimeError, match="Already started"):
            renderer.start()
        renderer.stop()
    
    def test_double_start_lenient_warns(self, caplog):
        set_error_mode(ErrorMode.LENIENT)
        renderer = NullRenderer()
        source = ConstantPE(1.0, channels=1)
        renderer.set_source(source)
        renderer.start()
        renderer.start()  # Should warn, not raise
        assert "Already started" in caplog.text
        renderer.stop()
    
    def test_set_source_while_started_strict_raises(self):
        set_error_mode(ErrorMode.STRICT)
        renderer = NullRenderer()
        source1 = ConstantPE(1.0, channels=1)
        source2 = ConstantPE(2.0, channels=1)
        renderer.set_source(source1)
        renderer.start()
        with pytest.raises(RuntimeError, match="Cannot set source while started"):
            renderer.set_source(source2)
        renderer.stop()
    
    def test_set_source_while_started_lenient_warns(self, caplog):
        set_error_mode(ErrorMode.LENIENT)
        renderer = NullRenderer()
        source1 = ConstantPE(1.0, channels=1)
        source2 = ConstantPE(2.0, channels=1)
        renderer.set_source(source1)
        renderer.start()
        renderer.set_source(source2)  # Should warn, not raise
        assert "Cannot set source while started" in caplog.text
        # Source should NOT have changed in lenient mode
        assert renderer.source is source1
        renderer.stop()
    
    def test_render_without_source_always_fatal(self):
        """Render without source is always fatal, even in lenient mode."""
        set_error_mode(ErrorMode.LENIENT)
        renderer = NullRenderer()
        with pytest.raises(RuntimeError, match="No source set"):
            renderer.render(0, 100)
    
    def test_render_without_start_always_fatal(self):
        """Render without start is always fatal, even in lenient mode."""
        set_error_mode(ErrorMode.LENIENT)
        renderer = NullRenderer()
        source = ConstantPE(1.0, channels=1)
        renderer.set_source(source)
        with pytest.raises(RuntimeError, match="Not started"):
            renderer.render(0, 100)
    
    def test_start_without_source_always_fatal(self):
        """Start without source is always fatal, even in lenient mode."""
        set_error_mode(ErrorMode.LENIENT)
        renderer = NullRenderer()
        with pytest.raises(RuntimeError, match="No source set"):
            renderer.start()


class TestPEErrorHandling:
    """Test ProcessingElement error handling with different modes."""
    
    def setup_method(self):
        """Save original mode before each test."""
        self._original_mode = get_error_mode()
    
    def teardown_method(self):
        """Restore original mode after each test."""
        set_error_mode(self._original_mode)
    
    def test_missing_sample_rate_always_fatal(self):
        """Constructing a PE without global sample_rate is always fatal."""
        import pygmu2.config as cfg
        set_error_mode(ErrorMode.LENIENT)
        prev = cfg.get_sample_rate()
        cfg._SAMPLE_RATE = None  # test-only: clear global sample rate
        try:
            with pytest.raises(RuntimeError, match="Global sample_rate is required"):
                ConstantPE(1.0, channels=1)
        finally:
            cfg._SAMPLE_RATE = prev
