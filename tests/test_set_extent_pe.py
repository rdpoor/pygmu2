"""
Tests for SetExtentPE.
"""

import numpy as np

from pygmu2 import (
    ConstantPE,
    ExtendMode,
    IdentityPE,
    NullRenderer,
    SetExtentPE,
)


def test_set_extent_finite_range():
    source = IdentityPE()
    pe = SetExtentPE(source, 10, 5)
    extent = pe.extent()
    assert extent.start == 10
    assert extent.end == 15


def test_set_extent_open_end():
    source = IdentityPE()
    pe = SetExtentPE(source, 10, None)
    extent = pe.extent()
    assert extent.start == 10
    assert extent.end is None


def test_set_extent_open_start():
    source = IdentityPE()
    pe = SetExtentPE(source, None, 5)
    extent = pe.extent()
    assert extent.start is None
    assert extent.end == 5


def test_set_extent_extends_beyond_source():
    source = ConstantPE(1.0)
    pe = SetExtentPE(source, 0, 10)
    extent = pe.extent()
    assert extent.start == 0
    assert extent.end == 10


def test_set_extent_render_open_start_passthrough():
    source = IdentityPE()
    pe = SetExtentPE(source, None, 5)
    r = NullRenderer(sample_rate=44100)
    r.set_source(pe)
    r.start()

    snippet = pe.render(-3, 5)
    expected = np.arange(-3, 2, dtype=np.float32).reshape(-1, 1)
    np.testing.assert_array_equal(snippet.data, expected)

    snippet2 = pe.render(3, 5)
    expected2 = np.zeros((5, 1), dtype=np.float32)
    expected2[:2, :] = np.arange(3, 5, dtype=np.float32).reshape(-1, 1)
    np.testing.assert_array_equal(snippet2.data, expected2)

    r.stop()


def test_set_extent_hold_first_before_start():
    source = IdentityPE()
    pe = SetExtentPE(source, 0, 5, extend_mode=ExtendMode.HOLD_FIRST)
    r = NullRenderer(sample_rate=44100)
    r.set_source(pe)
    r.start()

    snippet = pe.render(-3, 3)
    expected = np.zeros((3, 1), dtype=np.float32)
    np.testing.assert_array_equal(snippet.data, expected)

    r.stop()
