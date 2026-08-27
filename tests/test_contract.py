"""
Universal PE contract suite (Tier 1 — DESIGN_PHILOSOPHY.md §6).

Auto-discovers every ProcessingElement exported from pygmu2.__all__ and
asserts the base-class contract against each. With runtime graph
validation deleted (PD-2), this suite is the system's systematic
correctness check:

- render(s, d) returns exactly d samples starting at s; 2-D float32;
  read-only buffer; channel count consistent with channel_count()
- samples outside extent() are zero-filled; extent() is stable
- a full-graph reset_state() restores first-render behaviour
- randomised-order rendering either matches the contiguous reference
  (stateless graph) or raises the contiguity error (stateful graph) —
  falsifying the `stateful` declaration in both directions

Coverage is inherited automatically: a PE cannot be exported without a
factory in pe_factories.py (enforced by test_every_pe_has_a_factory),
and every factory is parametrised into every test below.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import pygmu2 as pg
from pygmu2.processing_element import ProcessingElement

from tests.pe_factories import FACTORIES, HARDWARE

BLOCK = 64
NBLOCKS = 6
# Pull 2 first (any first start is legal), then 0 — guaranteed
# non-contiguous for a stateful graph.
SHUFFLED = [2, 0, 3, 1, 5, 4]


def discover_pe_names() -> list[str]:
    """Every concrete ProcessingElement subclass exported by pygmu2."""
    names = []
    for name in pg.__all__:
        try:
            obj = getattr(pg, name)
        except AttributeError:
            continue
        if (
            isinstance(obj, type)
            and issubclass(obj, ProcessingElement)
            and not inspect.isabstract(obj)
        ):
            names.append(name)
    return sorted(names)


PE_NAMES = discover_pe_names()
RENDERABLE = [n for n in PE_NAMES if n not in HARDWARE]


def make(name: str) -> ProcessingElement:
    """Fresh, started instance (real Renderer lifecycle)."""
    pe = FACTORIES[name]()
    renderer = pg.NullRenderer(sample_rate=44100)
    renderer.set_source(pe)
    renderer.start()
    return pe


def graph_walk(pe, seen=None):
    seen = set() if seen is None else seen
    if id(pe) in seen:
        return
    seen.add(id(pe))
    yield pe
    for inp in pe.inputs():
        yield from graph_walk(inp, seen)


def graph_stateful(pe) -> bool:
    return any(p.stateful for p in graph_walk(pe))


def graph_reset(pe) -> None:
    for p in graph_walk(pe):
        p.reset_state()


def render_blocks(pe, order) -> dict[int, np.ndarray]:
    return {i: pe.render(i * BLOCK, BLOCK).data.copy() for i in order}


def test_every_pe_has_a_factory():
    """Registration is the gate: a PE cannot be exported without a
    factory, so a new PE inherits contract coverage for free."""
    missing = set(PE_NAMES) - set(FACTORIES)
    stale = set(FACTORIES) - set(PE_NAMES)
    assert (
        not missing
    ), f"Exported PEs with no factory in pe_factories.py: {sorted(missing)}"
    assert (
        not stale
    ), f"Factories for names not exported (rename fallout?): {sorted(stale)}"


@pytest.mark.parametrize("name", PE_NAMES)
def test_constructs(name):
    pe = FACTORIES[name]()
    assert isinstance(pe, ProcessingElement)
    assert isinstance(pe.stateful, bool)


@pytest.mark.parametrize("name", RENDERABLE)
def test_framing_dtype_and_immutability(name):
    pe = make(name)
    for i in range(2):  # two contiguous blocks
        snippet = pe.render(i * BLOCK, BLOCK)
        assert snippet.start == i * BLOCK
        assert snippet.duration == BLOCK
        assert snippet.data.ndim == 2
        assert snippet.data.shape[0] == BLOCK
        assert snippet.data.dtype == np.float32
        assert not snippet.data.flags.writeable, "Snippet buffers must be read-only"
        declared = pe.channel_count()
        if declared is not None:
            assert snippet.channels == declared


@pytest.mark.parametrize("name", RENDERABLE)
def test_zero_duration_render(name):
    pe = make(name)
    snippet = pe.render(0, 0)
    assert snippet.duration == 0


@pytest.mark.parametrize("name", RENDERABLE)
def test_zero_fill_outside_extent(name):
    pe = make(name)
    extent = pe.extent()
    if extent.end is not None:
        outside = pe.render(extent.end + BLOCK, BLOCK)  # first render: any start
        assert np.all(outside.data == 0.0), "samples past extent.end must be zero"
    pe2 = make(name)
    extent2 = pe2.extent()
    if extent2.start is not None:
        outside = pe2.render(extent2.start - 2 * BLOCK, BLOCK)
        assert np.all(outside.data == 0.0), "samples before extent.start must be zero"


@pytest.mark.parametrize("name", RENDERABLE)
def test_extent_stable(name):
    pe = make(name)
    before = pe.extent()
    pe.render(0, BLOCK)
    pe.render(BLOCK, BLOCK)
    assert pe.extent() == before, "extent() must not change over the PE's lifetime"


@pytest.mark.parametrize("name", RENDERABLE)
def test_reset_restores_first_render(name):
    pe = make(name)
    first = render_blocks(pe, range(2))
    graph_reset(pe)
    again = render_blocks(pe, range(2))
    for i in first:
        np.testing.assert_allclose(
            again[i],
            first[i],
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"{name}: reset_state() did not restore first-render behaviour",
        )


@pytest.mark.parametrize("name", RENDERABLE)
def test_randomised_order_falsifies_stateful(name):
    """The `stateful` declaration, checked in both directions:

    - stateless graph: shuffled pulls must reproduce the contiguous
      reference exactly (a lying 'stateless' PE silently differs -> FAIL)
    - stateful graph: the first out-of-order pull must raise the
      contiguity error (the base class enforces the declaration)
    """
    reference = render_blocks(make(name), range(NBLOCKS))

    pe = make(name)
    if graph_stateful(pe):
        with pytest.raises(RuntimeError, match="non-contiguous render"):
            render_blocks(pe, SHUFFLED)
    else:
        shuffled = render_blocks(pe, SHUFFLED)
        for i in range(NBLOCKS):
            np.testing.assert_allclose(
                shuffled[i],
                reference[i],
                rtol=1e-6,
                atol=1e-7,
                err_msg=(
                    f"{name}: declared stateless but out-of-order rendering "
                    f"differs from the contiguous reference at block {i}"
                ),
            )
