"""
Render diagnostics: pull counts and per-PE timing per block.

Use to see how many times each PE is rendered in one block (duplicate pulls)
and how much time each PE type spends in _render(). Enable before a block,
reset_block() at block start, render the block, then get_block_report().

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pygmu2.processing_element import ProcessingElement

_thread_local = threading.local()


def _state():
    if not hasattr(_thread_local, "pull_count"):
        _thread_local.pull_count = {}
        _thread_local.pull_count_class = {}
        _thread_local.timings = []  # list of (class_name, duration_ns)
        _thread_local.enabled_pull = False
        _thread_local.enabled_timing = False
    return _thread_local


def enable(pull_count: bool = True, timing: bool = True) -> None:
    """Enable pull-count and/or per-PE timing for the current thread."""
    s = _state()
    s.enabled_pull = pull_count
    s.enabled_timing = timing


def disable() -> None:
    """Disable diagnostics for the current thread."""
    s = _state()
    s.enabled_pull = False
    s.enabled_timing = False


def reset_block() -> None:
    """Clear pull counts and timings for the next block. Call at block start."""
    s = _state()
    s.pull_count.clear()
    s.pull_count_class.clear()
    s.timings.clear()


def record_pull(pe: "ProcessingElement") -> None:
    """Record one render() call for this PE. No-op if pull_count disabled."""
    s = _state()
    if not s.enabled_pull:
        return
    pe_id = id(pe)
    s.pull_count[pe_id] = s.pull_count.get(pe_id, 0) + 1
    s.pull_count_class[pe_id] = pe.__class__.__name__


def record_timing(pe: "ProcessingElement", duration_ns: int) -> None:
    """Record _render() duration for this PE. No-op if timing disabled."""
    s = _state()
    if not s.enabled_timing:
        return
    s.timings.append((pe.__class__.__name__, duration_ns))


def is_enabled() -> bool:
    """True if either pull_count or timing is enabled."""
    s = _state()
    return s.enabled_pull or s.enabled_timing


def pull_count_enabled() -> bool:
    return _state().enabled_pull


def timing_enabled() -> bool:
    return _state().enabled_timing


def get_block_report() -> str:
    """
    Return a concise report for the last block: pull counts by PE class,
    then timing by PE class (total ms, call count, avg ms), sorted by total time.
    """
    s = _state()
    lines = []

    if s.enabled_pull and (s.pull_count or s.pull_count_class):
        by_class = defaultdict(int)
        for pe_id, count in s.pull_count.items():
            cls = s.pull_count_class.get(pe_id, "?")
            by_class[cls] += count
        lines.append("pull_count:")
        for cls in sorted(by_class.keys()):
            lines.append(f"  {cls}: {by_class[cls]}")
        lines.append("")

    if s.enabled_timing and s.timings:
        by_class = defaultdict(lambda: {"total_ns": 0, "count": 0})
        for cls, dur_ns in s.timings:
            by_class[cls]["total_ns"] += dur_ns
            by_class[cls]["count"] += 1
        lines.append("timing_ms (total, count, avg):")
        sorted_classes = sorted(
            by_class.keys(),
            key=lambda c: by_class[c]["total_ns"],
            reverse=True,
        )
        for cls in sorted_classes:
            total_ns = by_class[cls]["total_ns"]
            count = by_class[cls]["count"]
            total_ms = total_ns / 1_000_000
            avg_ms = total_ms / count if count else 0
            lines.append(f"  {cls}: total={total_ms:.2f} count={count} avg={avg_ms:.4f}")
        lines.append("")

    if not lines:
        return "diagnostics: (no data; enable and reset_block before render)"
    return "diagnostics:\n" + "\n".join(lines).rstrip()
