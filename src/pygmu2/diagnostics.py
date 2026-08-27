"""
Render diagnostics: per-PE render timing and pull counts.

This is the sanctioned profiling mechanism (DESIGN_PHILOSOPHY.md PD-2:
optimization waits for a profile). Hooks in ProcessingElement.render()
are zero-cost when disabled.

Simplest use — the context manager:

    from pygmu2 import diagnostics

    with diagnostics.profile() as report:
        renderer.render(0, 44100)
    print(report.summary())

Report rows are per PE class: total ms, call count, avg ms, samples/s,
and realtime ratio (>1 means faster than realtime at the given rate).

Lower-level use (per-block): enable(), reset_block() at block start,
render, then get_block_report().

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from pygmu2.processing_element import ProcessingElement

_thread_local = threading.local()


def _state() -> Any:
    if not hasattr(_thread_local, "pull_count"):
        _thread_local.pull_count = {}
        _thread_local.pull_count_class = {}
        _thread_local.timings = []  # list of (class_name, duration_ns, samples)
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


def record_timing(pe: "ProcessingElement", duration_ns: int, samples: int) -> None:
    """Record _render() duration for this PE. No-op if timing disabled."""
    s = _state()
    if not s.enabled_timing:
        return
    s.timings.append((pe.__class__.__name__, duration_ns, samples))


def is_enabled() -> bool:
    """True if either pull_count or timing is enabled."""
    s = _state()
    return bool(s.enabled_pull or s.enabled_timing)


def pull_count_enabled() -> bool:
    return bool(_state().enabled_pull)


def timing_enabled() -> bool:
    return bool(_state().enabled_timing)


def get_block_report() -> str:
    """
    Return a concise report for the last block: pull counts by PE class,
    then timing by PE class (total ms, call count, avg ms), sorted by total time.
    """
    s = _state()
    lines = []

    if s.enabled_pull and (s.pull_count or s.pull_count_class):
        by_class: dict[str, int] = defaultdict(int)
        for pe_id, count in s.pull_count.items():
            cls = s.pull_count_class.get(pe_id, "?")
            by_class[cls] += count
        lines.append("pull_count:")
        for cls in sorted(by_class.keys()):
            lines.append(f"  {cls}: {by_class[cls]}")
        lines.append("")

    if s.enabled_timing and s.timings:
        timing_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total_ns": 0, "count": 0}
        )
        for cls, dur_ns, _samples in s.timings:
            timing_stats[cls]["total_ns"] += dur_ns
            timing_stats[cls]["count"] += 1
        lines.append("timing_ms (total, count, avg):")
        sorted_classes = sorted(
            timing_stats.keys(),
            key=lambda c: timing_stats[c]["total_ns"],
            reverse=True,
        )
        for cls in sorted_classes:
            total_ns = timing_stats[cls]["total_ns"]
            count = timing_stats[cls]["count"]
            total_ms = total_ns / 1_000_000
            avg_ms = total_ms / count if count else 0
            lines.append(
                f"  {cls}: total={total_ms:.2f} count={count} avg={avg_ms:.4f}"
            )
        lines.append("")

    if not lines:
        return "diagnostics: (no data; enable and reset_block before render)"
    return "diagnostics:\n" + "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Context-manager facade
# ---------------------------------------------------------------------------


@dataclass
class ClassStats:
    """Aggregated render statistics for one PE class."""

    total_ns: int = 0
    count: int = 0
    samples: int = 0

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    @property
    def samples_per_second(self) -> float:
        if self.total_ns == 0:
            return 0.0
        return self.samples / (self.total_ns / 1_000_000_000)

    def realtime_ratio(self, sample_rate: int = 44100) -> float:
        """>1 means this class renders faster than realtime at sample_rate."""
        if sample_rate <= 0:
            return 0.0
        return self.samples_per_second / sample_rate


@dataclass
class Profile:
    """Captured profiling data from a `with diagnostics.profile():` block."""

    by_class: dict[str, ClassStats] = field(default_factory=dict)
    pull_counts: dict[str, int] = field(default_factory=dict)

    def summary(self, sample_rate: int = 44100) -> str:
        lines = [
            "RENDER PROFILE (per PE class, sorted by total time)",
            "-" * 78,
            f"{'PE class':<24} {'calls':>7} {'total ms':>10} {'avg ms':>9} "
            f"{'samples/s':>12} {'x realtime':>11}",
            "-" * 78,
        ]
        for cls, st in sorted(
            self.by_class.items(), key=lambda kv: kv[1].total_ns, reverse=True
        ):
            lines.append(
                f"{cls:<24} {st.count:>7} {st.total_ms:>10.2f} {st.avg_ms:>9.4f} "
                f"{st.samples_per_second:>12,.0f} "
                f"{st.realtime_ratio(sample_rate):>10.1f}x"
            )
        if self.pull_counts:
            lines.append("")
            lines.append("pull counts (renders per class):")
            for cls in sorted(self.pull_counts):
                lines.append(f"  {cls}: {self.pull_counts[cls]}")
        return "\n".join(lines)


@contextmanager
def profile(pull_count: bool = True, timing: bool = True) -> "Iterator[Profile]":
    """Profile every PE render inside the block.

    Yields a Profile whose data is filled in when the block exits:

        with diagnostics.profile() as report:
            renderer.render(0, 44100)
        print(report.summary(sample_rate=44100))
    """
    report = Profile()
    was_pull, was_timing = pull_count_enabled(), timing_enabled()
    enable(pull_count=pull_count, timing=timing)
    reset_block()
    try:
        yield report
    finally:
        s = _state()
        for cls, dur_ns, samples in s.timings:
            st = report.by_class.setdefault(cls, ClassStats())
            st.total_ns += dur_ns
            st.count += 1
            st.samples += samples
        by_class_pulls: dict[str, int] = defaultdict(int)
        for pe_id, count in s.pull_count.items():
            by_class_pulls[s.pull_count_class.get(pe_id, "?")] += count
        report.pull_counts = dict(by_class_pulls)
        if was_pull or was_timing:
            enable(pull_count=was_pull, timing=was_timing)
        else:
            disable()
        reset_block()
