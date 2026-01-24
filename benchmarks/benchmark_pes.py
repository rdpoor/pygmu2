#!/usr/bin/env python3
"""
Benchmark suite for pygmu2 Processing Elements.

Auto-discovers all PE classes and benchmarks them. Each PE can optionally
define a `benchmark_configs()` class method to provide custom configurations.

Run with: python benchmarks/benchmark_pes.py

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

import sys
import time
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Any
import numpy as np

# Add src to path for development
sys.path.insert(0, 'src')

import pygmu2
from pygmu2 import ProcessingElement, NullRenderer


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""
    name: str
    factory: Callable[[], ProcessingElement]
    category: str = "uncategorized"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    category: str
    samples_per_run: int
    num_runs: int
    total_time_s: float
    times_s: list[float]
    
    @property
    def mean_time_ms(self) -> float:
        return np.mean(self.times_s) * 1000
    
    @property
    def std_time_ms(self) -> float:
        return np.std(self.times_s) * 1000
    
    @property
    def min_time_ms(self) -> float:
        return np.min(self.times_s) * 1000
    
    @property
    def max_time_ms(self) -> float:
        return np.max(self.times_s) * 1000
    
    @property
    def samples_per_second(self) -> float:
        if self.mean_time_ms == 0:
            return 0
        return self.samples_per_run / (self.mean_time_ms / 1000)
    
    @property
    def realtime_ratio(self) -> float:
        """Ratio vs realtime at 44100 Hz (>1 = faster than realtime)."""
        realtime_s = self.samples_per_run / 44100
        return realtime_s / (self.mean_time_ms / 1000) if self.mean_time_ms > 0 else 0


def discover_pe_classes() -> list[type]:
    """
    Discover all ProcessingElement subclasses exported by pygmu2.
    
    Uses __all__ to find exports (including lazy imports).
    
    Returns:
        List of PE classes
    """
    pe_classes = []
    
    # Use __all__ to get all exports (includes lazy imports)
    for name in pygmu2.__all__:
        if name.startswith('_'):
            continue
        try:
            obj = getattr(pygmu2, name)
            if (
                isinstance(obj, type) 
                and issubclass(obj, ProcessingElement)
                and obj is not ProcessingElement
            ):
                pe_classes.append(obj)
        except (ImportError, AttributeError) as e:
            # Skip items that fail to import (e.g., missing scipy)
            print(f"  Note: {name} not available ({e})")
    
    return pe_classes


def get_benchmark_configs(pe_class: type) -> list[BenchmarkConfig]:
    """
    Get benchmark configurations for a PE class.
    
    If the class defines a `benchmark_configs()` class method, use that.
    Otherwise, try to create a default configuration.
    
    Args:
        pe_class: The ProcessingElement subclass
    
    Returns:
        List of BenchmarkConfig, may be empty if PE can't be benchmarked
    """
    # Check for explicit benchmark_configs method
    if hasattr(pe_class, 'benchmark_configs'):
        try:
            return pe_class.benchmark_configs()
        except Exception as e:
            print(f"  Warning: {pe_class.__name__}.benchmark_configs() failed: {e}")
            return []
    
    # Try default instantiation (works for some source PEs)
    try:
        sig = inspect.signature(pe_class.__init__)
        params = list(sig.parameters.values())[1:]  # Skip 'self'
        
        # Check if all params have defaults
        all_have_defaults = all(
            p.default is not inspect.Parameter.empty 
            for p in params
        )
        
        if all_have_defaults:
            return [BenchmarkConfig(
                name=pe_class.__name__,
                factory=lambda cls=pe_class: cls(),
                category="auto-discovered",
            )]
    except Exception:
        pass
    
    return []


def benchmark_pe(
    config: BenchmarkConfig,
    sample_rate: int = 44100,
    samples_per_run: int = 44100,  # 1 second by default
    num_runs: int = 50,
    warmup_runs: int = 5,
) -> BenchmarkResult:
    """
    Benchmark a ProcessingElement's render performance.
    
    Args:
        config: Benchmark configuration
        sample_rate: Sample rate for the renderer
        samples_per_run: Number of samples to render per run
        num_runs: Number of timed runs
        warmup_runs: Number of warmup runs (not timed)
    
    Returns:
        BenchmarkResult with timing statistics
    """
    # Create PE and renderer
    pe = config.factory()
    renderer = NullRenderer(sample_rate=sample_rate)
    renderer.set_source(pe)
    renderer.start()
    
    # Warmup
    for i in range(warmup_runs):
        pe.render(i * samples_per_run, samples_per_run)
    
    # Timed runs
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        pe.render((warmup_runs + i) * samples_per_run, samples_per_run)
        end = time.perf_counter()
        times.append(end - start)
    
    renderer.stop()
    
    return BenchmarkResult(
        name=config.name,
        category=config.category,
        samples_per_run=samples_per_run,
        num_runs=num_runs,
        total_time_s=sum(times),
        times_s=times,
    )


# =============================================================================
# Default benchmark configurations for PEs that need special setup
# =============================================================================
# These are registered via the benchmark_configs() class method on each PE.
# For PEs without this method, we try default instantiation.
#
# To add benchmarks to a PE, add this class method:
#
#     @classmethod
#     def benchmark_configs(cls):
#         from benchmarks.benchmark_pes import BenchmarkConfig
#         return [
#             BenchmarkConfig(
#                 name="MyPE (config 1)",
#                 factory=lambda: cls(param1=value1),
#                 category="source",
#             ),
#         ]
# =============================================================================

# Fallback configurations for PEs that don't define benchmark_configs()
# This allows the benchmark to work without modifying PE source files
FALLBACK_CONFIGS: dict[str, list[BenchmarkConfig]] = {}


def register_fallback(pe_name: str, configs: list[BenchmarkConfig]) -> None:
    """Register fallback benchmark configs for a PE class."""
    FALLBACK_CONFIGS[pe_name] = configs


def setup_fallback_configs():
    """Set up fallback configurations for standard PEs."""
    
    # Import what we need
    from pygmu2 import (
        SinePE, ConstantPE, RampPE, IdentityPE, DiracPE,
        BlitSawPE, SuperSawPE,
        GainPE, DelayPE, CropPE, MixPE, TransformPE,
        EnvelopePE, WindowPE, LoopPE,
        DynamicsPE, DynamicsMode, CompressorPE, LimiterPE, GatePE,
        RandomPE, RandomMode,
        Extent,
    )
    
    # Try to import optional PEs
    try:
        from pygmu2 import BiquadPE, BiquadMode
        has_biquad = True
    except ImportError:
        has_biquad = False
    
    # === Source PEs ===
    register_fallback("SinePE", [
        BenchmarkConfig("SinePE (440 Hz)", lambda: SinePE(frequency=440.0), "source"),
    ])
    
    register_fallback("ConstantPE", [
        BenchmarkConfig("ConstantPE", lambda: ConstantPE(0.5), "source"),
    ])
    
    register_fallback("RampPE", [
        BenchmarkConfig("RampPE", lambda: RampPE(0.0, 1.0, duration=44100), "source"),
    ])
    
    register_fallback("IdentityPE", [
        BenchmarkConfig("IdentityPE", lambda: IdentityPE(), "source"),
    ])
    
    register_fallback("DiracPE", [
        BenchmarkConfig("DiracPE", lambda: DiracPE(), "source"),
    ])
    
    # === Oscillators ===
    register_fallback("BlitSawPE", [
        BenchmarkConfig("BlitSawPE (440 Hz, auto M)", lambda: BlitSawPE(frequency=440.0), "oscillator"),
        BenchmarkConfig("BlitSawPE (440 Hz, M=20)", lambda: BlitSawPE(frequency=440.0, m=20), "oscillator"),
    ])
    
    register_fallback("SuperSawPE", [
        BenchmarkConfig("SuperSawPE (7 voices)", lambda: SuperSawPE(frequency=440.0, voices=7), "oscillator"),
        BenchmarkConfig("SuperSawPE (3 voices)", lambda: SuperSawPE(frequency=440.0, voices=3), "oscillator"),
    ])
    
    # === Transform PEs ===
    register_fallback("GainPE", [
        BenchmarkConfig("GainPE (constant)", lambda: GainPE(SinePE(frequency=440.0), gain=0.5), "transform"),
        BenchmarkConfig("GainPE (modulated)", lambda: GainPE(SinePE(frequency=440.0), gain=SinePE(frequency=5.0)), "transform"),
    ])
    
    register_fallback("DelayPE", [
        BenchmarkConfig("DelayPE (1000 samples)", lambda: DelayPE(SinePE(frequency=440.0), delay=1000), "transform"),
    ])
    
    register_fallback("CropPE", [
        BenchmarkConfig("CropPE", lambda: CropPE(SinePE(frequency=440.0), Extent(0, 44100)), "transform"),
    ])
    
    register_fallback("MixPE", [
        BenchmarkConfig("MixPE (2 sources)", lambda: MixPE(SinePE(frequency=440.0), SinePE(frequency=550.0)), "transform"),
        BenchmarkConfig("MixPE (4 sources)", lambda: MixPE(
            SinePE(frequency=440.0), SinePE(frequency=550.0),
            SinePE(frequency=660.0), SinePE(frequency=880.0),
        ), "transform"),
    ])
    
    register_fallback("LoopPE", [
        BenchmarkConfig("LoopPE", lambda: LoopPE(CropPE(SinePE(frequency=440.0), Extent(0, 4410))), "transform"),
    ])
    
    # === Dynamics PEs ===
    register_fallback("DynamicsPE", [
        BenchmarkConfig("DynamicsPE (compress)", lambda: DynamicsPE(
            source=SinePE(frequency=440.0), 
            envelope=EnvelopePE(SinePE(frequency=440.0)),
            mode=DynamicsMode.COMPRESS, 
            threshold=-10.0, 
            ratio=4.0
        ), "dynamics"),
    ])

    register_fallback("CompressorPE", [
        BenchmarkConfig("CompressorPE", lambda: CompressorPE(SinePE(frequency=440.0)), "dynamics"),
    ])

    register_fallback("LimiterPE", [
        BenchmarkConfig("LimiterPE", lambda: LimiterPE(SinePE(frequency=440.0)), "dynamics"),
    ])

    register_fallback("GatePE", [
        BenchmarkConfig("GatePE", lambda: GatePE(SinePE(frequency=440.0)), "dynamics"),
    ])

    # === Random PEs ===
    register_fallback("RandomPE", [
        BenchmarkConfig("RandomPE (Sample & Hold)", lambda: RandomPE(mode=RandomMode.SAMPLE_HOLD), "source"),
        BenchmarkConfig("RandomPE (Linear)", lambda: RandomPE(mode=RandomMode.LINEAR), "source"),
        BenchmarkConfig("RandomPE (Walk)", lambda: RandomPE(mode=RandomMode.WALK), "source"),
    ])

    # === Analysis PEs ===
    register_fallback("EnvelopePE", [
        BenchmarkConfig("EnvelopePE (RMS)", lambda: EnvelopePE(SinePE(frequency=440.0)), "analysis"),
    ])
    
    register_fallback("WindowPE", [
        BenchmarkConfig("WindowPE (max)", lambda: WindowPE(SinePE(frequency=440.0)), "analysis"),
    ])
    
    # === Filter PEs ===
    if has_biquad:
        register_fallback("BiquadPE", [
            BenchmarkConfig("BiquadPE (lowpass, fixed)", lambda: BiquadPE(
                SinePE(frequency=440.0), mode=BiquadMode.LOWPASS, frequency=1000.0, q=0.707,
            ), "filter"),
            BenchmarkConfig("BiquadPE (bandpass, fixed)", lambda: BiquadPE(
                SinePE(frequency=440.0), mode=BiquadMode.BANDPASS, frequency=1000.0, q=2.0,
            ), "filter"),
            BenchmarkConfig("BiquadPE (lowpass, modulated freq)", lambda: BiquadPE(
                SinePE(frequency=440.0), mode=BiquadMode.LOWPASS, 
                frequency=SinePE(frequency=5.0, amplitude=500.0), q=0.707,
            ), "filter"),
            BenchmarkConfig("BiquadPE (bandpass, modulated Q)", lambda: BiquadPE(
                SinePE(frequency=440.0), mode=BiquadMode.BANDPASS, 
                frequency=1000.0, q=SinePE(frequency=2.0, amplitude=1.0),
            ), "filter"),
        ])


def collect_all_configs() -> list[BenchmarkConfig]:
    """
    Collect all benchmark configurations from discovered PEs.
    
    Returns:
        List of all BenchmarkConfig objects
    """
    setup_fallback_configs()
    
    all_configs = []
    pe_classes = discover_pe_classes()
    
    print(f"Discovered {len(pe_classes)} PE classes")
    print()
    
    for pe_class in sorted(pe_classes, key=lambda c: c.__name__):
        name = pe_class.__name__
        
        # Check for class method first
        if hasattr(pe_class, 'benchmark_configs'):
            try:
                configs = pe_class.benchmark_configs()
                all_configs.extend(configs)
                print(f"  {name}: {len(configs)} config(s) from class method")
                continue
            except Exception as e:
                print(f"  {name}: benchmark_configs() failed: {e}")
        
        # Check fallback configs
        if name in FALLBACK_CONFIGS:
            configs = FALLBACK_CONFIGS[name]
            all_configs.extend(configs)
            print(f"  {name}: {len(configs)} config(s) from fallback")
            continue
        
        # Try default instantiation
        configs = get_benchmark_configs(pe_class)
        if configs:
            all_configs.extend(configs)
            print(f"  {name}: {len(configs)} config(s) from auto-discovery")
        else:
            print(f"  {name}: skipped (no config available)")
    
    return all_configs


def run_benchmarks(
    configs: list[BenchmarkConfig],
    num_runs: int = 50,
    warmup_runs: int = 5,
) -> list[BenchmarkResult]:
    """Run benchmarks for all configurations."""
    results = []
    
    print()
    print("=" * 70)
    print("RUNNING BENCHMARKS")
    print("=" * 70)
    
    # Group by category
    by_category: dict[str, list[BenchmarkConfig]] = {}
    for config in configs:
        by_category.setdefault(config.category, []).append(config)
    
    for category in sorted(by_category.keys()):
        print()
        print(f"{category.upper()}:")
        print("-" * 40)
        
        for config in by_category[category]:
            try:
                result = benchmark_pe(config, num_runs=num_runs, warmup_runs=warmup_runs)
                results.append(result)
                print(f"  {result.name}: {result.mean_time_ms:.3f} ms ({result.realtime_ratio:.1f}x realtime)")
            except Exception as e:
                print(f"  {config.name}: FAILED - {e}")
    
    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary table of all results."""
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Benchmark':<35} {'Mean (ms)':>10} {'Std (ms)':>10} {'RT Ratio':>10}")
    print("-" * 70)
    
    # Sort by mean time descending
    sorted_results = sorted(results, key=lambda r: r.mean_time_ms, reverse=True)
    
    for r in sorted_results:
        print(f"{r.name:<35} {r.mean_time_ms:>10.3f} {r.std_time_ms:>10.3f} {r.realtime_ratio:>10.1f}x")
    
    print("-" * 70)
    
    # Identify bottlenecks
    print()
    print("POTENTIAL BOTTLENECKS (slowest 5):")
    for r in sorted_results[:5]:
        if r.realtime_ratio < 100:
            status = "⚠️  SLOW" if r.realtime_ratio < 10 else "OK"
        else:
            status = "FAST"
        print(f"  {r.name}: {r.mean_time_ms:.3f} ms/render, {r.realtime_ratio:.1f}x realtime [{status}]")


def run_scaling_test() -> None:
    """Test how performance scales with buffer size."""
    from pygmu2 import BlitSawPE, SuperSawPE
    
    print()
    print("=" * 70)
    print("BUFFER SIZE SCALING TEST")
    print("=" * 70)
    print()
    
    buffer_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    print(f"{'Buffer Size':>12} {'BlitSaw (ms)':>14} {'SuperSaw (ms)':>14} {'Ratio':>10}")
    print("-" * 55)
    
    for buf_size in buffer_sizes:
        blit_config = BenchmarkConfig("BlitSaw", lambda: BlitSawPE(frequency=440.0), "test")
        super_config = BenchmarkConfig("SuperSaw", lambda: SuperSawPE(frequency=440.0, voices=7), "test")
        
        blit_result = benchmark_pe(blit_config, samples_per_run=buf_size, num_runs=100, warmup_runs=10)
        super_result = benchmark_pe(super_config, samples_per_run=buf_size, num_runs=100, warmup_runs=10)
        
        ratio = super_result.mean_time_ms / blit_result.mean_time_ms if blit_result.mean_time_ms > 0 else 0
        
        print(f"{buf_size:>12} {blit_result.mean_time_ms:>14.4f} {super_result.mean_time_ms:>14.4f} {ratio:>10.2f}x")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark pygmu2 PEs")
    parser.add_argument("--scaling", action="store_true", help="Run buffer size scaling test")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer runs)")
    parser.add_argument("--list", action="store_true", help="List discovered PEs without running benchmarks")
    args = parser.parse_args()
    
    print("pygmu2 PE Benchmark Suite")
    print("=" * 70)
    print()
    
    configs = collect_all_configs()
    
    if args.list:
        print(f"\nTotal: {len(configs)} benchmark configurations")
        sys.exit(0)
    
    num_runs = 10 if args.quick else 50
    warmup_runs = 2 if args.quick else 5
    
    results = run_benchmarks(configs, num_runs=num_runs, warmup_runs=warmup_runs)
    print_summary(results)
    
    if args.scaling:
        run_scaling_test()
