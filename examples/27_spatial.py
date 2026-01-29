"""
Example 27: Spatial Audio - Panning and Channel Conversion

Demonstrates SpatialPE for channel conversion and spatialization (panning)
using various techniques: SpatialAdapter, SpatialLinear, and SpatialConstantPower.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import (
    AudioRenderer,
    WavReaderPE,
    SpatialPE,
    SpatialAdapter,
    SpatialLinear,
    SpatialConstantPower,
    RampPE,
    DelayPE,
    MixPE,
    GainPE,
    CropPE,
    Extent,
    seconds_to_samples,
)

# Path to audio files (relative to this script)
AUDIO_DIR = Path(__file__).parent / "audio"
SAMPLE_RATE = 44100

def demo_channel_conversion():
    """Demo 1: Channel conversion using SpatialAdapter."""
    print("=== Demo 1: Channel Conversion (Mono → Stereo) ===")
    
    # Load mono audio file
    mono_file = AUDIO_DIR / "acoustic_drums_mono44.wav"
    mono_source = WavReaderPE(str(mono_file))
    
    # Convert mono to stereo
    stereo_output = SpatialPE(mono_source, method=SpatialAdapter(channels=2))
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(stereo_output)
    
    extent = stereo_output.extent()
    duration_samples = extent.end - extent.start
    duration_seconds = duration_samples / SAMPLE_RATE
    print(f"Playing {duration_seconds:.2f} seconds of mono→stereo conversion...", flush=True)
    
    with renderer:
        renderer.start()
        renderer.play_extent()
    
    print("Done!\n", flush=True)


def demo_linear_panning():
    """Demo 2: Linear panning - static positions."""
    print("=== Demo 2: Linear Panning (Static Positions) ===")
    
    # Load mono audio
    mono_file = AUDIO_DIR / "djembe_mono44.wav"
    mono_source = WavReaderPE(str(mono_file))
    
    # Create three panned versions: left, center, right
    left_panned = SpatialPE(mono_source, method=SpatialLinear(azimuth=-90.0))
    center_panned = SpatialPE(mono_source, method=SpatialLinear(azimuth=0.0))
    right_panned = SpatialPE(mono_source, method=SpatialLinear(azimuth=90.0))
    
    # Delay them so they play sequentially
    delay_samples = seconds_to_samples(2.0, SAMPLE_RATE)
    mixed_stream = MixPE(
        left_panned,
        DelayPE(center_panned, delay=delay_samples),
        DelayPE(right_panned, delay=2 * delay_samples),
    )
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(mixed_stream)
    
    extent = mixed_stream.extent()
    duration_samples = extent.end - extent.start
    duration_seconds = duration_samples / SAMPLE_RATE
    print(f"Playing {duration_seconds:.2f} seconds: Left → Center → Right...", flush=True)
    
    with renderer:
        renderer.start()
        renderer.play_extent()
    
    print("Done!\n", flush=True)


def demo_constant_power_panning():
    """Demo 3: Constant-power panning - better stereo balance."""
    print("=== Demo 3: Constant-Power Panning (Static Positions) ===")
    
    # Load mono audio
    mono_file = AUDIO_DIR / "djembe_mono44.wav"
    mono_source = WavReaderPE(str(mono_file))
    
    # Create three panned versions using constant-power
    left_panned = SpatialPE(mono_source, method=SpatialConstantPower(azimuth=-90.0))
    center_panned = SpatialPE(mono_source, method=SpatialConstantPower(azimuth=0.0))
    right_panned = SpatialPE(mono_source, method=SpatialConstantPower(azimuth=90.0))
    
    # Delay them so they play sequentially
    delay_samples = seconds_to_samples(2.0, SAMPLE_RATE)
    mixed_stream = MixPE(
        left_panned,
        DelayPE(center_panned, delay=delay_samples),
        DelayPE(right_panned, delay=2 * delay_samples),
    )
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(mixed_stream)
    
    extent = mixed_stream.extent()
    duration_samples = extent.end - extent.start
    duration_seconds = duration_samples / SAMPLE_RATE
    print(f"Playing {duration_seconds:.2f} seconds: Left → Center → Right (constant-power)...", flush=True)
    
    with renderer:
        renderer.start()
        renderer.play_extent()
    
    print("Done!\n", flush=True)


def demo_dynamic_panning():
    """Demo 4: Dynamic panning - sweeping left to right."""
    print("=== Demo 4: Dynamic Panning (Sweep Left → Right) ===")
    
    # Load mono audio
    mono_file = AUDIO_DIR / "acoustic_drums_mono44.wav"
    mono_source = WavReaderPE(str(mono_file))
    
    # Get duration for the sweep
    extent = mono_source.extent()
    duration_samples = extent.end - extent.start
    
    # Create a ramp that sweeps azimuth from -90° to +90°
    pan_control = RampPE(-90.0, 90.0, duration=duration_samples)
    
    # Apply constant-power panning with dynamic azimuth
    panned_stream = SpatialPE(mono_source, method=SpatialConstantPower(azimuth=pan_control))
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(panned_stream)
    
    duration_seconds = duration_samples / SAMPLE_RATE
    print(f"Playing {duration_seconds:.2f} seconds with panning sweep...", flush=True)
    
    with renderer:
        renderer.start()
        renderer.play_extent()
    
    print("Done!\n", flush=True)


def demo_stereo_to_mono():
    """Demo 5: Stereo to mono conversion."""
    print("=== Demo 5: Stereo to Mono Conversion ===")
    
    # Load stereo audio file (if available, otherwise use mono converted to stereo)
    stereo_file = AUDIO_DIR / "acoustic_drums44.wav"
    if not stereo_file.exists():
        # Fallback: convert mono to stereo first, then back to mono
        mono_file = AUDIO_DIR / "acoustic_drums_mono44.wav"
        mono_source = WavReaderPE(str(mono_file))
        stereo_source = SpatialPE(mono_source, method=SpatialAdapter(channels=2))
    else:
        stereo_source = WavReaderPE(str(stereo_file))
    
    # Convert stereo to mono
    mono_output = SpatialPE(stereo_source, method=SpatialAdapter(channels=1))
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(mono_output)
    
    extent = mono_output.extent()
    duration_samples = extent.end - extent.start
    duration_seconds = duration_samples / SAMPLE_RATE
    print(f"Playing {duration_seconds:.2f} seconds of stereo→mono conversion...", flush=True)
    
    with renderer:
        renderer.start()
        renderer.play_extent()
    
    print("Done!\n", flush=True)


def demo_multiple_sources_panned():
    """Demo 6: Multiple sources panned to different positions."""
    print("=== Demo 6: Multiple Sources Panned to Different Positions ===")
    
    # Load multiple mono sources
    source1 = WavReaderPE(str(AUDIO_DIR / "acoustic_drums_mono44.wav"))
    source2 = WavReaderPE(str(AUDIO_DIR / "djembe_mono44.wav"))
    
    # Pan them to different positions
    panned1 = SpatialPE(source1, method=SpatialConstantPower(azimuth=-45.0))  # Left of center
    panned2 = SpatialPE(source2, method=SpatialConstantPower(azimuth=45.0))   # Right of center
    
    # Mix them together
    mixed_stream = MixPE(panned1, panned2)
    
    # Crop to shorter duration for demo
    extent1 = source1.extent()
    extent2 = source2.extent()
    min_duration = min(extent1.end - extent1.start, extent2.end - extent2.start)
    cropped_stream = CropPE(mixed_stream, Extent(0, min_duration))
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(cropped_stream)
    
    duration_seconds = min_duration / SAMPLE_RATE
    print(f"Playing {duration_seconds:.2f} seconds of two sources panned left and right...", flush=True)
    
    with renderer:
        renderer.start()
        renderer.play_extent()
    
    print("Done!\n", flush=True)


if __name__ == "__main__":
    print("=== pygmu2 Example 27: Spatial Audio ===\n", flush=True)
    
    # Run demos
    demo_channel_conversion()
    demo_linear_panning()
    demo_constant_power_panning()
    demo_dynamic_panning()
    demo_stereo_to_mono()
    demo_multiple_sources_panned()
    
    print("All demos complete!", flush=True)
