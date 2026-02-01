"""
Example 27: Spatial Audio - Panning and Channel Conversion

Demonstrates SpatialPE for channel conversion and spatialization (panning)
using various techniques: SpatialAdapter, SpatialLinear, SpatialConstantPower,
and HRTF binaural spatialization (KEMAR) with ConvolvePE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from pathlib import Path

import soundfile as sf

from pygmu2 import (
    ArrayPE,
    AudioRenderer,
    ConvolvePE,
    CropPE,
    DelayPE,
    Extent,
    GainPE,
    MixPE,
    PiecewisePE,
    SpatialPE,
    SpatialAdapter,
    SpatialConstantPower,
    SpatialHRTF,
    SpatialLinear,
    WavReaderPE,
    seconds_to_samples,
)

# Path to audio files (relative to this script)
AUDIO_DIR = Path(__file__).parent / "audio"
KEMAR_DIR = AUDIO_DIR / "kemar"
SAMPLE_RATE = 44100
DJEMBE_HIT_PATH = AUDIO_DIR / "djembe_hit.wav"

def _load_hrtf_ir_pe(filename: str, swap_lr: bool = False) -> ArrayPE:
    """Load a KEMAR HRTF WAV as an ArrayPE filter. If swap_lr, swap L/R for left-side rendering."""
    path = KEMAR_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"HRTF file not found: {path}")
    data, _ = sf.read(path, dtype="float32", always_2d=True)
    if swap_lr and data.shape[1] >= 2:
        data = data[:, [1, 0]]
    return ArrayPE(data)


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
    pan_control = PiecewisePE([(0, -90.0), (duration_samples, 90.0)])
    
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


def demo_hrtf_spatialization():
    """Demo 7: HRTF binaural spatialization of djembe hit at several positions."""
    print("=== Demo 7: HRTF Binaural Spatialization (djembe hit) ===")

    if not DJEMBE_HIT_PATH.exists():
        print(f"  Skipping: {DJEMBE_HIT_PATH} not found.", flush=True)
        return
    if not KEMAR_DIR.exists():
        print(f"  Skipping: KEMAR directory {KEMAR_DIR} not found.", flush=True)
        return

    # Mono source (djembe hit)
    mono_source = WavReaderPE(str(DJEMBE_HIT_PATH))
    # Ensure mono for convolution (ConvolvePE mono + stereo filter -> stereo out)
    if mono_source.channel_count() != 1:
        mono_source = SpatialPE(mono_source, method=SpatialAdapter(channels=1))

    # Positions: 7 azimuths [-90 to +90] × 3 elevations [0, 10, 20] (by elevation, then azimuth)
    azimuths = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
    elevations = [0.0, 20.0, 40.0]
    positions = [(az, el) for el in elevations for az in azimuths]

    panned_streams = []
    delay_samples = 0
    gap = seconds_to_samples(0.4, SAMPLE_RATE)

    for az, el in positions:
        filename = SpatialHRTF.hrtf_filename_for(az, el)
        swap_lr = az < 0
        ir_pe = _load_hrtf_ir_pe(filename, swap_lr=swap_lr)
        hrtf_out = ConvolvePE(mono_source, ir_pe)
        panned_streams.append(DelayPE(hrtf_out, delay=delay_samples))
        extent = mono_source.extent()
        delay_samples += (extent.end - extent.start) + gap

    mixed_stream = MixPE(*panned_streams)

    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(mixed_stream)

    extent = mixed_stream.extent()
    duration_seconds = (extent.end - extent.start) / SAMPLE_RATE
    print(
        f"Playing {duration_seconds:.2f} seconds: 7 azimuths (-90° to +90°) × 3 elevations (0°, 20°, 40°) — {len(positions)} positions (HRTF)...",
        flush=True,
    )

    with renderer:
        renderer.start()
        renderer.play_extent()

    print("Done!\n", flush=True)


def demo_all():
    """Run all spatial demos in order."""
    demo_channel_conversion()
    demo_linear_panning()
    demo_constant_power_panning()
    demo_dynamic_panning()
    demo_stereo_to_mono()
    demo_multiple_sources_panned()
    demo_hrtf_spatialization()
    print("All demos complete!", flush=True)


if __name__ == "__main__":
    import sys

    demos = [
        ("1", "Channel conversion (Mono → Stereo)", demo_channel_conversion),
        ("2", "Linear panning (Left → Center → Right)", demo_linear_panning),
        ("3", "Constant-power panning (Left → Center → Right)", demo_constant_power_panning),
        ("4", "Dynamic panning (sweep Left → Right)", demo_dynamic_panning),
        ("5", "Stereo to mono conversion", demo_stereo_to_mono),
        ("6", "Multiple sources panned (left and right)", demo_multiple_sources_panned),
        ("7", "HRTF binaural spatialization (djembe hit)", demo_hrtf_spatialization),
        ("a", "All demos", demo_all),
    ]

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print("Example 27: Spatial Audio - Panning and Channel Conversion")
        print("-----------------------------------------------------------")
        for key, name, _ in demos:
            print(f"  {key}: {name}")
        print()
        choice = input("Choice (1-7 or 'a'): ").strip().lower()

    for key, _name, fn in demos:
        if key == choice:
            fn()
            break
    else:
        print("Invalid choice.")
