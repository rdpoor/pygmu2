"""
Play a WAV file with effects through the AudioRenderer.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from pygmu2 import (
    AudioRenderer,
    WavReaderPE,
    GainPE,
    DelayPE,
    MixPE,
    RampPE,
    CropPE,
    Extent,
)

# Path to the WAV file
WAV_PATH = r"C:\Users\r\Projects\pygmu\samples\sfx\s07.wav"


def play_original():
    """Play the original file."""
    print("=== Original ===")
    reader = WavReaderPE(WAV_PATH)
    sample_rate = reader.file_sample_rate or 44100
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(reader)
        renderer.start()
        renderer.play_extent()
    print()


def play_with_echo():
    """Play with a simple echo effect."""
    print("=== With Echo ===")
    reader = WavReaderPE(WAV_PATH)
    sample_rate = reader.file_sample_rate or 44100
    
    # Create echo: original + delayed copy at 50% volume
    delay_samples = int(0.15 * sample_rate)  # 150ms delay
    delayed = DelayPE(reader, delay=delay_samples)
    echo = GainPE(delayed, gain=0.4)
    
    # Mix original with echo
    mixed = MixPE(reader, echo)
    
    # Extend the crop to include the echo tail
    original_extent = reader.extent()
    extended = CropPE(mixed, Extent(0, original_extent.end + delay_samples))
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(extended)
        renderer.start()
        renderer.play_extent()
    print()


def play_with_fade():
    """Play with fade in and fade out."""
    print("=== With Fade In/Out ===")
    reader = WavReaderPE(WAV_PATH)
    sample_rate = reader.file_sample_rate or 44100
    extent = reader.extent()
    duration = extent.end
    
    # Create fade envelope: ramp up for first 20%, hold, ramp down for last 20%
    fade_duration = int(duration * 0.2)
    
    # Fade in
    fade_in = RampPE(0.0, 1.0, duration=fade_duration)
    fade_in_cropped = CropPE(fade_in, Extent(0, fade_duration))
    
    # Fade out (starts at duration - fade_duration)
    fade_out_start = duration - fade_duration
    fade_out = RampPE(1.0, 0.0, duration=fade_duration)
    fade_out_delayed = DelayPE(fade_out, delay=fade_out_start)
    
    # Hold at 1.0 in the middle
    from pygmu2 import ConstantPE
    hold = ConstantPE(1.0)
    hold_cropped = CropPE(hold, Extent(fade_duration, fade_out_start))
    
    # Combine envelope parts
    envelope = MixPE(
        CropPE(fade_in_cropped, Extent(0, duration)),
        CropPE(hold_cropped, Extent(0, duration)),
        CropPE(fade_out_delayed, Extent(0, duration)),
    )
    
    # Apply envelope to audio
    faded = GainPE(reader, gain=envelope)
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(faded)
        renderer.start()
        renderer.play_extent()
    print()


def play_quieter():
    """Play at reduced volume."""
    print("=== Quieter (50% volume) ===")
    reader = WavReaderPE(WAV_PATH)
    sample_rate = reader.file_sample_rate or 44100
    
    quiet = GainPE(reader, gain=0.5)
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(quiet)
        renderer.start()
        renderer.play_extent()
    print()


def main():
    import time
    
    print(f"Playing: {WAV_PATH}\n")
    
    play_original()
    time.sleep(0.3)
    
    play_quieter()
    time.sleep(0.3)
    
    play_with_echo()
    time.sleep(0.3)
    
    play_with_fade()
    
    print("All done!")


if __name__ == "__main__":
    main()
