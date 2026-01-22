"""
Play a continuous sine tone for 10 seconds to test for audio glitches.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from pygmu2 import AudioRenderer, SinePE, CropPE, Extent

SAMPLE_RATE = 44100
DURATION_SECONDS = 10
FREQUENCY = 440.0
AMPLITUDE = 0.2

def main():
    duration_samples = SAMPLE_RATE * DURATION_SECONDS
    
    print(f"Playing {FREQUENCY} Hz sine wave")
    print(f"Amplitude: {AMPLITUDE}")
    print(f"Duration: {DURATION_SECONDS} seconds ({duration_samples} samples)")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print()
    print("Listen for any clicks, pops, or dropouts...")
    print()
    
    # Create sine wave
    sine = SinePE(frequency=FREQUENCY, amplitude=AMPLITUDE)
    cropped = CropPE(sine, Extent(0, duration_samples))
    
    with AudioRenderer(sample_rate=SAMPLE_RATE) as renderer:
        renderer.set_source(cropped)
        renderer.start()
        renderer.play_extent()
    
    print("Done!")


if __name__ == "__main__":
    main()
