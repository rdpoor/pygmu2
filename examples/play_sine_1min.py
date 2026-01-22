"""
Play a continuous sine tone for 1 minute to verify long playback works.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import time
from pygmu2 import AudioRenderer, SinePE, CropPE, Extent

SAMPLE_RATE = 44100
DURATION_SECONDS = 60
FREQUENCY = 440.0
AMPLITUDE = 0.2

def main():
    duration_samples = SAMPLE_RATE * DURATION_SECONDS
    
    print(f"Playing {FREQUENCY} Hz sine wave")
    print(f"Amplitude: {AMPLITUDE}")
    print(f"Duration: {DURATION_SECONDS} seconds ({duration_samples:,} samples)")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print()
    print("Press Ctrl+C to stop early...")
    print()
    
    # Create sine wave
    sine = SinePE(frequency=FREQUENCY, amplitude=AMPLITUDE)
    cropped = CropPE(sine, Extent(0, duration_samples))
    
    start_time = time.time()
    
    try:
        with AudioRenderer(sample_rate=SAMPLE_RATE) as renderer:
            renderer.set_source(cropped)
            renderer.start()
            renderer.play_extent()
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\nStopped after {elapsed:.1f} seconds")
        return
    
    elapsed = time.time() - start_time
    print(f"Done! Played for {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
