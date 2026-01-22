"""
Play a WAV file through the AudioRenderer.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from pygmu2 import AudioRenderer, WavReaderPE

# Path to the WAV file
WAV_PATH = r"C:\Users\r\Projects\pygmu\samples\sfx\s07.wav"

def main():
    print(f"Loading: {WAV_PATH}")
    
    # Read the WAV file
    reader = WavReaderPE(WAV_PATH)
    
    # Get file info
    extent = reader.extent()
    channels = reader.channel_count()
    
    print(f"Duration: {extent.end} samples")
    print(f"Channels: {channels}")
    
    # Create audio renderer (use file's sample rate if available)
    sample_rate = reader.file_sample_rate or 44100
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {extent.end / sample_rate:.2f} seconds")
    
    print("\nPlaying...")
    
    with AudioRenderer(sample_rate=sample_rate) as renderer:
        renderer.set_source(reader)
        renderer.start()
        renderer.play_extent()
    
    print("Done!")


if __name__ == "__main__":
    main()
