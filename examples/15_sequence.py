#!/usr/bin/env python3
"""
Example 15: SequencePE - Sequencing audio in time

Demonstrates SequencePE for arranging multiple audio sources in time,
both with and without overlap.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from pathlib import Path
from pygmu2 import (
    AudioRenderer,
    SequencePE,
    WavReaderPE,
    GainPE,
    CropPE,
    Extent,
    seconds_to_samples,
    BlitSawPE,
    pitch_to_freq,
    MixPE,
)

# Path to audio files
AUDIO_DIR = Path(__file__).parent / "audio"

SAMPLE_RATE = 44100


def demo_drum_pattern():
    """
    Demonstrate non-overlapping sequence: a drum pattern.
    
    Sequences kick, djembe, and drums in a rhythmic pattern
    where each sound plays one after another (no overlap).
    """
    print("=== Demo: Drum Pattern Sequencer (Non-overlapping) ===")
    print("Sequencing kick, djembe, and drums in a rhythmic pattern...")
    print()
    
    # Load drum samples
    kick = WavReaderPE(str(AUDIO_DIR / "kick.wav"))
    djembe = WavReaderPE(str(AUDIO_DIR / "djembe.wav"))
    drums = WavReaderPE(str(AUDIO_DIR / "acoustic_drums.wav"))
    
    sample_rate = kick.file_sample_rate
    
    # Get durations of each sample
    kick_duration = kick.extent().duration or 0
    djembe_duration = djembe.extent().duration or 0
    drums_duration = drums.extent().duration or 0
    
    # Create a pattern: kick, djembe, drums, kick, djembe, drums
    # Each sound plays for its full duration, then the next starts
    beat_duration = int(sample_rate * 0.5)  # 0.5 seconds per beat
    
    sequence = [
        (kick, 0),                              # Kick at 0s
        (djembe, kick_duration),               # Djembe after kick
        (drums, kick_duration + djembe_duration),  # Drums after djembe
        (kick, kick_duration + djembe_duration + drums_duration),  # Kick again
        (djembe, kick_duration + djembe_duration + drums_duration + kick_duration),
        (drums, kick_duration + djembe_duration + drums_duration + kick_duration + djembe_duration),
    ]
    
    # Create sequence with overlap=False (each sound plays fully, then next starts)
    seq = SequencePE(sequence, overlap=False)
    
    # Apply gain to avoid clipping
    output = GainPE(seq, gain=0.7)
    
    # Calculate total duration (all samples + a bit of space)
    total_duration = (
        kick_duration + djembe_duration + drums_duration +
        kick_duration + djembe_duration + drums_duration +
        int(sample_rate * 0.5)  # Extra space at end
    )
    output = CropPE(output, Extent(0, total_duration))
    
    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(output)
    
    with renderer:
        renderer.start()
        renderer.play_extent()


def demo_layered_entry():
    """
    Demonstrate overlapping sequence: instruments enter one by one.
    
    Multiple instruments start at different times and overlap,
    creating a layered, building texture.
    """
    print("\n=== Demo: Layered Entry (Overlapping) ===")
    print("Instruments enter one by one, building a layered texture...")
    print()
    
    # Load audio sources
    bass = WavReaderPE(str(AUDIO_DIR / "bass.wav"))
    choir = WavReaderPE(str(AUDIO_DIR / "choir.wav"))
    drums = WavReaderPE(str(AUDIO_DIR / "acoustic_drums.wav"))
    
    sample_rate = bass.file_sample_rate
    
    # Create sequence where instruments enter at different times
    # All will overlap and play together after they've all started
    entry_times = [
        (bass, 0),                           # Bass starts immediately
        (choir, int(sample_rate * 2.0)),     # Choir enters at 2 seconds
        (drums, int(sample_rate * 4.0)),     # Drums enter at 4 seconds
    ]
    
    # Create sequence with overlap=True (all sounds mix together)
    seq = SequencePE(entry_times, overlap=True)
    
    # Apply gain to avoid clipping (multiple sources mixing)
    output = GainPE(seq, gain=0.5)
    
    # Play for about 8 seconds total
    total_duration = int(sample_rate * 8.0)
    output = CropPE(output, Extent(0, total_duration))
    
    renderer = AudioRenderer(sample_rate=sample_rate)
    renderer.set_source(output)
    
    with renderer:
        renderer.start()
        renderer.play_range(0, total_duration)


def demo_arpeggio():
    """
    Demonstrate overlapping sequence: a musical arpeggio.
    
    Sequences individual notes of a C major chord to create
    an ascending arpeggio pattern.
    """
    print("\n=== Demo: Musical Arpeggio (Overlapping) ===")
    print("Playing C-E-G-C arpeggio...")
    print()
    
    # MIDI note numbers for C major arpeggio (C4, E4, G4, C5)
    C4 = 60
    E4 = 64
    G4 = 67
    C5 = 72
    
    note_duration = int(SAMPLE_RATE * 0.4)  # 0.4 seconds per note
    
    # Create sequence of notes, cropping each to finite duration
    # This ensures each note plays for exactly note_duration samples
    c_note = CropPE(BlitSawPE(frequency=pitch_to_freq(C4), amplitude=0.3), Extent(0, None))
    e_note = CropPE(BlitSawPE(frequency=pitch_to_freq(E4), amplitude=0.3), Extent(0, None))
    g_note = CropPE(BlitSawPE(frequency=pitch_to_freq(G4), amplitude=0.3), Extent(0, None))
    c5_note = CropPE(BlitSawPE(frequency=pitch_to_freq(C5), amplitude=0.3), Extent(0, None))
    
    sequence = [
        (c_note, 0),
        (e_note, note_duration),
        (g_note, note_duration * 2),
        (c5_note, note_duration * 3),
    ]
    
    # Overlapping: each note continues when next note starts
    seq = SequencePE(sequence, overlap=True)
    
    # Apply gain
    output = GainPE(seq, gain=0.4)
    
    # Play for duration of all notes
    total_duration = note_duration * 4
    output = CropPE(output, Extent(0, total_duration))
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(output)
    
    with renderer:
        renderer.start()
        renderer.play_extent()


def demo_chord_progression():
    """
    Demonstrate non-overlapping sequence    
    Sequences non-overlapping chords.
    """
    print("\n=== Demo: Chord Progression (Non-overlapping) ===")
    print("C major -> F major -> G major")
    print()
    
    # Create chords (each chord is a mix of 3 notes)
    # C major: C4, E4, G4
    c_chord = MixPE(
        BlitSawPE(frequency=pitch_to_freq(60), amplitude=0.2),  # C4
        BlitSawPE(frequency=pitch_to_freq(64), amplitude=0.2),  # E4
        BlitSawPE(frequency=pitch_to_freq(67), amplitude=0.2),  # G4
    )
    
    # F major: F4, A4, C5
    f_chord = MixPE(
        BlitSawPE(frequency=pitch_to_freq(65), amplitude=0.2),  # F4
        BlitSawPE(frequency=pitch_to_freq(69), amplitude=0.2),  # A4
        BlitSawPE(frequency=pitch_to_freq(72), amplitude=0.2),  # C5
    )
    
    # G major: G4, B4, D5
    g_chord = MixPE(
        BlitSawPE(frequency=pitch_to_freq(67), amplitude=0.2),  # G4
        BlitSawPE(frequency=pitch_to_freq(71), amplitude=0.2),  # B4
        BlitSawPE(frequency=pitch_to_freq(74), amplitude=0.2),  # D5
    )
    
    chord_duration = int(SAMPLE_RATE * 2.0)  # 2 seconds per chord
    
    # Crop each chord to finite duration (starting at time 0)
    # This ensures each chord plays for exactly chord_duration samples
    c_chord_cropped = CropPE(c_chord, Extent(0, None))
    f_chord_cropped = CropPE(f_chord, Extent(0, None))
    g_chord_cropped = CropPE(g_chord, Extent(0, None))
    
    # Verify extents are finite (for debugging)
    print(f"  C chord extent: {c_chord_cropped.extent()}")
    print(f"  F chord extent: {f_chord_cropped.extent()}")
    print(f"  G chord extent: {g_chord_cropped.extent()}")
    
    sequence = [
        (c_chord_cropped, 0),
        (f_chord_cropped, chord_duration),
        (g_chord_cropped, chord_duration * 2),
    ]
    
    # Overlapping: chords blend together smoothly
    seq = SequencePE(sequence, overlap=False)
    
    # Apply gain
    output = GainPE(seq, gain=0.4)
    
    # Play for duration of all chords
    total_duration = chord_duration * 3
    output = CropPE(output, Extent(0, total_duration))
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(output)
    
    with renderer:
        renderer.start()
        renderer.play_extent()


if __name__ == "__main__":
    import sys
    
    demos = {
        "1": ("Drum Pattern", demo_drum_pattern),
        "2": ("Layered Entry", demo_layered_entry),
        "3": ("Arpeggio", demo_arpeggio),
        "4": ("Chord Progression", demo_chord_progression),
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("=== pygmu2 Example 15: SequencePE ===")
        print()
        print("Choose a demo:")
        for key, (name, _) in demos.items():
            print(f"  {key}: {name}")
        print("  a: All demos")
        print()
        choice = input("Choice (1-4 or 'a' for all): ").strip().lower()
    
    if choice == "a":
        for name, func in demos.values():
            try:
                func()
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
    elif choice in demos:
        name, func = demos[choice]
        try:
            func()
        except KeyboardInterrupt:
            print("\nInterrupted.")
    else:
        print(f"Invalid choice: {choice}")
