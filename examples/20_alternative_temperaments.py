"""
Alternative Temperaments Example

Demonstrates chord quality differences between musical tuning systems:
- 12-tone equal temperament (12-ET) - the default/standard tuning
- Just intonation (5-limit) - pure harmonic ratios (beatless)
- Pythagorean tuning - based on perfect fifths

Plays a C major triad (C-E-G) in each temperament to compare
consonance and beating characteristics.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from pygmu2 import (
import pygmu2 as pg
pg.set_sample_rate(44100)

    AudioRenderer,
    SinePE,
    MixPE,
    CropPE,
    Extent,
    pitch_to_freq,
    EqualTemperament,
    JustIntonation,
    PythagoreanTuning,
    set_temperament,
    set_reference_frequency,
    set_concert_pitch,
    set_verdi_tuning,
    get_reference_frequency,
)

# Sample rate for conversions
SAMPLE_RATE = 44100

# Chord parameters
CHORD_DURATION = 4.0  # seconds
PAUSE_BETWEEN = 1.0   # seconds


# ============================================================================
# Note Name <-> MIDI Conversion Utilities
# ============================================================================

def note_to_midi(note_name):
    """
    Convert a note name to MIDI note number.
    
    Args:
        note_name: Note name like "C4", "F#5", "Bb3"
    
    Returns:
        MIDI note number (int)
    
    Example:
        >>> note_to_midi("C4")
        60
        >>> note_to_midi("A4")
        69
        >>> note_to_midi("C#5")
        73
    """
    # Parse note name (preserve case for now)
    note_name = note_name.strip()
    
    # Note to semitone mapping (C = 0)
    note_map = {
        'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11
    }
    
    # Extract note letter
    note_letter = note_name[0].upper()
    if note_letter not in note_map:
        raise ValueError(f"Invalid note letter: {note_letter}")
    
    semitone = note_map[note_letter]
    
    # Check for sharp or flat
    idx = 1
    if len(note_name) > idx and note_name[idx] in '#♯':
        semitone += 1
        idx += 1
    elif len(note_name) > idx and note_name[idx] in 'bB♭':  # Accept both 'b' and 'B' for flat
        semitone -= 1
        idx += 1
    
    # Extract octave number
    try:
        octave = int(note_name[idx:])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid octave in note name: {note_name}")
    
    # MIDI note number: (octave + 1) * 12 + semitone
    # Middle C (C4) = 60
    midi_note = (octave + 1) * 12 + semitone
    
    return midi_note


def midi_to_note(midi_number):
    """
    Convert MIDI note number to note name.
    
    Args:
        midi_number: MIDI note number (0-127)
    
    Returns:
        Note name string (e.g., "C4", "F#5")
    
    Example:
        >>> midi_to_note(60)
        "C4"
        >>> midi_to_note(69)
        "A4"
    """
    if not 0 <= midi_number <= 127:
        raise ValueError(f"MIDI note number must be 0-127, got {midi_number}")
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    octave = (midi_number // 12) - 1
    semitone = midi_number % 12
    
    return f"{note_names[semitone]}{octave}"


def notes_to_midi(note_names):
    """
    Convert a list of note names to MIDI numbers.
    
    Args:
        note_names: List of note name strings or single string
    
    Returns:
        List of MIDI note numbers
    
    Example:
        >>> notes_to_midi(["C4", "E4", "G4"])
        [60, 64, 67]
        >>> notes_to_midi("A4")
        [69]
    """
    if isinstance(note_names, str):
        note_names = [note_names]
    
    return [note_to_midi(name) for name in note_names]

def play_chord(notes, temperament_name, temperament, start_time, duration=CHORD_DURATION):
    """
    Create a chord at a specific time using a given temperament.
    
    Args:
        notes: List of MIDI note numbers (e.g., [60, 64, 67] for C major)
        temperament_name: Name of the temperament (for display)
        temperament: The Temperament object to use
        start_time: When to start the chord (in seconds)
        duration: How long the chord should last (in seconds)
    
    Returns:
        ProcessingElement containing the chord
    """
    # Create sine waves for each note
    sines = []
    for note in notes:
        freq = pitch_to_freq(note, temperament=temperament)
        sine = SinePE(frequency=freq, amplitude=0.2 / len(notes))  # Scale by number of notes
        sines.append(sine)
    
    # Mix all notes together
    chord = MixPE(*sines)
    
    # Convert time to samples and crop
    start_sample = int(start_time * SAMPLE_RATE)
    duration_samples = int(duration * SAMPLE_RATE)
    
    return CropPE(chord, Extent(start_sample, start_sample + duration_samples))


def print_chord_info(chord_name, notes, temperaments):
    """Print frequency information for a chord in different temperaments."""
    print(f"\n=== {chord_name} Frequencies (Hz) ===\n")
    header = "Note    " + "".join(f"{name:12}" for name, _ in temperaments)
    print(header)
    print("-" * len(header))
    
    for note in notes:
        note_name = midi_to_note(note)
        freqs = []
        for _, temp in temperaments:
            freq = pitch_to_freq(note, temperament=temp).item()
            freqs.append(f"{freq:8.2f}")
        print(f"{note_name:6}  " + "    ".join(freqs))


# Define temperaments globally
TEMPERAMENTS = [
    ("12-ET", EqualTemperament(12)),
    ("Just Intonation", JustIntonation()),
    ("Pythagorean", PythagoreanTuning()),
]


def play_chord_comparison(chord_name, notes):
    """
    Play a specific chord in all temperaments.
    
    Args:
        chord_name: Name of the chord (e.g., "C Major")
        notes: List of MIDI note numbers
    """
    print(f"\n{'='*60}")
    print(f"{chord_name}: {notes}")
    print(f"{'='*60}")
    
    all_chords = []
    current_time = 0.0
    chord_num = 1
    
    for temp_name, temp in TEMPERAMENTS:
        print(f"\n{chord_num}. {temp_name} (t={current_time:.1f}s)")
        
        if temp_name == "12-ET":
            print("   - Standard Western tuning")
            print("   - Slightly beating, good overall consonance")
        elif temp_name == "Just Intonation":
            print("   - Pure harmonic ratios: Major third = 5/4, Perfect fifth = 3/2")
            print("   - Beatless, maximum consonance")
        elif temp_name == "Pythagorean":
            print("   - Based on pure perfect fifths (3/2)")
            print("   - Major third is sharper (81/64), more beating")
        
        # Create chord
        chord_pe = play_chord(notes, temp_name, temp, current_time)
        all_chords.append(chord_pe)
        
        current_time += CHORD_DURATION + PAUSE_BETWEEN
        chord_num += 1
    
    # Print frequency comparison
    print_chord_info(chord_name, notes, TEMPERAMENTS)
    
    # Mix all chords and play
    mixed = MixPE(*all_chords)
    
    print(f"\n{'='*60}")
    print(f"Playing {current_time - PAUSE_BETWEEN:.1f} seconds of audio...")
    print("Listen for the differences in consonance and beating.")
    print(f"{'='*60}\n")
    
    renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
    renderer.set_source(mixed)
    renderer.start()
    renderer.play_extent()
    renderer.stop()
    
    print("\n✅ Playback complete!\n")


def demo_c_major():
    """Compare C Major triad across temperaments."""
    play_chord_comparison("C Major", notes_to_midi(["C4", "E4", "G4"]))


def demo_a_minor():
    """Compare A Minor triad across temperaments."""
    play_chord_comparison("A Minor", notes_to_midi(["A4", "C5", "E4"]))

def demo_fifth():
    """Compare stacked fifths (C-G-D) across temperaments."""
    play_chord_comparison("5th Stack", notes_to_midi(["C4", "G4"]))

def demo_fifth_stack():
    """Compare stacked fifths (C-G-D-A) across temperaments."""
    play_chord_comparison("5th Stack", notes_to_midi(["C3", "G3", "D4", "A4"]))


def demo_reference_frequency():
    """
    Compare A=440 Hz (concert pitch) vs A=432 Hz (Verdi tuning).
    
    Demonstrates changing the global reference frequency.
    """
    print(f"\n{'='*60}")
    print("Reference Frequency Comparison: A=440 vs A=432")
    print(f"{'='*60}")
    print("\nSame C major chord, different reference frequencies.\n")
    
    # Save original
    original_ref = get_reference_frequency()
    
    try:
        notes = notes_to_midi(["C4", "E4", "G4"])
        et12 = EqualTemperament(12)
        
        # A=440 Hz (concert pitch)
        set_concert_pitch()
        freq_440, _ = get_reference_frequency()
        
        print(f"1. Concert Pitch (A4 = {freq_440:.0f} Hz) at t=0.0s")
        print("   - Modern standard (ISO 16)")
        print("   Frequencies:")
        for note in notes:
            note_name = midi_to_note(note)
            freq = pitch_to_freq(note, temperament=et12).item()
            print(f"     {note_name}: {freq:.2f} Hz")
        
        chord_440 = play_chord(notes, "A=440", et12, 0.0)
        
        # A=432 Hz (Verdi tuning)
        set_verdi_tuning()
        freq_432, _ = get_reference_frequency()
        
        start_time_432 = CHORD_DURATION + PAUSE_BETWEEN
        print(f"\n2. Verdi Tuning (A4 = {freq_432:.0f} Hz) at t={start_time_432:.1f}s")
        print("   - Alternative 'philosophical pitch'")
        print("   Frequencies:")
        for note in notes:
            note_name = midi_to_note(note)
            freq = pitch_to_freq(note, temperament=et12).item()
            print(f"     {note_name}: {freq:.2f} Hz")
        
        chord_432 = play_chord(notes, "A=432", et12, start_time_432)
        
        # Calculate difference
        c4_diff = 261.63 - pitch_to_freq(60, temperament=et12).item()
        print(f"\n   All notes are ~{c4_diff:.2f} Hz lower at A=432")
        
        # Mix and play
        mixed = MixPE(chord_440, chord_432)
        
        total_duration = start_time_432 + CHORD_DURATION
        print(f"\nPlaying {total_duration:.1f} seconds...")
        print("Listen for the subtle pitch difference.\n")
        
        renderer = AudioRenderer(sample_rate=SAMPLE_RATE)
        renderer.set_source(mixed)
        renderer.start()
        renderer.play_extent()
        renderer.stop()
        
        print("\n✅ Playback complete!\n")
    
    finally:
        # Restore original reference
        set_reference_frequency(*original_ref)


def main():
    """Main menu for selecting temperament demos."""
    print("\npygmu2 Alternative Temperaments Demo")
    print("=" * 60)
    print()
    print("Compare chord quality across different tuning systems:")
    print("  - 12-Tone Equal Temperament (standard)")
    print("  - Just Intonation (pure ratios)")
    print("  - Pythagorean Tuning (pure fifths)")
    print()
    
    demos = [
        ("1", "C Major Triad (temperaments)", demo_c_major),
        ("2", "A Minor Triad (temperaments)", demo_a_minor),
        ("3", "5th (C-G) (temperaments)", demo_fifth),
        ("4", "5th Stack (C-G-D-A) (temperaments)", demo_fifth_stack),
        ("5", "A=440 vs A=432 (reference freq)", demo_reference_frequency),
        ("a", "All demos", None),
    ]
    
    print("Available demos:")
    for key, name, _ in demos:
        print(f"  {key}: {name}")
    print()
    
    choice = input("Choose a demo (1-5 or 'a' for all): ").strip().lower()
    print()
    
    if choice == "a":
        for key, name, func in demos[:-1]:
            func()
    else:
        for key, name, func in demos:
            if key == choice and func:
                func()
                break
        else:
            print(f"Unknown choice: {choice}")
    
    print("\nTips:")
    print("  Change temperament:")
    print("    >>> from pygmu2 import set_temperament, JustIntonation")
    print("    >>> set_temperament(JustIntonation())")
    print("  Change reference frequency:")
    print("    >>> from pygmu2 import set_verdi_tuning")
    print("    >>> set_verdi_tuning()  # A4 = 432 Hz")


if __name__ == "__main__":
    main()
