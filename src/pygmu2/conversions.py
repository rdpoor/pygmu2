"""
Audio conversion utility functions.

All functions are vectorized and work with numpy arrays or scalars.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
from numpy.typing import ArrayLike


def pitch_to_freq(pitch: ArrayLike) -> np.ndarray:
    """
    Convert MIDI note number to frequency in Hz.
    
    Uses the standard formula: freq = 440 * 2^((pitch - 69) / 12)
    
    Args:
        pitch: MIDI note number(s). Can be fractional.
               A4 (440 Hz) = 69, Middle C (C4) = 60
    
    Returns:
        Frequency in Hz
    
    Example:
        >>> pitch_to_freq(69)  # A4
        440.0
        >>> pitch_to_freq(60)  # Middle C
        261.6255653...
        >>> pitch_to_freq([60, 64, 67])  # C major chord
        array([261.626, 329.628, 391.995])
    """
    pitch = np.asarray(pitch, dtype=np.float64)
    return 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))


def freq_to_pitch(freq: ArrayLike) -> np.ndarray:
    """
    Convert frequency in Hz to MIDI note number.
    
    Uses the standard formula: pitch = 69 + 12 * log2(freq / 440)
    
    Args:
        freq: Frequency in Hz. Must be positive.
    
    Returns:
        MIDI note number (can be fractional for microtones)
    
    Example:
        >>> freq_to_pitch(440.0)  # A4
        69.0
        >>> freq_to_pitch(261.6256)  # Middle C
        60.0
        >>> freq_to_pitch([261.626, 329.628, 391.995])  # C major chord
        array([60., 64., 67.])
    """
    freq = np.asarray(freq, dtype=np.float64)
    # Protect against log of zero/negative
    freq = np.maximum(freq, 1e-10)
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def ratio_to_db(ratio: ArrayLike) -> np.ndarray:
    """
    Convert linear amplitude ratio to decibels.
    
    Uses the formula: dB = 20 * log10(ratio)
    
    Args:
        ratio: Linear amplitude ratio. Must be positive.
               1.0 = 0 dB, 2.0 ≈ 6.02 dB, 0.5 ≈ -6.02 dB
    
    Returns:
        Value in decibels
    
    Example:
        >>> ratio_to_db(1.0)
        0.0
        >>> ratio_to_db(2.0)
        6.0206...
        >>> ratio_to_db(0.5)
        -6.0206...
        >>> ratio_to_db(10.0)
        20.0
    """
    ratio = np.asarray(ratio, dtype=np.float64)
    # Protect against log of zero/negative, floor at -200 dB
    ratio = np.maximum(ratio, 1e-10)
    return 20.0 * np.log10(ratio)


def db_to_ratio(db: ArrayLike) -> np.ndarray:
    """
    Convert decibels to linear amplitude ratio.
    
    Uses the formula: ratio = 10^(dB / 20)
    
    Args:
        db: Value in decibels
            0 dB = 1.0, 6 dB ≈ 2.0, -6 dB ≈ 0.5
    
    Returns:
        Linear amplitude ratio
    
    Example:
        >>> db_to_ratio(0.0)
        1.0
        >>> db_to_ratio(6.0)
        1.9952...
        >>> db_to_ratio(-6.0)
        0.5011...
        >>> db_to_ratio(20.0)
        10.0
    """
    db = np.asarray(db, dtype=np.float64)
    return 10.0 ** (db / 20.0)


def semitones_to_ratio(semitones: ArrayLike) -> np.ndarray:
    """
    Convert semitones to frequency ratio.
    
    Uses the formula: ratio = 2^(semitones / 12)
    
    Args:
        semitones: Interval in semitones.
                   12 semitones = octave (ratio 2.0)
    
    Returns:
        Frequency ratio
    
    Example:
        >>> semitones_to_ratio(12)  # Octave
        2.0
        >>> semitones_to_ratio(7)   # Perfect fifth
        1.4983...
        >>> semitones_to_ratio(-12) # Octave down
        0.5
    """
    semitones = np.asarray(semitones, dtype=np.float64)
    return 2.0 ** (semitones / 12.0)


def ratio_to_semitones(ratio: ArrayLike) -> np.ndarray:
    """
    Convert frequency ratio to semitones.
    
    Uses the formula: semitones = 12 * log2(ratio)
    
    Args:
        ratio: Frequency ratio. Must be positive.
               2.0 = octave (12 semitones)
    
    Returns:
        Interval in semitones
    
    Example:
        >>> ratio_to_semitones(2.0)  # Octave
        12.0
        >>> ratio_to_semitones(1.5)  # ~Perfect fifth
        7.0195...
        >>> ratio_to_semitones(0.5)  # Octave down
        -12.0
    """
    ratio = np.asarray(ratio, dtype=np.float64)
    ratio = np.maximum(ratio, 1e-10)
    return 12.0 * np.log2(ratio)


def samples_to_seconds(samples: ArrayLike, sample_rate: float) -> np.ndarray:
    """
    Convert sample count to seconds.
    
    Args:
        samples: Number of samples
        sample_rate: Sample rate in Hz
    
    Returns:
        Duration in seconds
    
    Example:
        >>> samples_to_seconds(44100, 44100)
        1.0
        >>> samples_to_seconds(22050, 44100)
        0.5
    """
    samples = np.asarray(samples, dtype=np.float64)
    return samples / sample_rate


def seconds_to_samples(seconds: ArrayLike, sample_rate: float) -> np.ndarray:
    """
    Convert seconds to sample count.
    
    Args:
        seconds: Duration in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        Number of samples (float, caller may want to round)
    
    Example:
        >>> seconds_to_samples(1.0, 44100)
        44100.0
        >>> seconds_to_samples(0.5, 44100)
        22050.0
    """
    seconds = np.asarray(seconds, dtype=np.float64)
    return seconds * sample_rate
