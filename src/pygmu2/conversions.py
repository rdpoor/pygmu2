"""
Audio conversion utility functions.

All functions are vectorized and work with numpy arrays or scalars.

These functions now support alternative temperaments via the temperament
parameter. If not specified, the global default temperament is used
(12-tone equal temperament by default).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np
from numpy.typing import ArrayLike

from pygmu2.temperament import Temperament, get_temperament, get_reference_frequency


def pitch_to_freq(
    pitch: ArrayLike,
    temperament: Temperament | None = None,
    reference_pitch: float | None = None,
    reference_freq: float | None = None
) -> np.ndarray:
    """
    Convert pitch number to frequency in Hz.
    
    By default uses 12-tone equal temperament with A4 = 440 Hz.
    Alternative temperaments can be specified via the temperament parameter
    or by setting a global default with set_temperament().
    Reference frequency can be changed globally with set_reference_frequency().
    
    Args:
        pitch: Pitch number(s). Can be fractional.
               In 12-ET: A4 = 69, Middle C (C4) = 60
        temperament: Temperament to use (default: uses global temperament)
        reference_pitch: Reference pitch number (default: global, typically 69.0 for A4)
        reference_freq: Reference frequency in Hz (default: global, typically 440.0)
    
    Returns:
        Frequency in Hz
    
    Example:
        >>> pitch_to_freq(69)  # A4 in 12-ET
        440.0
        >>> pitch_to_freq(60)  # Middle C in 12-ET
        261.6255653...
        >>> pitch_to_freq([60, 64, 67])  # C major chord in 12-ET
        array([261.626, 329.628, 391.995])
        
        >>> # Using alternative tuning (A4 = 432 Hz)
        >>> from pygmu2 import set_reference_frequency
        >>> set_reference_frequency(432.0)
        >>> pitch_to_freq(69)  # A4 now = 432 Hz
        432.0
    """
    temp = temperament if temperament is not None else get_temperament()
    
    # Get global reference if not specified
    if reference_freq is None or reference_pitch is None:
        global_freq, global_pitch = get_reference_frequency()
        reference_freq = reference_freq if reference_freq is not None else global_freq
        reference_pitch = reference_pitch if reference_pitch is not None else global_pitch
    
    return temp.pitch_to_freq(pitch, reference_pitch, reference_freq)


def freq_to_pitch(
    freq: ArrayLike,
    temperament: Temperament | None = None,
    reference_pitch: float | None = None,
    reference_freq: float | None = None
) -> np.ndarray:
    """
    Convert frequency in Hz to pitch number.
    
    By default uses 12-tone equal temperament with A4 = 440 Hz.
    Alternative temperaments can be specified via the temperament parameter
    or by setting a global default with set_temperament().
    Reference frequency can be changed globally with set_reference_frequency().
    
    Args:
        freq: Frequency in Hz. Must be positive.
        temperament: Temperament to use (default: uses global temperament)
        reference_pitch: Reference pitch number (default: global, typically 69.0 for A4)
        reference_freq: Reference frequency in Hz (default: global, typically 440.0)
    
    Returns:
        Pitch number (can be fractional for microtones)
    
    Example:
        >>> freq_to_pitch(440.0)  # A4 in 12-ET
        69.0
        >>> freq_to_pitch(261.6256)  # Middle C in 12-ET
        60.0
        >>> freq_to_pitch([261.626, 329.628, 391.995])  # C major chord
        array([60., 64., 67.])
    """
    temp = temperament if temperament is not None else get_temperament()
    
    # Get global reference if not specified
    if reference_freq is None or reference_pitch is None:
        global_freq, global_pitch = get_reference_frequency()
        reference_freq = reference_freq if reference_freq is not None else global_freq
        reference_pitch = reference_pitch if reference_pitch is not None else global_pitch
    
    return temp.freq_to_pitch(freq, reference_pitch, reference_freq)


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


def semitones_to_ratio(
    semitones: ArrayLike,
    temperament: Temperament | None = None
) -> np.ndarray:
    """
    Convert interval (in scale degrees) to frequency ratio.
    
    By default uses 12-tone equal temperament where 12 semitones = octave.
    In alternative temperaments, the interval is interpreted as scale degrees
    of that temperament.
    
    Args:
        semitones: Interval in scale degrees (semitones in 12-ET).
                   In 12-ET: 12 semitones = octave (ratio 2.0)
        temperament: Temperament to use (default: uses global temperament)
    
    Returns:
        Frequency ratio
    
    Example:
        >>> semitones_to_ratio(12)  # Octave in 12-ET
        2.0
        >>> semitones_to_ratio(7)   # Perfect fifth in 12-ET
        1.4983...
        >>> semitones_to_ratio(-12) # Octave down in 12-ET
        0.5
        
        >>> # In 19-ET, 19 scale degrees = octave
        >>> from pygmu2 import EqualTemperament
        >>> et19 = EqualTemperament(19)
        >>> semitones_to_ratio(19, temperament=et19)
        2.0
    """
    temp = temperament if temperament is not None else get_temperament()
    return temp.interval_to_ratio(semitones)


def ratio_to_semitones(
    ratio: ArrayLike,
    temperament: Temperament | None = None
) -> np.ndarray:
    """
    Convert frequency ratio to interval (in scale degrees).
    
    By default uses 12-tone equal temperament where octave = 12 semitones.
    In alternative temperaments, returns the interval in scale degrees
    of that temperament.
    
    Args:
        ratio: Frequency ratio. Must be positive.
               2.0 = octave
        temperament: Temperament to use (default: uses global temperament)
    
    Returns:
        Interval in scale degrees (semitones in 12-ET)
    
    Example:
        >>> ratio_to_semitones(2.0)  # Octave in 12-ET
        12.0
        >>> ratio_to_semitones(1.5)  # ~Perfect fifth in 12-ET
        7.0195...
        >>> ratio_to_semitones(0.5)  # Octave down in 12-ET
        -12.0
        
        >>> # In 19-ET, octave = 19 scale degrees
        >>> from pygmu2 import EqualTemperament
        >>> et19 = EqualTemperament(19)
        >>> ratio_to_semitones(2.0, temperament=et19)
        19.0
    """
    temp = temperament if temperament is not None else get_temperament()
    return temp.ratio_to_interval(ratio)


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
