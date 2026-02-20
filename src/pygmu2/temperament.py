"""
Musical temperament and tuning system support.

Provides a flexible system for alternative temperaments including equal temperaments,
just intonation, historical temperaments, and custom tuning systems.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike


class Temperament(ABC):
    """
    Base class for musical temperaments.
    
    A temperament defines how pitch numbers map to frequencies and how
    intervals relate to frequency ratios. Different temperaments produce
    different tuning characteristics and harmonic relationships.
    """
    
    @abstractmethod
    def pitch_to_freq(
        self, 
        pitch: ArrayLike,
        reference_pitch: float = 69.0,
        reference_freq: float = 440.0
    ) -> np.ndarray:
        """
        Convert pitch number to frequency in Hz.
        
        Args:
            pitch: Pitch number(s). Can be fractional.
            reference_pitch: Reference pitch number (default: 69.0 for A4)
            reference_freq: Reference frequency in Hz (default: 440.0)
        
        Returns:
            Frequency in Hz
        """
        pass
    
    @abstractmethod
    def freq_to_pitch(
        self,
        freq: ArrayLike,
        reference_pitch: float = 69.0,
        reference_freq: float = 440.0
    ) -> np.ndarray:
        """
        Convert frequency in Hz to pitch number.
        
        Args:
            freq: Frequency in Hz. Must be positive.
            reference_pitch: Reference pitch number (default: 69.0 for A4)
            reference_freq: Reference frequency in Hz (default: 440.0)
        
        Returns:
            Pitch number (can be fractional)
        """
        pass
    
    @abstractmethod
    def interval_to_ratio(self, interval: ArrayLike) -> np.ndarray:
        """
        Convert interval (in scale degrees) to frequency ratio.
        
        Args:
            interval: Interval in scale degrees
        
        Returns:
            Frequency ratio
        """
        pass
    
    @abstractmethod
    def ratio_to_interval(self, ratio: ArrayLike) -> np.ndarray:
        """
        Convert frequency ratio to interval (in scale degrees).
        
        Args:
            ratio: Frequency ratio. Must be positive.
        
        Returns:
            Interval in scale degrees
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this temperament."""
        pass


class EqualTemperament(Temperament):
    """
    Equal temperament with configurable divisions per octave.
    
    In equal temperament, the octave is divided into a fixed number of
    equal steps (divisions). 12-tone equal temperament (12-ET) is the
    standard in Western music, but other divisions are common:
    - 19-ET: Better approximation of just intervals
    - 24-ET: Quarter-tone music
    - 31-ET: Excellent approximation of meantone
    - 53-ET: Very close to Pythagorean tuning
    
    Args:
        divisions: Number of equal divisions per octave (default: 12)
    
    Example:
        >>> et12 = EqualTemperament(12)  # Standard tuning
        >>> et12.pitch_to_freq(69)  # A4
        440.0
        
        >>> et19 = EqualTemperament(19)  # 19-tone equal temperament
        >>> et19.pitch_to_freq(69)  # A4
        440.0
        
        >>> et24 = EqualTemperament(24)  # Quarter-tone music
        >>> et24.interval_to_ratio(12)  # Half-octave
        1.4142...
    """
    
    def __init__(self, divisions: int = 12):
        if divisions < 1:
            raise ValueError(f"Divisions must be positive, got {divisions}")
        self._divisions = divisions
    
    @property
    def divisions(self) -> int:
        """Number of equal divisions per octave."""
        return self._divisions
    
    def pitch_to_freq(
        self,
        pitch: ArrayLike,
        reference_pitch: float = 69.0,
        reference_freq: float = 440.0
    ) -> np.ndarray:
        """Convert pitch number to frequency using equal temperament."""
        pitch = np.asarray(pitch, dtype=np.float64)
        return reference_freq * (2.0 ** ((pitch - reference_pitch) / self._divisions))
    
    def freq_to_pitch(
        self,
        freq: ArrayLike,
        reference_pitch: float = 69.0,
        reference_freq: float = 440.0
    ) -> np.ndarray:
        """Convert frequency to pitch number using equal temperament."""
        freq = np.asarray(freq, dtype=np.float64)
        freq = np.maximum(freq, 1e-10)  # Protect against log of zero
        return reference_pitch + self._divisions * np.log2(freq / reference_freq)
    
    def interval_to_ratio(self, interval: ArrayLike) -> np.ndarray:
        """Convert interval to frequency ratio."""
        interval = np.asarray(interval, dtype=np.float64)
        return 2.0 ** (interval / self._divisions)
    
    def ratio_to_interval(self, ratio: ArrayLike) -> np.ndarray:
        """Convert frequency ratio to interval."""
        ratio = np.asarray(ratio, dtype=np.float64)
        ratio = np.maximum(ratio, 1e-10)  # Protect against log of zero
        return self._divisions * np.log2(ratio)
    
    def name(self) -> str:
        """Return name of this temperament."""
        return f"{self._divisions}-tone Equal Temperament ({self._divisions}-ET)"
    
    def __repr__(self) -> str:
        return f"EqualTemperament(divisions={self._divisions})"


class JustIntonation(Temperament):
    """
    Just intonation with configurable ratio table.
    
    Just intonation uses simple whole-number frequency ratios, producing
    pure harmonic intervals. Different JI systems exist based on which
    intervals are prioritized (3-limit, 5-limit, 7-limit, etc.).
    
    This implementation uses a ratio table for one octave, with interpolation
    for fractional pitches and octave transposition for pitches outside [0, 12).
    
    Args:
        ratios: List of frequency ratios for one octave. If None, uses
                5-limit JI ratios (12 notes). First ratio should be 1.0 (unison).
        reference_pitch: The pitch number that corresponds to ratios[0]
                         (default: 60.0 for Middle C)
    
    Example:
        >>> # 5-limit just intonation (default)
        >>> ji = JustIntonation()
        >>> ji.pitch_to_freq(60)  # C (1/1)
        261.6255...
        
        >>> # Pythagorean tuning (3-limit)
        >>> pythagorean_ratios = [1, 256/243, 9/8, 32/27, 81/64, 4/3, 
        ...                       1024/729, 3/2, 128/81, 27/16, 16/9, 243/128]
        >>> pyth = JustIntonation(pythagorean_ratios)
    """
    
    def __init__(
        self,
        ratios: list[float] | None = None,
        reference_pitch: float = 60.0
    ):
        if ratios is None:
            # Default: 5-limit just intonation
            # Based on the major scale with pure 3/2 fifths and 5/4 major thirds
            self._ratios = np.array([
                1.0,      # C  - Unison (1/1)
                16/15,    # C# - Minor second
                9/8,      # D  - Major second
                6/5,      # Eb - Minor third
                5/4,      # E  - Major third
                4/3,      # F  - Perfect fourth
                45/32,    # F# - Augmented fourth / Diminished fifth
                3/2,      # G  - Perfect fifth
                8/5,      # Ab - Minor sixth
                5/3,      # A  - Major sixth
                9/5,      # Bb - Minor seventh
                15/8,     # B  - Major seventh
            ], dtype=np.float64)
        else:
            self._ratios = np.asarray(ratios, dtype=np.float64)
            if len(self._ratios) < 2:
                raise ValueError("Need at least 2 ratios (including unison)")
            if not np.isclose(self._ratios[0], 1.0):
                raise ValueError("First ratio must be 1.0 (unison)")
        
        self._reference_pitch = reference_pitch
        self._num_notes = len(self._ratios)
    
    @property
    def ratios(self) -> np.ndarray:
        """The frequency ratios for one octave."""
        return self._ratios.copy()
    
    @property
    def num_notes(self) -> int:
        """Number of notes in the ratio table."""
        return self._num_notes
    
    def pitch_to_freq(
        self,
        pitch: ArrayLike,
        reference_pitch: float = 69.0,
        reference_freq: float = 440.0
    ) -> np.ndarray:
        """
        Convert pitch to frequency using just intonation.
        
        Pitches are mapped to the ratio table modulo the number of notes,
        with octave transposition. Fractional pitches are linearly interpolated
        in log-frequency space.
        """
        pitch = np.asarray(pitch, dtype=np.float64)
        
        # Offset pitch to be relative to our reference_pitch
        relative_pitch = pitch - self._reference_pitch
        
        # Split into octaves and scale degrees
        octaves = np.floor(relative_pitch / self._num_notes)
        scale_degrees = relative_pitch - octaves * self._num_notes
        
        # Get ratios by interpolation
        ratios = self._interpolate_ratios(scale_degrees)
        
        # Apply octave transposition
        total_ratio = ratios * (2.0 ** octaves)
        
        # Convert reference_pitch to frequency in our JI system
        ref_offset = reference_pitch - self._reference_pitch
        ref_octaves = np.floor(ref_offset / self._num_notes)
        ref_scale_degree = ref_offset - ref_octaves * self._num_notes
        ref_ratio = self._interpolate_ratios(ref_scale_degree) * (2.0 ** ref_octaves)
        
        # Base frequency at our reference pitch
        base_freq = reference_freq / ref_ratio
        
        return base_freq * total_ratio
    
    def freq_to_pitch(
        self,
        freq: ArrayLike,
        reference_pitch: float = 69.0,
        reference_freq: float = 440.0
    ) -> np.ndarray:
        """
        Convert frequency to pitch using just intonation.
        
        Note: This is approximate for JI since multiple pitches can map to
        similar frequencies. Returns the closest pitch in the ratio table.
        """
        freq = np.asarray(freq, dtype=np.float64)
        freq = np.maximum(freq, 1e-10)
        
        # Get the reference ratio
        ref_offset = reference_pitch - self._reference_pitch
        ref_octaves = np.floor(ref_offset / self._num_notes)
        ref_scale_degree = ref_offset - ref_octaves * self._num_notes
        ref_ratio = self._interpolate_ratios(ref_scale_degree) * (2.0 ** ref_octaves)
        
        # Base frequency at our reference pitch
        base_freq = reference_freq / ref_ratio
        
        # Ratio from base
        ratio = freq / base_freq
        
        # Find octave
        octaves = np.floor(np.log2(ratio))
        ratio_in_octave = ratio / (2.0 ** octaves)
        
        # Find closest scale degree
        # This is approximate - just find the nearest ratio
        scale_degrees = np.zeros_like(ratio_in_octave)
        ratio_in_octave_scalar = np.atleast_1d(ratio_in_octave)
        
        for i, r in enumerate(ratio_in_octave_scalar):
            # Find closest ratio in table
            idx = np.argmin(np.abs(self._ratios - r))
            scale_degrees[i] = idx
        
        relative_pitch = octaves * self._num_notes + scale_degrees
        return self._reference_pitch + relative_pitch
    
    def interval_to_ratio(self, interval: ArrayLike) -> np.ndarray:
        """
        Convert interval to frequency ratio.
        
        Intervals are in scale degrees of this JI system.
        """
        interval = np.asarray(interval, dtype=np.float64)
        
        # Split into octaves and scale degrees
        octaves = np.floor(interval / self._num_notes)
        scale_degrees = interval - octaves * self._num_notes
        
        # Get ratios by interpolation
        ratios = self._interpolate_ratios(scale_degrees)
        
        # Apply octave transposition
        return ratios * (2.0 ** octaves)
    
    def ratio_to_interval(self, ratio: ArrayLike) -> np.ndarray:
        """
        Convert frequency ratio to interval.
        
        Returns approximate interval in scale degrees.
        """
        ratio = np.asarray(ratio, dtype=np.float64)
        ratio = np.maximum(ratio, 1e-10)
        
        # Find octave
        octaves = np.floor(np.log2(ratio))
        ratio_in_octave = ratio / (2.0 ** octaves)
        
        # Find closest scale degree (approximate)
        scale_degrees = np.zeros_like(ratio_in_octave)
        ratio_in_octave_scalar = np.atleast_1d(ratio_in_octave)
        
        for i, r in enumerate(ratio_in_octave_scalar):
            idx = np.argmin(np.abs(self._ratios - r))
            scale_degrees[i] = idx
        
        return octaves * self._num_notes + scale_degrees
    
    def _interpolate_ratios(self, scale_degrees: np.ndarray) -> np.ndarray:
        """
        Interpolate ratios for fractional scale degrees.
        
        Uses linear interpolation in log-frequency space.
        """
        scale_degrees = np.atleast_1d(scale_degrees)
        
        # Floor and fractional parts
        floor_idx = np.floor(scale_degrees).astype(int)
        frac = scale_degrees - floor_idx
        
        # Wrap indices
        floor_idx = floor_idx % self._num_notes
        ceil_idx = (floor_idx + 1) % self._num_notes
        
        # Get ratios
        floor_ratios = self._ratios[floor_idx]
        ceil_ratios = self._ratios[ceil_idx]
        
        # Handle wraparound (octave boundary)
        # If we wrapped, multiply ceiling by 2
        wrapped = (floor_idx == self._num_notes - 1) & (frac > 0)
        ceil_ratios = np.where(wrapped, ceil_ratios * 2.0, ceil_ratios)
        
        # Linear interpolation in log space (geometric in linear space)
        log_floor = np.log2(floor_ratios)
        log_ceil = np.log2(ceil_ratios)
        log_interp = log_floor + frac * (log_ceil - log_floor)
        
        return 2.0 ** log_interp
    
    def name(self) -> str:
        """Return name of this temperament."""
        return f"Just Intonation ({self._num_notes} notes)"
    
    def __repr__(self) -> str:
        return f"JustIntonation(num_notes={self._num_notes}, reference_pitch={self._reference_pitch})"


class PythagoreanTuning(JustIntonation):
    """
    Pythagorean tuning based on pure 3:2 fifths.
    
    Pythagorean tuning is a 3-limit just intonation system where all
    intervals are derived from stacking perfect fifths (3/2 ratio).
    This produces very pure fifths and fourths, but major thirds are
    quite sharp compared to just intonation.
    
    Example:
        >>> pyth = PythagoreanTuning()
        >>> pyth.pitch_to_freq(60)  # C
        261.6255...
        >>> pyth.interval_to_ratio(7)  # Perfect fifth
        1.5  # Exactly 3/2
    """
    
    def __init__(self, reference_pitch: float = 60.0):
        # Generate ratios by stacking fifths and reducing to one octave
        # Start from C and go: F C G D A E B F# C# G# D# A#
        pythagorean_ratios = [
            1.0,        # C   (0 fifths)
            256/243,    # C#  (7 fifths up)
            9/8,        # D   (2 fifths up)
            32/27,      # Eb  (5 fifths down)
            81/64,      # E   (4 fifths up)
            4/3,        # F   (1 fifth down)
            1024/729,   # F#  (6 fifths up)
            3/2,        # G   (1 fifth up)
            128/81,     # Ab  (4 fifths down)
            27/16,      # A   (3 fifths up)
            16/9,       # Bb  (2 fifths down)
            243/128,    # B   (5 fifths up)
        ]
        super().__init__(ratios=pythagorean_ratios, reference_pitch=reference_pitch)
    
    def name(self) -> str:
        """Return name of this temperament."""
        return "Pythagorean Tuning"
    
    def __repr__(self) -> str:
        return f"PythagoreanTuning(reference_pitch={self._reference_pitch})"


class CustomTemperament(Temperament):
    """
    User-defined temperament from pitch-to-frequency and frequency-to-pitch functions.
    
    Allows complete customization of the tuning system by providing callable
    functions for all required conversions.
    
    Args:
        pitch_to_freq_func: Callable that converts pitch to frequency
        freq_to_pitch_func: Callable that converts frequency to pitch
        interval_to_ratio_func: Callable that converts interval to ratio
        ratio_to_interval_func: Callable that converts ratio to interval
        name: Human-readable name for this temperament
    
    Example:
        >>> # Create a stretched tuning where octaves are slightly > 2:1
        >>> def stretched_p2f(pitch, ref_pitch=69, ref_freq=440):
        ...     return ref_freq * (2.001 ** ((pitch - ref_pitch) / 12))
        >>> 
        >>> def stretched_f2p(freq, ref_pitch=69, ref_freq=440):
        ...     return ref_pitch + 12 * np.log(freq / ref_freq) / np.log(2.001)
        >>> 
        >>> stretched = CustomTemperament(
        ...     pitch_to_freq_func=stretched_p2f,
        ...     freq_to_pitch_func=stretched_f2p,
        ...     interval_to_ratio_func=lambda i: 2.001 ** (i / 12),
        ...     ratio_to_interval_func=lambda r: 12 * np.log(r) / np.log(2.001),
        ...     name="Stretched Tuning"
        ... )
    """
    
    def __init__(
        self,
        pitch_to_freq_func,
        freq_to_pitch_func,
        interval_to_ratio_func,
        ratio_to_interval_func,
        name: str = "Custom Temperament"
    ):
        self._pitch_to_freq_func = pitch_to_freq_func
        self._freq_to_pitch_func = freq_to_pitch_func
        self._interval_to_ratio_func = interval_to_ratio_func
        self._ratio_to_interval_func = ratio_to_interval_func
        self._name = name
    
    def pitch_to_freq(
        self,
        pitch: ArrayLike,
        reference_pitch: float = 69.0,
        reference_freq: float = 440.0
    ) -> np.ndarray:
        """Convert pitch to frequency using custom function."""
        result = self._pitch_to_freq_func(pitch, reference_pitch, reference_freq)
        return np.asarray(result, dtype=np.float64)
    
    def freq_to_pitch(
        self,
        freq: ArrayLike,
        reference_pitch: float = 69.0,
        reference_freq: float = 440.0
    ) -> np.ndarray:
        """Convert frequency to pitch using custom function."""
        result = self._freq_to_pitch_func(freq, reference_pitch, reference_freq)
        return np.asarray(result, dtype=np.float64)
    
    def interval_to_ratio(self, interval: ArrayLike) -> np.ndarray:
        """Convert interval to ratio using custom function."""
        result = self._interval_to_ratio_func(interval)
        return np.asarray(result, dtype=np.float64)
    
    def ratio_to_interval(self, ratio: ArrayLike) -> np.ndarray:
        """Convert ratio to interval using custom function."""
        result = self._ratio_to_interval_func(ratio)
        return np.asarray(result, dtype=np.float64)
    
    def name(self) -> str:
        """Return name of this temperament."""
        return self._name
    
    def __repr__(self) -> str:
        return f"CustomTemperament(name='{self._name}')"


# Default global temperament (12-ET)
_DEFAULT_TEMPERAMENT: Temperament = EqualTemperament(12)

# Default reference frequency (A4 = 440 Hz, concert pitch)
_DEFAULT_REFERENCE_FREQ: float = 440.0

# Default reference pitch (MIDI note 69 = A4)
_DEFAULT_REFERENCE_PITCH: float = 69.0


def set_temperament(temperament: Temperament) -> None:
    """
    Set the global default temperament for all pygmu2 operations.
    
    This affects all calls to conversion functions (pitch_to_freq, etc.)
    that don't explicitly specify a temperament.
    
    Args:
        temperament: The temperament to use as default
    
    Example:
        >>> from pygmu2 import set_temperament, EqualTemperament
        >>> set_temperament(EqualTemperament(19))  # Use 19-ET globally
    """
    global _DEFAULT_TEMPERAMENT
    _DEFAULT_TEMPERAMENT = temperament


def get_temperament() -> Temperament:
    """
    Get the current global default temperament.
    
    Returns:
        The current default temperament
    
    Example:
        >>> from pygmu2 import get_temperament
        >>> temp = get_temperament()
        >>> print(temp.name())
        12-tone Equal Temperament (12-ET)
    """
    return _DEFAULT_TEMPERAMENT


def set_reference_frequency(freq: float, pitch: float = 69.0) -> None:
    """
    Set the global default reference frequency.
    
    This affects all pitch-to-frequency conversions that don't explicitly
    specify a reference frequency.
    
    Common reference frequencies:
    - 440.0 Hz: Modern concert pitch (ISO 16, default)
    - 432.0 Hz: Alternative "Verdi" tuning
    - 415.0 Hz: Baroque pitch
    - 392.0 Hz: Classical pitch (18th century)
    
    Args:
        freq: Reference frequency in Hz (typically for A4)
        pitch: MIDI pitch number for the reference (default: 69.0 = A4)
    
    Example:
        >>> from pygmu2 import set_reference_frequency
        >>> set_reference_frequency(432.0)  # A4 = 432 Hz
        >>> set_reference_frequency(415.0)  # Baroque pitch
    """
    global _DEFAULT_REFERENCE_FREQ, _DEFAULT_REFERENCE_PITCH
    
    if freq <= 0:
        raise ValueError(f"Reference frequency must be positive, got {freq}")
    
    _DEFAULT_REFERENCE_FREQ = float(freq)
    _DEFAULT_REFERENCE_PITCH = float(pitch)


def get_reference_frequency() -> tuple[float, float]:
    """
    Get the current global reference frequency and pitch.
    
    Returns:
        Tuple of (reference_freq, reference_pitch)
    
    Example:
        >>> from pygmu2 import get_reference_frequency
        >>> freq, pitch = get_reference_frequency()
        >>> print(f"A4 = {freq} Hz")
        A4 = 440.0 Hz
    """
    return (_DEFAULT_REFERENCE_FREQ, _DEFAULT_REFERENCE_PITCH)


def set_concert_pitch() -> None:
    """
    Set reference to modern concert pitch (A4 = 440 Hz).
    
    This is the ISO 16 standard and the default.
    
    Example:
        >>> from pygmu2 import set_concert_pitch
        >>> set_concert_pitch()
    """
    set_reference_frequency(440.0, 69.0)


def set_verdi_tuning() -> None:
    """
    Set reference to A4 = 432 Hz (Verdi tuning).
    
    Also known as "philosophical pitch" or "scientific pitch".
    Some believe this tuning is more harmonious.
    
    Example:
        >>> from pygmu2 import set_verdi_tuning
        >>> set_verdi_tuning()
    """
    set_reference_frequency(432.0, 69.0)


def set_baroque_pitch() -> None:
    """
    Set reference to A4 = 415 Hz (Baroque pitch).
    
    Common for historically informed performances of Baroque music.
    
    Example:
        >>> from pygmu2 import set_baroque_pitch
        >>> set_baroque_pitch()
    """
    set_reference_frequency(415.0, 69.0)
