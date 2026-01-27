# Alternative Temperament System Implementation

## Summary

Successfully implemented a comprehensive alternative temperament system for pygmu2, allowing users to work with different musical tuning systems beyond the standard 12-tone equal temperament (12-ET).

## What Was Implemented

### 1. Core Temperament Module (`src/pygmu2/temperament.py`)

**Base Classes:**
- `Temperament` - Abstract base class defining the temperament interface
  - `pitch_to_freq()` - Convert pitch numbers to frequencies
  - `freq_to_pitch()` - Convert frequencies to pitch numbers
  - `interval_to_ratio()` - Convert scale intervals to frequency ratios
  - `ratio_to_interval()` - Convert frequency ratios to intervals

**Concrete Temperament Implementations:**

1. **`EqualTemperament(divisions)`**
   - N-tone equal temperament (12-ET, 19-ET, 24-ET, 31-ET, 53-ET, etc.)
   - Divides the octave into equal logarithmic steps
   - Default: 12 divisions (standard Western tuning)

2. **`JustIntonation(ratios, reference_pitch)`**
   - Just intonation with pure harmonic ratios
   - Default: 5-limit JI with ratios like 5/4 (major third), 3/2 (fifth)
   - Supports custom ratio tables
   - Interpolates fractional pitches in log-frequency space

3. **`PythagoreanTuning(reference_pitch)`**
   - 3-limit tuning based on pure 3:2 fifths
   - Perfect fifths (exactly 3/2)
   - Sharper major thirds than just intonation (81/64 vs 5/4)

4. **`CustomTemperament(functions...)`**
   - User-defined temperament from custom conversion functions
   - Allows complete flexibility for experimental tunings

### 2. Updated Conversion Functions (`src/pygmu2/conversions.py`)

All pitch/frequency conversion functions now support temperaments:

- `pitch_to_freq(pitch, temperament=None, reference_pitch=69.0, reference_freq=440.0)`
- `freq_to_pitch(freq, temperament=None, reference_pitch=69.0, reference_freq=440.0)`
- `semitones_to_ratio(interval, temperament=None)`
- `ratio_to_semitones(ratio, temperament=None)`

**Backward Compatibility:**
- All existing code continues to work without modification
- Default behavior uses 12-ET (standard tuning)
- Optional `temperament` parameter allows explicit control

### 3. Global Temperament Configuration

```python
from pygmu2 import set_temperament, get_temperament, EqualTemperament

# Set global default
set_temperament(EqualTemperament(19))

# All subsequent conversions use 19-ET
freq = pitch_to_freq(60)  # Uses 19-ET

# Get current global temperament
current = get_temperament()
```

### 4. Comprehensive Tests (`tests/test_temperament.py`)

**41 test cases covering:**
- Equal temperament (12-ET, 19-ET, 24-ET)
- Just intonation (default and custom ratios)
- Pythagorean tuning (perfect fifths and intervals)
- Custom temperament (stretched tuning example)
- Global temperament configuration
- Integration with conversion functions
- Backward compatibility
- Array and scalar input handling

**All tests pass:** ✅ 79/79 tests passing (38 conversion + 41 temperament)

### 5. Example (`examples/18_alternative_temperaments.py`)

Comprehensive demonstration including:
- C major scales in 4 different temperaments (12-ET, 19-ET, JI, Pythagorean)
- Frequency comparison table
- Interval ratio analysis (major third in different systems)
- Chord quality comparison (C major triad)
- Clear audible differences between temperaments

### 6. Documentation Updates

**README.md:**
- Added "Alternative temperament support" to key features
- New section explaining temperament usage with examples
- Added example 18 to the examples list

## Usage Examples

### Basic Usage

```python
from pygmu2 import pitch_to_freq, EqualTemperament, JustIntonation

# Use 19-tone equal temperament
et19 = EqualTemperament(19)
freq = pitch_to_freq(69, temperament=et19)

# Use just intonation for pure harmonies
ji = JustIntonation()
c_major = [pitch_to_freq(p, temperament=ji) for p in [60, 64, 67]]
```

### Global Configuration

```python
from pygmu2 import set_temperament, pitch_to_freq, JustIntonation

# Set global temperament
set_temperament(JustIntonation())

# All conversions now use JI
freqs = [pitch_to_freq(p) for p in [60, 64, 67]]
```

### Custom Temperament

```python
from pygmu2 import CustomTemperament
import numpy as np

# Create stretched octave tuning
def stretched_p2f(pitch, ref_pitch=69, ref_freq=440):
    return ref_freq * (2.001 ** ((pitch - ref_pitch) / 12))

def stretched_f2p(freq, ref_pitch=69, ref_freq=440):
    return ref_pitch + 12 * np.log2(freq / ref_freq) / np.log2(2.001)

stretched = CustomTemperament(
    pitch_to_freq_func=stretched_p2f,
    freq_to_pitch_func=stretched_f2p,
    interval_to_ratio_func=lambda i: 2.001 ** (i / 12),
    ratio_to_interval_func=lambda r: 12 * np.log2(r) / np.log2(2.001),
    name="Stretched Tuning"
)
```

## Design Decisions

### 1. Backward Compatibility
- All existing code works without changes
- Default behavior identical to pre-temperament system (12-ET)
- Optional parameters maintain clean API

### 2. Flexibility vs Simplicity
- Provides both global default and per-call temperament specification
- Global default reduces boilerplate for consistent usage
- Explicit parameter allows mixing temperaments in same program

### 3. Extensibility
- Abstract base class allows unlimited custom temperaments
- `CustomTemperament` provides escape hatch for experimental tunings
- Ratio tables support non-Western scales (Arabic maqam, Indian shruti, etc.)

### 4. Performance
- All conversion functions vectorized (work with numpy arrays)
- No performance penalty for default 12-ET usage
- Efficient interpolation for fractional pitches in JI

## Future Enhancements

Possible future additions:
1. **Historical Temperaments** - Werckmeister, Kirnberger, meantone variants
2. **Non-Western Scales** - Arabic maqam, Indian shruti, Thai scales
3. **MIDI Tuning Standard** - Support for MTS (MIDI Tuning Standard)
4. **Adaptive Tuning** - Dynamic temperament per note (barbershop, a cappella)
5. **Temperament-Aware PEs** - Oscillators that adjust timbre based on temperament

## Compatibility

- ✅ All existing tests pass (100% backward compatible)
- ✅ No breaking changes to API
- ✅ Works with all existing oscillators and processing elements
- ✅ Integrates with existing pitch conversion workflows
- ✅ 76% test coverage for new temperament module

## Files Modified/Added

**New Files:**
- `src/pygmu2/temperament.py` (165 lines)
- `tests/test_temperament.py` (419 lines)
- `examples/18_alternative_temperaments.py` (309 lines)

**Modified Files:**
- `src/pygmu2/conversions.py` - Added temperament parameter support
- `src/pygmu2/__init__.py` - Export temperament classes
- `README.md` - Documentation updates

**Total:** ~900 lines of new code + comprehensive tests + example + documentation
