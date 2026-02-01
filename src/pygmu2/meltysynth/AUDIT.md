# Meltysynth source code audit

Suggested changes, in order of decreasing priority: **clarity** → **pythonic/idiomatic** → **efficiency**.

---

## 1. Clarity

### 1.1 Use specific exception types

**Current:** `raise Exception("...")` throughout (file format errors, invalid data, missing chunks).

**Recommendation:** Use standard exceptions so callers can handle them appropriately:

- **Invalid file format / chunk / data:** `ValueError` (e.g. wrong chunk ID, bad size, unsupported format).
- **Missing required chunk:** `ValueError` (e.g. "The PHDR sub-chunk was not found.").
- **Invalid envelope/program state:** `RuntimeError` or `ValueError` (e.g. "Invalid envelope stage.").

Optionally introduce a single `SoundFontError(Exception)` or `MeltysynthError` and use it for all format/parse errors, so `except MeltysynthError` can catch them without touching other code.

**Files:** All under `model/`, `io/`, `synth/`, `midi/` that raise `Exception`.

---

### 1.2 Avoid shadowing the builtin `id`

**Current:** `id = BinaryReaderEx.read_four_cc(reader)` and `id = self._gs[GeneratorType.SAMPLE_ID]` (and similar).

**Recommendation:** Rename to `chunk_id` / `sub_id` in parsers, and `inst_id` / `sample_id` / `preset_id` where it denotes an index or identifier. This avoids shadowing `id()` and keeps names meaningful.

**Files:** `model/soundfont_info.py` (loop variable), `model/parameters.py`, `model/sample_data.py`, `model/instrument.py` (sample id), `model/preset.py` (instrument id).

---

### 1.3 F-strings for error messages

**Current:** String concatenation, e.g. `"The type of the LIST chunk must be 'INFO', but was '" + list_type + "'."`.

**Recommendation:** Use f-strings: `f"The type of the LIST chunk must be 'INFO', but was '{list_type}'."` — easier to read and less error-prone.

**Files:** All files that build error messages with `+`.

---

### 1.4 SoundFontInfo (and similar): initialize all attributes

**Current:** In `SoundFontInfo.__init__`, attributes like `_version`, `_bank_name`, etc. are only set inside the `match id:` loop. If a chunk type never appears (e.g. no "ifil"), that attribute is never set and property access can raise `AttributeError`.

**Recommendation:** At the start of `__init__`, set every attribute to a default (e.g. `SoundFontVersion(0, 0)`, `""` for strings). Then the loop only overwrites when the chunk is present. Apply the same pattern to any other parser that has optional chunks.

**Files:** `model/soundfont_info.py`.

---

### 1.5 Naming: `actualLength` → `actual_length`

**Current:** `actualLength` in `binary_reader.read_fixed_length_string`.

**Recommendation:** Use `actual_length` (snake_case) for consistency with PEP 8.

**File:** `io/binary_reader.py`.

---

### 1.6 Docstrings

**Current:** Most classes and public methods lack docstrings.

**Recommendation:** Add short docstrings to public classes and key methods (e.g. `SoundFont.from_file`, `Synthesizer.render`, `BinaryReaderEx.read_int_variable_length`). One line is enough for many; clarify semantics (e.g. "Returns …", "Raises ValueError if …").

---

## 2. Pythonic / idiomatic

### 2.1 Static-only classes → module-level functions (optional)

**Current:** `BinaryReaderEx`, `SoundFontMath`, and `ArrayMath` are classes used only as namespaces of static methods (e.g. `BinaryReaderEx.read_uint16(reader)`).

**Recommendation:** For a more Pythonic style, move these to module-level functions (e.g. in `binary_reader.py`: `def read_uint16(reader: BufferedIOBase) -> int:`). Call sites become `binary_reader.read_uint16(reader)`. This is a refactor; if you prefer to keep the class as a namespace, the current style is acceptable.

---

### 2.2 Integer division

**Current:** `int(size / 4)`, `int(size / 2)`, `int(size % 4)`.

**Recommendation:** Use `size // 4`, `size // 2`, and `size % 4` (no `int()`). Clearer and avoids float intermediate.

**Files:** `io/binary_reader.py`, `model/generator.py`, `model/zone_info.py`, `model/preset_info.py`, `model/instrument_info.py`, `model/sample_header.py`, etc.

---

### 2.3 List comprehensions where appropriate

**Current:** In `Zone.create`, a loop builds `gs` then `zones.append(Zone(gs))`.

**Recommendation:**  
`gs = [generators[info.generator_index + j] for j in range(info.generator_count)]`  
then `zones.append(Zone(gs))`. Same pattern elsewhere where you build a list in a loop and append once.

**File:** `model/zone.py`.

---

### 2.4 Simplify one-line conditionals

**Current:** e.g. in `SoundFontMath.exp_cutoff`: multi-line if/else; in `clamp`: if/elif/else.

**Recommendation:**  
- `exp_cutoff`: `return 0.0 if x < SoundFontMath.log_non_audible() else math.exp(x)`  
- `clamp`: `return max(min_val, min(max_val, value))`  

Only if it doesn’t hurt readability.

**File:** `math_utils.py`.

---

### 2.5 MidiMessage.type property

**Current:** `match self.channel:` with `case int(MidiMessageType.TEMPO_CHANGE):` — compares first data byte to enum integer values.

**Recommendation:** Compare explicitly to the enum:  
`if self._data[0] == MidiMessageType.TEMPO_CHANGE: return MidiMessageType.TEMPO_CHANGE`  
and similarly for `END_OF_TRACK`, else `NORMAL`. Avoids relying on `int(enum)` in a match and makes the intent clear.

**File:** `midi/message.py`.

---

### 2.6 Constants for magic numbers

**Current:** e.g. `min_preset_id = 10000000000` in synthesizer; similar sentinels elsewhere.

**Recommendation:** Name them at module or class level, e.g. `_INITIAL_MIN_PRESET_ID = 10**10` (or a clear name like `_PRESET_ID_SENTINEL`), and use that. Improves readability and intent.

**Files:** `synth/synthesizer.py`, `midi/midi_file.py` (merge loop), etc.

---

### 2.7 SoundFontVersion as a dataclass

**Current:** Simple class with `__init__` and two properties.

**Recommendation:**  
`@dataclass(frozen=True)` with `major: int` and `minor: int`. Shorter and idiomatic for value-like types.

**File:** `model/types.py`.

---

### 2.8 Remove no-op TYPE_CHECKING block

**Current:** `if TYPE_CHECKING: pass` in `math_utils.py`.

**Recommendation:** Remove it. Use `if TYPE_CHECKING:` only when you have imports inside it.

**File:** `math_utils.py`.

---

## 3. Efficiency

### 3.1 Module-level constants for frequently used math values

**Current:** `SoundFontMath.non_audible()` and `SoundFontMath.log_non_audible()` are called often (e.g. in voice process loop, envelope).

**Recommendation:** In `math_utils`, define module-level constants, e.g. `NON_AUDIBLE = 1e-3` and `LOG_NON_AUDIBLE = math.log(1e-3)`, and use them in the class methods and call sites. Avoids repeated calls and repeated log computation. Same for other pure constants if they’re hot (e.g. `HALF_PI = math.pi / 2`).

**Files:** `math_utils.py`, and call sites in `synth/`, `model/`.

---

### 3.2 read_int16_array_as_float_array: avoid lambda

**Current:** `array("f", map(lambda x: x / 32768.0, data))`.

**Recommendation:** Use a generator: `array("f", (x / 32768.0 for x in data))`. Slightly more idiomatic and avoids lambda. For large arrays, the difference is small; for a larger win you’d need something like numpy, which may be out of scope.

**File:** `io/binary_reader.py`.

---

### 3.3 Reuse buffer in voice (optional)

**Current:** Each voice allocates a block buffer at init and reuses it; that’s already good.

**Recommendation:** No change needed. If you ever add voice pooling, reusing buffers is already in place.

---

## Summary table

| Priority   | Category   | Item                          | Files / scope                    |
|-----------|------------|--------------------------------|----------------------------------|
| Clarity   | Exceptions | Use ValueError / custom       | All raise Exception              |
| Clarity   | Naming     | Rename `id` → chunk_id etc.    | soundfont_info, parameters, etc. |
| Clarity   | Messages   | F-strings for errors          | All error strings                |
| Clarity   | Init       | Default all SoundFontInfo attrs| soundfont_info.py                |
| Clarity   | Naming     | actualLength → actual_length   | binary_reader.py                 |
| Clarity   | Docs       | Add docstrings                | Public API                       |
| Pythonic  | Structure  | Static classes → functions (opt)| binary_reader, math_utils        |
| Pythonic  | Literals   | Use // and %                  | All int division                 |
| Pythonic  | Comprehensions | Zone.create list comp       | zone.py                          |
| Pythonic  | Conditionals | One-line exp_cutoff, clamp   | math_utils.py                    |
| Pythonic  | MidiMessage | type property comparison      | midi/message.py                  |
| Pythonic  | Constants  | Name magic numbers            | synthesizer, midi_file           |
| Pythonic  | Types      | SoundFontVersion dataclass    | model/types.py                   |
| Pythonic  | Cleanup    | Remove empty TYPE_CHECKING    | math_utils.py                    |
| Efficiency| Constants  | NON_AUDIBLE at module level   | math_utils + call sites          |
| Efficiency| Array      | Generator instead of lambda   | binary_reader.py                 |

Implementing the clarity items first will give the largest benefit; pythonic and efficiency changes can be done incrementally.
