"""
Tests for meltysynth (SoundFont loading and synthesis).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import io
import math
from pathlib import Path

import pytest

from pygmu2.meltysynth import (
    SoundFont,
    Synthesizer,
    SynthesizerSettings,
    create_buffer,
)
from pygmu2.meltysynth.io.binary_reader import BinaryReaderEx
from pygmu2.meltysynth.math_utils import ArrayMath, SoundFontMath
from pygmu2.meltysynth.model.types import GeneratorType, LoopMode, SoundFontVersion


# Path to optional SoundFont (from repo root). Tests that need it are skipped if missing.
def _soundfont_path() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    path = root / "examples" / "audio" / "TimGM6mb.sf2"
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# create_buffer
# ---------------------------------------------------------------------------
class TestCreateBuffer:
    def test_returns_array_of_requested_length(self):
        buf = create_buffer(100)
        assert len(buf) == 100

    def test_values_are_zero(self):
        buf = create_buffer(10)
        for i in range(10):
            assert buf[i] == 0.0

    def test_is_mutable(self):
        buf = create_buffer(5)
        buf[2] = 1.0
        assert buf[2] == 1.0


# ---------------------------------------------------------------------------
# SoundFontMath
# ---------------------------------------------------------------------------
class TestSoundFontMath:
    def test_half_pi(self):
        assert SoundFontMath.half_pi() == pytest.approx(math.pi / 2)

    def test_non_audible(self):
        assert SoundFontMath.non_audible() == 1e-3

    def test_log_non_audible(self):
        assert SoundFontMath.log_non_audible() == pytest.approx(math.log(1e-3))

    def test_timecents_to_seconds_zero(self):
        assert SoundFontMath.timecents_to_seconds(0) == pytest.approx(1.0)

    def test_cents_to_hertz_zero(self):
        assert SoundFontMath.cents_to_hertz(0) == pytest.approx(8.176)

    def test_cents_to_multiplying_factor_zero(self):
        assert SoundFontMath.cents_to_multiplying_factor(0) == pytest.approx(1.0)

    def test_decibels_to_linear_zero_db(self):
        assert SoundFontMath.decibels_to_linear(0) == pytest.approx(1.0)

    def test_linear_to_decibels_one(self):
        assert SoundFontMath.linear_to_decibels(1.0) == pytest.approx(0.0)

    def test_clamp_in_range(self):
        assert SoundFontMath.clamp(0.5, 0.0, 1.0) == 0.5

    def test_clamp_below_min(self):
        assert SoundFontMath.clamp(-1.0, 0.0, 1.0) == 0.0

    def test_clamp_above_max(self):
        assert SoundFontMath.clamp(2.0, 0.0, 1.0) == 1.0

    def test_exp_cutoff_below_threshold(self):
        assert SoundFontMath.exp_cutoff(-100) == 0.0

    def test_exp_cutoff_above_threshold(self):
        assert SoundFontMath.exp_cutoff(0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ArrayMath
# ---------------------------------------------------------------------------
class TestArrayMath:
    def test_multiply_add(self):
        dest = create_buffer(3)
        dest[0], dest[1], dest[2] = 1.0, 2.0, 3.0
        x = [0.1, 0.2, 0.3]
        ArrayMath.multiply_add(2.0, x, dest)
        assert dest[0] == pytest.approx(1.2)
        assert dest[1] == pytest.approx(2.4)
        assert dest[2] == pytest.approx(3.6)

    def test_multiply_add_slope(self):
        dest = create_buffer(3)
        dest[0], dest[1], dest[2] = 0.0, 0.0, 0.0
        x = [1.0, 1.0, 1.0]
        ArrayMath.multiply_add_slope(1.0, 0.5, x, dest)
        # a starts 1, then 1.5, then 2.0
        assert dest[0] == pytest.approx(1.0)
        assert dest[1] == pytest.approx(1.5)
        assert dest[2] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# BinaryReaderEx (using BytesIO)
# ---------------------------------------------------------------------------
class TestBinaryReaderEx:
    def test_read_int32_little_endian(self):
        data = (0x01).to_bytes(1, "little") + (0x02).to_bytes(1, "little") + (0x03).to_bytes(1, "little") + (0x04).to_bytes(1, "little")
        r = io.BytesIO(data)
        assert BinaryReaderEx.read_int32(r) == 0x04030201

    def test_read_uint16(self):
        data = (0x34).to_bytes(1, "little") + (0x12).to_bytes(1, "little")
        r = io.BytesIO(data)
        assert BinaryReaderEx.read_uint16(r) == 0x1234

    def test_read_four_cc(self):
        r = io.BytesIO(b"LIST")
        assert BinaryReaderEx.read_four_cc(r) == "LIST"

    def test_read_four_cc_replaces_non_printable(self):
        r = io.BytesIO(bytes([0x20, 0x00, 0x20, 0x20]))  # space, NUL, space, space
        assert BinaryReaderEx.read_four_cc(r) == " ?  "

    def test_read_fixed_length_string_stops_at_nul(self):
        r = io.BytesIO(b"hello\x00\x00\x00")
        assert BinaryReaderEx.read_fixed_length_string(r, 8) == "hello"

    def test_read_int_variable_length_single_byte(self):
        r = io.BytesIO(bytes([0x7f]))
        assert BinaryReaderEx.read_int_variable_length(r) == 0x7F

    def test_read_int_variable_length_two_bytes(self):
        # 0x81 0x00 = 128 (high bit set, then 0)
        r = io.BytesIO(bytes([0x81, 0x00]))
        assert BinaryReaderEx.read_int_variable_length(r) == 128


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class TestSoundFontVersion:
    def test_major_minor(self):
        v = SoundFontVersion(2, 1)
        assert v.major == 2
        assert v.minor == 1


class TestGeneratorType:
    def test_sample_id_value(self):
        assert GeneratorType.SAMPLE_ID == 53


class TestLoopMode:
    def test_no_loop_value(self):
        assert LoopMode.NO_LOOP == 0

    def test_continuous_value(self):
        assert LoopMode.CONTINUOUS == 1


# ---------------------------------------------------------------------------
# SoundFont + Synthesizer (require TimGM6mb.sf2 in examples/audio)
# ---------------------------------------------------------------------------
class TestSoundFontLoad:
    def test_from_file_raises_when_file_invalid(self):
        with pytest.raises(Exception):
            SoundFont.from_file("/nonexistent/path.sf2")

    def test_from_file_loads_valid_soundfont(self):
        path = _soundfont_path()
        if path is None:
            pytest.skip("examples/audio/TimGM6mb.sf2 not found")
        sf = SoundFont.from_file(str(path))
        assert sf is not None
        assert sf.presets is not None
        assert len(sf.presets) > 0
        assert sf.wave_data is not None


class TestSynthesizerSettings:
    def test_sample_rate_stored(self):
        s = SynthesizerSettings(44100)
        assert s.sample_rate == 44100

    def test_block_size_default(self):
        s = SynthesizerSettings(44100)
        assert s.block_size == 64

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(Exception):
            SynthesizerSettings(1000)


class TestSynthesizerRender:
    def test_note_on_and_render_produces_audio(self):
        path = _soundfont_path()
        if path is None:
            pytest.skip("examples/audio/TimGM6mb.sf2 not found")
        sf = SoundFont.from_file(str(path))
        settings = SynthesizerSettings(44100)
        synth = Synthesizer(sf, settings)
        synth.note_on(0, 60, 100)
        left = create_buffer(1024)
        right = create_buffer(1024)
        synth.render(left, right)
        # Should not be all zeros (we played a note)
        assert max(abs(x) for x in left) > 0 or max(abs(x) for x in right) > 0

    def test_render_without_note_on_is_silent(self):
        path = _soundfont_path()
        if path is None:
            pytest.skip("examples/audio/TimGM6mb.sf2 not found")
        sf = SoundFont.from_file(str(path))
        settings = SynthesizerSettings(44100)
        synth = Synthesizer(sf, settings)
        left = create_buffer(64)
        right = create_buffer(64)
        synth.render(left, right)
        for i in range(64):
            assert left[i] == 0.0
            assert right[i] == 0.0
