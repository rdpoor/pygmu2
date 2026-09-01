"""
Tests for AudioLibrary — device-free and network-free.

Remote fetches are stubbed at request.urlopen; local resolution, map
parsing, wildcard selection, path sanitisation, caching, and WAV
conversion run against real files in tmp_path.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

import pygmu2.audio_library as al
from pygmu2.audio_library import AudioLibrary
from pygmu2.wav_reader_pe import WavReaderPE


def _write_wav(path: Path, sample_rate: int = 44100, n: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((n, 1), dtype=np.float32), sample_rate)


def _strudel_json(tmp_path: Path, data: dict, name: str = "map.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class _FakeResponse:
    """Minimal urlopen context manager serving one payload."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self._served = False

    def read(self, n: int = -1) -> bytes:
        if self._served:
            return b""
        self._served = True
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestMapParsing:
    def test_samples_form(self, tmp_path):
        path = _strudel_json(
            tmp_path,
            {
                "base": "snd",
                "samples": {"bd": "bd/kick.wav", "sd": ["s1.wav", "s2.wav"]},
            },
        )
        lib = AudioLibrary.from_strudel_json(path, cache_dir=tmp_path / "cache")
        assert lib._audio_paths == {"bd": ["bd/kick.wav"], "sd": ["s1.wav", "s2.wav"]}
        assert lib._base == "snd"

    def test_top_level_form_skips_meta_keys(self, tmp_path):
        path = _strudel_json(
            tmp_path,
            {"_base": "snd", "_comment": "ignore me", "hh": "hh.wav"},
        )
        lib = AudioLibrary.from_strudel_json(path, cache_dir=tmp_path / "cache")
        assert lib._audio_paths == {"hh": ["hh.wav"]}
        assert lib._base == "snd"

    def test_non_dict_map_raises(self, tmp_path):
        path = _strudel_json(tmp_path, {"bd": 42})
        with pytest.raises(RuntimeError, match="must be str or list"):
            AudioLibrary.from_strudel_json(path, cache_dir=tmp_path / "cache")


class TestLocalResolve:
    def _lib(self, tmp_path, paths, base=""):
        data = {"base": base, "samples": paths}
        map_path = _strudel_json(tmp_path, data)
        return AudioLibrary.from_strudel_json(map_path, cache_dir=tmp_path / "cache")

    def test_resolve_relative_to_source_dir(self, tmp_path):
        _write_wav(tmp_path / "snd" / "kick.wav")
        lib = self._lib(tmp_path, {"bd": "kick.wav"}, base="snd")
        resolved = Path(lib.resolve("bd"))
        assert resolved == (tmp_path / "snd" / "kick.wav").resolve()

    def test_resolve_index_selects_variant(self, tmp_path):
        lib = self._lib(tmp_path, {"sd": ["a.wav", "b.wav"]})
        assert Path(lib.resolve("sd", index=1)).name == "b.wav"

    def test_unknown_name_raises(self, tmp_path):
        lib = self._lib(tmp_path, {"bd": "kick.wav"})
        with pytest.raises(RuntimeError, match="Sound name not found"):
            lib.resolve("nope")

    def test_index_out_of_range_raises(self, tmp_path):
        lib = self._lib(tmp_path, {"bd": ["kick.wav"]})
        with pytest.raises(RuntimeError, match="out of range"):
            lib.resolve("bd", index=1)

    def test_traversal_in_map_raises(self, tmp_path):
        lib = self._lib(tmp_path, {"evil": "../outside.wav"})
        with pytest.raises(RuntimeError, match="Invalid relative path"):
            lib.resolve("evil")

    def test_wildcard_selects_among_matches(self, tmp_path):
        lib = self._lib(tmp_path, {"bd_a": "a.wav", "bd_b": "b.wav", "sd": "s.wav"})
        with patch.object(al.random, "choice", side_effect=lambda xs: sorted(xs)[0]):
            assert Path(lib.resolve("bd?")).name == "a.wav"

    def test_wildcard_no_match_raises(self, tmp_path):
        lib = self._lib(tmp_path, {"bd": "a.wav"})
        with pytest.raises(RuntimeError, match="No sounds match"):
            lib.resolve("zz?")

    def test_reader_returns_wav_reader_pe(self, tmp_path):
        _write_wav(tmp_path / "kick.wav")
        lib = self._lib(tmp_path, {"bd": "kick.wav"})
        assert isinstance(lib.reader("bd"), WavReaderPE)


class TestNormalization:
    def test_base_github_shortcut(self):
        base = AudioLibrary._normalize_base("github:owner/repo/main/snd")
        assert base == "https://raw.githubusercontent.com/owner/repo/main/snd/"

    def test_base_url_gets_trailing_slash(self):
        assert AudioLibrary._normalize_base("https://x.org/a") == "https://x.org/a/"
        assert AudioLibrary._normalize_base("plain/dir") == "plain/dir"

    def test_relative_path_backslashes(self):
        assert AudioLibrary._normalize_relative_path(r"a\b\c.wav") == "a/b/c.wav"

    def test_url_base_dir(self):
        assert (
            AudioLibrary._url_base_dir("https://x.org/maps/m.json")
            == "https://x.org/maps/"
        )
        assert AudioLibrary._url_base_dir("not a url") == ""


class TestRemoteResolve:
    BASE = "https://example.org/snd"

    def _lib(self, tmp_path, **kwargs):
        return AudioLibrary(
            {"bd": ["kick.wav"]},
            base=self.BASE,
            cache_dir=tmp_path / "cache",
            **kwargs,
        )

    def _wav_bytes(self) -> bytes:
        import io

        buf = io.BytesIO()
        sf.write(buf, np.zeros((64, 1), dtype=np.float32), 44100, format="WAV")
        return buf.getvalue()

    def test_remote_disallowed_raises(self, tmp_path):
        lib = self._lib(tmp_path, allow_remote=False)
        with pytest.raises(RuntimeError, match="Remote base not allowed"):
            lib.resolve("bd")

    def test_remote_downloads_then_caches(self, tmp_path):
        lib = self._lib(tmp_path)
        with patch.object(
            al.request, "urlopen", return_value=_FakeResponse(self._wav_bytes())
        ) as fake:
            first = lib.resolve("bd")
        assert Path(first).exists()
        assert fake.call_count == 1
        requested_url = fake.call_args[0][0]
        assert requested_url == f"{self.BASE}/kick.wav"
        # second resolve is served from cache — no network call
        with patch.object(al.request, "urlopen") as fake2:
            second = lib.resolve("bd")
        assert second == first
        fake2.assert_not_called()

    def test_download_failure_raises(self, tmp_path):
        from urllib.error import URLError

        lib = self._lib(tmp_path)
        with patch.object(al.request, "urlopen", side_effect=URLError("no route")):
            with pytest.raises(RuntimeError, match="Failed to download"):
                lib.resolve("bd")


class TestWavConversion:
    def test_converts_non_wav_to_wav(self, tmp_path):
        flac = tmp_path / "tone.flac"
        data = (0.1 * np.sin(np.linspace(0, 40, 256))).astype(np.float32)
        sf.write(flac, data.reshape(-1, 1), 44100)
        lib = AudioLibrary({}, base="", cache_dir=tmp_path / "cache")
        out = Path(lib._maybe_convert_to_wav(flac))
        assert out.suffix == ".wav"
        round_trip, sr = sf.read(out, dtype="float32")
        assert sr == 44100
        np.testing.assert_allclose(round_trip, data, atol=1e-4)

    def test_wav_passthrough(self, tmp_path):
        wav = tmp_path / "x.wav"
        _write_wav(wav)
        lib = AudioLibrary({}, base="", cache_dir=tmp_path / "cache")
        assert lib._maybe_convert_to_wav(wav) == str(wav)

    def test_conversion_disabled_passthrough(self, tmp_path):
        flac = tmp_path / "x.flac"
        sf.write(flac, np.zeros((16, 1), dtype=np.float32), 44100)
        lib = AudioLibrary(
            {}, base="", cache_dir=tmp_path / "cache", convert_to_wav=False
        )
        assert lib._maybe_convert_to_wav(flac) == str(flac)


class TestFromUrl:
    def test_map_downloaded_and_base_derived_from_url(self, tmp_path):
        payload = json.dumps({"samples": {"bd": "kick.wav"}}).encode()
        with patch.object(al.request, "urlopen", return_value=_FakeResponse(payload)):
            lib = AudioLibrary.from_url(
                "https://example.org/maps/strudel.json", cache_dir=tmp_path / "cache"
            )
        assert lib._audio_paths == {"bd": ["kick.wav"]}
        # map had no base of its own -> derived from the map URL's directory
        assert lib._base == "https://example.org/maps/"
        # the map file itself was cached
        with patch.object(al.request, "urlopen") as fake:
            again = AudioLibrary.from_url(
                "https://example.org/maps/strudel.json", cache_dir=tmp_path / "cache"
            )
        fake.assert_not_called()
        assert again._audio_paths == lib._audio_paths
