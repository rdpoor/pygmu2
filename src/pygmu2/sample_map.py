"""
SampleMap - resolve Strudel-style sample maps to local audio files.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Union
from urllib import request
from urllib.parse import urlparse

import soundfile as sf

from pygmu2.config import handle_error
from pygmu2.logger import get_logger
from pygmu2.wav_reader_pe import WavReaderPE

logger = get_logger(__name__)

SampleValue = Union[str, List[str]]


class SampleMap:
    """
    Resolve Strudel-style sample maps to local files with lazy download.
    """

    def __init__(
        self,
        samples: Dict[str, List[str]],
        base: str,
        cache_dir: Optional[Path] = None,
        convert_to_wav: bool = True,
        allow_remote: bool = True,
        source_dir: Optional[Path] = None,
    ) -> None:
        self._samples = samples
        self._base = base
        self._cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else self._default_cache_dir()
        )
        self._convert_to_wav = convert_to_wav
        self._allow_remote = allow_remote
        self._source_dir = source_dir

    @classmethod
    def from_strudel_json(
        cls,
        path: Union[str, Path],
        cache_dir: Optional[Path] = None,
        convert_to_wav: bool = True,
        allow_remote: bool = True,
    ) -> "SampleMap":
        json_path = Path(path).expanduser()
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return cls._from_strudel_data(
            data,
            cache_dir=cache_dir,
            convert_to_wav=convert_to_wav,
            allow_remote=allow_remote,
            source_dir=json_path.parent,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        cache_dir: Optional[Path] = None,
        convert_to_wav: bool = True,
        allow_remote: bool = True,
    ) -> "SampleMap":
        cache_root = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else cls._default_cache_dir()
        )
        json_path = cls._ensure_remote_json(url, cache_root)
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        base_override = cls._url_base_dir(url)
        return cls._from_strudel_data(
            data,
            cache_dir=cache_root,
            convert_to_wav=convert_to_wav,
            allow_remote=allow_remote,
            source_dir=None,
            base_override=base_override,
        )

    def resolve(self, name: str, index: int = 0) -> str:
        """
        Resolve a sample name to a local file path.

        If the sample is remote, it is downloaded on demand to the cache.
        """
        name = self._maybe_select_pattern(name)
        rel_path = self._select_sample_path(name, index)
        rel_posix = self._normalize_relative_path(rel_path)
        base = self._normalize_base(self._base)

        if self._is_remote(base):
            if not self._allow_remote:
                handle_error(
                    f"Remote base not allowed: {self._base!r}", fatal=True
                )
            url = base + rel_posix
            cached = self._cache_path(base, rel_posix)
            local_path = self._ensure_downloaded(url, cached)
            return self._maybe_convert_to_wav(local_path)

        base_path = self._local_base_path(base)
        return str(base_path / rel_posix)

    def reader(self, name: str, index: int = 0) -> WavReaderPE:
        """
        Create a WavReaderPE for a sample name.
        """
        return WavReaderPE(self.resolve(name, index=index))

    def print_keys(self, columns: int = 3, width: int = 25) -> None:
        """
        Print all sample keys formatted in columns.
        """
        keys = sorted(self._samples.keys())
        if not keys:
            print("(no samples)")
            return

        col_count = max(1, columns)
        col_width = max(1, width)
        for idx, key in enumerate(keys):
            end = "\n" if (idx + 1) % col_count == 0 else ""
            print(key.ljust(col_width), end=end)
        if len(keys) % col_count != 0:
            print()

    @classmethod
    def _from_strudel_data(
        cls,
        data: dict,
        cache_dir: Optional[Path],
        convert_to_wav: bool,
        allow_remote: bool,
        source_dir: Optional[Path],
        base_override: Optional[str] = None,
    ) -> "SampleMap":
        if not isinstance(data, dict):
            handle_error("strudel.json must contain a top-level object.", fatal=True)

        base = data.get("base") or data.get("_base") or ""
        if not base and base_override:
            base = base_override

        if "samples" in data and isinstance(data["samples"], dict):
            mapping = data["samples"]
        else:
            mapping = {
                key: value
                for key, value in data.items()
                if key not in ("base", "_base") and not str(key).startswith("_")
            }

        samples = cls._normalize_samples(mapping)
        return cls(
            samples,
            base=base,
            cache_dir=cache_dir,
            convert_to_wav=convert_to_wav,
            allow_remote=allow_remote,
            source_dir=source_dir,
        )

    @staticmethod
    def _normalize_samples(mapping: Dict[str, SampleValue]) -> Dict[str, List[str]]:
        samples: Dict[str, List[str]] = {}
        for key, value in mapping.items():
            if isinstance(value, list):
                samples[str(key)] = [str(item) for item in value]
            elif isinstance(value, str):
                samples[str(key)] = [value]
            else:
                handle_error(
                    f"Sample map value for {key!r} must be str or list.",
                    fatal=True,
                )
        return samples

    def _select_sample_path(self, name: str, index: int) -> str:
        if name not in self._samples:
            handle_error(f"Sample name not found: {name!r}", fatal=True)
        values = self._samples[name]
        if not values:
            handle_error(f"No paths defined for sample: {name!r}", fatal=True)
        if index < 0 or index >= len(values):
            handle_error(
                f"Index {index} out of range for sample {name!r}.",
                fatal=True,
            )
        return values[index]

    def _maybe_select_pattern(self, name: str) -> str:
        if "?" not in name:
            return name

        escaped = re.escape(name)
        pattern = "^" + escaped.replace(r"\?", ".*") + "$"
        matches = [key for key in self._samples.keys() if re.match(pattern, key)]
        if not matches:
            handle_error(f"No samples match pattern: {name!r}", fatal=True)
        return random.choice(matches)

    @staticmethod
    def _normalize_relative_path(rel_path: str) -> str:
        normalized = rel_path.replace("\\", "/")
        posix_path = PurePosixPath(normalized)
        if posix_path.is_absolute() or ".." in posix_path.parts:
            handle_error(
                f"Invalid relative path in sample map: {rel_path!r}",
                fatal=True,
            )
        return posix_path.as_posix()

    @staticmethod
    def _normalize_base(base: str) -> str:
        if base.startswith("github:"):
            base = "https://raw.githubusercontent.com/" + base[len("github:") :]
        if base and (base.startswith("http://") or base.startswith("https://")):
            if not base.endswith("/"):
                base += "/"
        return base

    @staticmethod
    def _is_remote(base: str) -> bool:
        return base.startswith("http://") or base.startswith("https://")

    def _local_base_path(self, base: str) -> Path:
        if base:
            base_path = Path(base).expanduser()
            if not base_path.is_absolute() and self._source_dir is not None:
                base_path = self._source_dir / base_path
            return base_path.resolve()
        if self._source_dir is not None:
            return self._source_dir.resolve()
        return Path.cwd()

    def _cache_path(self, base: str, rel_posix: str) -> Path:
        base_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]
        rel_parts = PurePosixPath(rel_posix).parts
        return self._cache_dir / base_hash / Path(*rel_parts)

    def _ensure_downloaded(self, url: str, dest: Path) -> Path:
        if dest.exists() and dest.stat().st_size > 0:
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        logger.info(f"Downloading sample: {url}")
        try:
            with request.urlopen(url) as response, tmp_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp_path.replace(dest)
        except Exception as exc:
            handle_error(
                f"Failed to download sample {url!r}: {exc}", fatal=True
            )
        return dest

    def _maybe_convert_to_wav(self, path: Path) -> str:
        if not self._convert_to_wav:
            return str(path)
        if path.suffix.lower() == ".wav":
            return str(path)

        wav_path = path.with_suffix(".wav")
        if wav_path.exists() and wav_path.stat().st_size > 0:
            return str(wav_path)

        try:
            data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            sf.write(wav_path, data, sample_rate)
            logger.info(f"Converted {path} to {wav_path}")
            return str(wav_path)
        except Exception as exc:
            if handle_error(
                f"Failed to convert {path} to WAV: {exc}", fatal=False
            ):
                return str(path)
            return str(path)

    @staticmethod
    def _default_cache_dir() -> Path:
        home = Path.home()
        if os.name == "nt":
            local_app_data = os.environ.get("LOCALAPPDATA", str(home))
            return Path(local_app_data) / "pygmu2" / "strudel"
        if os.name == "posix" and "darwin" in os.uname().sysname.lower():
            return home / "Library" / "Caches" / "pygmu2" / "strudel"
        return home / ".cache" / "pygmu2" / "strudel"

    @staticmethod
    def _ensure_remote_json(url: str, cache_root: Path) -> Path:
        cache_root.mkdir(parents=True, exist_ok=True)
        base_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        json_path = cache_root / "maps" / f"{base_hash}.json"
        if json_path.exists() and json_path.stat().st_size > 0:
            return json_path

        json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = json_path.with_suffix(".part")
        logger.info(f"Downloading Strudel map: {url}")
        try:
            with request.urlopen(url) as response, tmp_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp_path.replace(json_path)
        except Exception as exc:
            handle_error(
                f"Failed to download Strudel map {url!r}: {exc}", fatal=True
            )
        return json_path

    @staticmethod
    def _url_base_dir(url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return ""
        base_path = parsed.path.rsplit("/", 1)[0] + "/"
        return f"{parsed.scheme}://{parsed.netloc}{base_path}"

