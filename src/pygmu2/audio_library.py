"""
AudioLibrary - resolve Strudel-style sample maps to local audio files.

AudioLibrary provides a mechanism to map sound names (e.g., "bd", "sd", "hh")
to local or remote audio files, inspired by the Strudel live coding environment
(https://strudel.cc/).

Key features:
- Load audio maps from local JSON files or remote URLs.
- Resolve sound names to file paths, with support for wildcards and indexing.
- Lazily download remote audio files to a local cache.
- Optionally convert downloaded audio (e.g., MP3, OGG) to WAV format.
- Create WavReaderPE instances directly from sound names.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import ssl
from pathlib import Path, PurePosixPath
from typing import Dict, List
from urllib import request
from urllib.error import URLError
from urllib.parse import urlparse

import soundfile as sf

from pygmu2.config import handle_error
from pygmu2.logger import get_logger
from pygmu2.wav_reader_pe import WavReaderPE

logger = get_logger(__name__)

def _create_ssl_context() -> ssl.SSLContext:
    """
    Create an SSL context for HTTPS requests.
    
    Tries to use certifi's certificate bundle if available,
    otherwise falls back to system defaults.
    """
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
        logger.debug("Using certifi certificate bundle")
        return context
    except ImportError:
        pass
    
    # Fall back to default context (uses system certificates)
    return ssl.create_default_context()


def _format_ssl_error_message(url: str, exc: Exception) -> str:
    """
    Format a helpful error message for SSL certificate errors.
    """
    base_msg = f"SSL certificate verification failed for {url!r}: {exc}"
    
    if platform.system() == "Darwin":
        # macOS-specific instructions
        return (
            f"{base_msg}\n\n"
            "This is a common issue on macOS with Python from python.org.\n"
            "To fix this, try one of these solutions:\n\n"
            "1. Run the certificate installer (recommended):\n"
            "   Open Finder, go to Applications > Python 3.x folder,\n"
            "   and double-click 'Install Certificates.command'\n\n"
            "2. Or install certifi: pip install certifi\n\n"
            "3. Or if using Homebrew Python, ensure openssl is linked:\n"
            "   brew install openssl && brew link openssl"
        )
    elif platform.system() == "Windows":
        return (
            f"{base_msg}\n\n"
            "To fix SSL certificate issues on Windows:\n"
            "1. Install certifi: pip install certifi\n"
            "2. Or update your Python installation"
        )
    else:
        # Linux and other systems
        return (
            f"{base_msg}\n\n"
            "To fix SSL certificate issues:\n"
            "1. Install certifi: pip install certifi\n"
            "2. Or install/update system CA certificates:\n"
            "   - Debian/Ubuntu: sudo apt install ca-certificates\n"
            "   - Fedora/RHEL: sudo dnf install ca-certificates"
        )

SoundValue = str | List[str]


class AudioLibrary:
    """
    Resolve Strudel-style audio maps to local files with lazy download.
    """

    def __init__(
        self,
        audio_paths: Dict[str, List[str]],
        base: str,
        cache_dir: Path | None = None,
        convert_to_wav: bool = True,
        allow_remote: bool = True,
        source_dir: Path | None = None,
    ) -> None:
        """
        Initialize an AudioLibrary.

        Args:
            audio_paths: Dictionary mapping sound names to lists of file paths/URLs.
            base: Base URL or path to prepend to relative sound paths.
            cache_dir: Directory to store downloaded files. If None, uses default
                system cache location.
            convert_to_wav: If True, convert downloaded audio to WAV format.
            allow_remote: If True, allow downloading from remote URLs.
            source_dir: Directory containing the source map file (for resolving
                relative local paths).
        """
        self._audio_paths = audio_paths
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
        path: str | Path,
        cache_dir: Path | None = None,
        convert_to_wav: bool = True,
        allow_remote: bool = True,
    ) -> "AudioLibrary":
        """
        Load an AudioLibrary from a local Strudel JSON file.

        Args:
            path: Path to the JSON file.
            cache_dir: Directory for caching downloads.
            convert_to_wav: Auto-convert to WAV.
            allow_remote: Allow remote downloads.

        Returns:
            Initialized AudioLibrary.
        """
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
        cache_dir: Path | None = None,
        convert_to_wav: bool = True,
        allow_remote: bool = True,
    ) -> "AudioLibrary":
        """
        Load an AudioLibrary from a remote URL.

        Args:
            url: URL of the JSON map file.
            cache_dir: Directory for caching downloads.
            convert_to_wav: Auto-convert to WAV.
            allow_remote: Must be True to function.

        Returns:
            Initialized AudioLibrary.
        """
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
        Resolve a sound name to a local file path.

        This method:
        1. Selects the sound entry matching `name`.
        2. Picks the specific file at `index`.
        3. Downloads the file if it's remote and not cached.
        4. Converts to WAV if requested.

        Args:
            name: Sound name (can contain '?' wildcards).
            index: Index into the list of files for this sound.

        Returns:
            Absolute path to the local audio file.
        """
        name = self._maybe_select_pattern(name)
        rel_path = self._select_sound_path(name, index)
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
        Create a WavReaderPE for a sound name.

        Args:
            name: Sound name.
            index: Variant index.

        Returns:
            Configured WavReaderPE.
        """
        return WavReaderPE(self.resolve(name, index=index))

    def print_keys(self, columns: int = 3, width: int = 25) -> None:
        """
        Print all available sound keys in a grid format.

        Args:
            columns: Number of columns.
            width: Width of each column.
        """
        keys = sorted(self._audio_paths.keys())
        if not keys:
            print("(no sounds)")
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
        cache_dir: Path | None,
        convert_to_wav: bool,
        allow_remote: bool,
        source_dir: Path | None,
        base_override: str | None = None,
    ) -> "AudioLibrary":
        """Internal helper to create AudioLibrary from parsed JSON data."""
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

        audio_paths = cls._normalize_audio_paths(mapping)
        return cls(
            audio_paths,
            base=base,
            cache_dir=cache_dir,
            convert_to_wav=convert_to_wav,
            allow_remote=allow_remote,
            source_dir=source_dir,
        )

    @staticmethod
    def _normalize_audio_paths(mapping: Dict[str, SoundValue]) -> Dict[str, List[str]]:
        """Ensure all sound entries are lists of strings."""
        audio_paths: Dict[str, List[str]] = {}
        for key, value in mapping.items():
            if isinstance(value, list):
                audio_paths[str(key)] = [str(item) for item in value]
            elif isinstance(value, str):
                audio_paths[str(key)] = [value]
            else:
                handle_error(
                    f"Sound map value for {key!r} must be str or list.",
                    fatal=True,
                )
        return audio_paths

    def _select_sound_path(self, name: str, index: int) -> str:
        """Get the relative path for a sound by name and index."""
        if name not in self._audio_paths:
            handle_error(f"Sound name not found: {name!r}", fatal=True)
        values = self._audio_paths[name]
        if not values:
            handle_error(f"No paths defined for sound: {name!r}", fatal=True)
        if index < 0 or index >= len(values):
            handle_error(
                f"Index {index} out of range for sound {name!r}.",
                fatal=True,
            )
        return values[index]

    def _maybe_select_pattern(self, name: str) -> str:
        """Resolve wildcard patterns (e.g. 'foo?') to a specific key."""
        if "?" not in name:
            return name

        escaped = re.escape(name)
        pattern = "^" + escaped.replace(r"\?", ".*") + "$"
        matches = [key for key in self._audio_paths.keys() if re.match(pattern, key)]
        if not matches:
            handle_error(f"No sounds match pattern: {name!r}", fatal=True)
        return random.choice(matches)

    @staticmethod
    def _normalize_relative_path(rel_path: str) -> str:
        """Sanitize relative path and prevent directory traversal."""
        normalized = rel_path.replace("\\", "/")
        posix_path = PurePosixPath(normalized)
        if posix_path.is_absolute() or ".." in posix_path.parts:
            handle_error(
                f"Invalid relative path in audio map: {rel_path!r}",
                fatal=True,
            )
        return posix_path.as_posix()

    @staticmethod
    def _normalize_base(base: str) -> str:
        """Normalize base URL/path, handling github: shortcut."""
        if base.startswith("github:"):
            base = "https://raw.githubusercontent.com/" + base[len("github:") :]
        if base and (base.startswith("http://") or base.startswith("https://")):
            if not base.endswith("/"):
                base += "/"
        return base

    @staticmethod
    def _is_remote(base: str) -> bool:
        """Check if base path is a URL."""
        return base.startswith("http://") or base.startswith("https://")

    def _local_base_path(self, base: str) -> Path:
        """Resolve local base path relative to source directory or CWD."""
        if base:
            base_path = Path(base).expanduser()
            if not base_path.is_absolute() and self._source_dir is not None:
                base_path = self._source_dir / base_path
            return base_path.resolve()
        if self._source_dir is not None:
            return self._source_dir.resolve()
        return Path.cwd()

    def _cache_path(self, base: str, rel_posix: str) -> Path:
        """Determine local cache path for a remote file."""
        base_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]
        rel_parts = PurePosixPath(rel_posix).parts
        return self._cache_dir / base_hash / Path(*rel_parts)

    def _ensure_downloaded(self, url: str, dest: Path) -> Path:
        """Download file if not already cached."""
        if dest.exists() and dest.stat().st_size > 0:
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        logger.info(f"Downloading sound: {url}")
        try:
            ssl_context = _create_ssl_context()
            with request.urlopen(url, context=ssl_context) as response, tmp_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp_path.replace(dest)
        except URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                handle_error(
                    _format_ssl_error_message(url, exc.reason), fatal=True
                )
            handle_error(
                f"Failed to download sound {url!r}: {exc}", fatal=True
            )
        except Exception as exc:
            handle_error(
                f"Failed to download sound {url!r}: {exc}", fatal=True
            )
        return dest

    def _maybe_convert_to_wav(self, path: Path) -> str:
        """Convert audio file to WAV if configured and needed."""
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
        """Get OS-specific default cache directory."""
        home = Path.home()
        if os.name == "nt":
            local_app_data = os.environ.get("LOCALAPPDATA", str(home))
            return Path(local_app_data) / "pygmu2" / "strudel"
        if os.name == "posix" and "darwin" in os.uname().sysname.lower():
            return home / "Library" / "Caches" / "pygmu2" / "strudel"
        return home / ".cache" / "pygmu2" / "strudel"

    @staticmethod
    def _ensure_remote_json(url: str, cache_root: Path) -> Path:
        """Download remote JSON map file if not cached."""
        cache_root.mkdir(parents=True, exist_ok=True)
        base_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        json_path = cache_root / "maps" / f"{base_hash}.json"
        if json_path.exists() and json_path.stat().st_size > 0:
            return json_path

        json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = json_path.with_suffix(".part")
        logger.info(f"Downloading Strudel map: {url}")
        try:
            ssl_context = _create_ssl_context()
            with request.urlopen(url, context=ssl_context) as response, tmp_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp_path.replace(json_path)
        except URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                handle_error(
                    _format_ssl_error_message(url, exc.reason), fatal=True
                )
            handle_error(
                f"Failed to download Strudel map {url!r}: {exc}", fatal=True
            )
        except Exception as exc:
            handle_error(
                f"Failed to download Strudel map {url!r}: {exc}", fatal=True
            )
        return json_path

    @staticmethod
    def _url_base_dir(url: str) -> str:
        """Extract base directory URL from a full file URL."""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return ""
        base_path = parsed.path.rsplit("/", 1)[0] + "/"
        return f"{parsed.scheme}://{parsed.netloc}{base_path}"
