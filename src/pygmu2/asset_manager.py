"""
AssetManager - download and cache audio assets from remote sources.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import fnmatch
import json
import os
import ssl
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
from urllib import request
from urllib.error import URLError
from urllib.parse import urlencode, quote

from pygmu2.logger import get_logger

logger = get_logger(__name__)


"""Get OS-specific cache base directory.

Windows: LOCALAPPDATA
macOS: ~/Library/Caches
Linux: ~/.cache
"""
def _default_cache_base() -> Path:
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", str(Path.home()))
        return Path(local_app_data)
    if os.name == "posix" and "darwin" in os.uname().sysname.lower():
        return Path.home() / "Library" / "Caches"
    return Path.home() / ".cache"


"""Get OS-specific config base directory.

Windows: LOCALAPPDATA
macOS: ~/Library/Application Support
Linux: ~/.config
"""
def _default_config_base() -> Path:
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", str(Path.home()))
        return Path(local_app_data)
    if os.name == "posix" and "darwin" in os.uname().sysname.lower():
        return Path.home() / "Library" / "Application Support"
    return Path.home() / ".config"

class AssetLoadFailed(RuntimeError):
    pass


class AssetNotFound(FileNotFoundError):
    pass


class AssetLoader(ABC):
    """
    Abstract base class for remote asset loaders.

    Implementations should provide:
        - list_remote_assets(): return alphabetically sorted matching names
        - load_remote_asset(): download first matching asset into cache_dir
    """

    @staticmethod
    def _create_ssl_context() -> ssl.SSLContext:
        """
        Create an SSL context for HTTPS requests.

        Tries to use certifi's certificate bundle if available, otherwise falls
        back to system defaults.
        """
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
            logger.debug("Using certifi certificate bundle")
            return context
        except ImportError:
            pass
        return ssl.create_default_context()

    @abstractmethod
    def list_remote_assets(self, wildcard_spec: str) -> list[str]:
        """
        List all remote assets that match the wildcard specification.

        Returns:
            A list of matching remote asset names, sorted alphabetically.
        """
        raise NotImplementedError

    @abstractmethod
    def load_remote_asset(self, wildcard_spec: str, cache_dir: Path) -> Path | None:
        """
        Download the first matching remote asset into cache_dir.

        Matching is performed using list_remote_assets(wildcard_spec), and the
        first match is selected after alphabetic sorting.
        """
        raise NotImplementedError


class AssetManager:

    def __init__(
        self,
        cache_dir: Path | None = None,
        asset_loader: AssetLoader | None = None,
    ):
        """
        Return a locally cached asset, loading it from a remote site if needed.

        Args:
            cache_dir: Directory to store downloaded assets.  If None, uses the
                default system cache.
            asset_loader: An AssetLoader object to invoke if the asset is not
                already present in the cache.  If omitted, don't allow remote
                loading.
        """
        self._cache_dir = cache_dir if cache_dir else self._default_cache_dir()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._asset_loader = asset_loader

    def load_asset(self, asset_specification: str, force: bool = False) -> Path:
        """
        Locate asset_specification in local cache, or load it from remote cache.

        Args:
            asset_specification: String that names the asset.  May include
            wildcards.
        Returns:
            path to locally cached asset.
        Raises:
            AssetLoadFailed if remote load failed
            AssetNotFound if nothing matched asset_specification
        """
        if force:
            for cached in self.list_cached_assets(asset_specification):
                try:
                    cached.unlink()
                except FileNotFoundError:
                    pass

        resolved_name = None if force else self.locate_local_asset(asset_specification)
        if resolved_name is None:
            # asset not in cache or force reload requested
            if self._asset_loader is not None:
                # raises AssetLoadFailed on system error
                resolved_name = self._asset_loader.load_remote_asset(
                    asset_specification,
                    self._cache_dir)
            else:
                raise AssetLoadFailed(
                    "remote asset loading is not configured for this AssetManager"
                )
        if resolved_name is None:
            # failed to find named asset
            raise AssetNotFound(
                f"could not find asset named {asset_specification}")
        return resolved_name

    def list_remote_assets(self, asset_specification: str) -> list[Path]:
        """
        List remote assets matching the asset specification.

        Returns relative paths (cache-relative) for each matching remote asset.
        """
        if self._asset_loader is None:
            raise AssetLoadFailed(
                "remote asset loading is not configured for this AssetManager"
            )
        return [
            Path(name)
            for name in self._asset_loader.list_remote_assets(asset_specification)
        ]

    def list_cached_assets(self, asset_specification: str) -> list[Path]:
        """
        List cached assets matching the asset specification.
        """
        return sorted(
            (p for p in self._cache_dir.glob(asset_specification) if p.exists()),
            key=lambda p: str(p).casefold(),
        )

    def has_cached_asset(self, asset_specification: str) -> bool:
        """
        Return True if any cached asset matches the specification.
        """
        return bool(self.list_cached_assets(asset_specification))

    def cache_path(self) -> Path:
        """
        Return the cache directory used by this AssetManager.
        """
        return self._cache_dir

    def clear_cache(self) -> None:
        """
        Delete all cached assets. Recreates the cache directory afterward.
        """
        resolved = self._cache_dir.resolve()
        expected = self._default_cache_dir().resolve()
        if resolved != expected:
            raise AssetLoadFailed(
                f"refusing to clear non-default cache directory: {resolved}"
            )

        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def locate_local_asset(self, asset_specification: str) -> Path | None:
        """
        Resolve an asset specification against the local cache.

        Returns:
            Path to the first matching local asset, or None if not found.

        Examples:
            Absolute path:
                "/Users/r/Music/oneshot.wav"

            Cache-relative filename:
                "drums/kick.wav"

            Wildcard match (first match returned):
                "loops/*.wav"
        """
        candidate = Path(asset_specification)
        if candidate.is_absolute():
            return candidate if candidate.exists() else None

        matches = sorted(
            p for p in self._cache_dir.glob(asset_specification) if p.exists()
        )
        if matches:
            return matches[0]
        return None

    @staticmethod
    def _default_cache_dir() -> Path:
        """Get OS-specific default cache directory."""
        return _default_cache_base() / "pygmu2" / "asset_cache"


class GoogleDriveAssetLoader(AssetLoader):
    """
    AssetLoader backed by a single Google Drive folder.

    Notes:
        OAuth is required to access private or user-shared files. API key
        authentication only works for files that are public or shared broadly
        enough to be accessible without user OAuth.

        Finding a Google Drive folder ID:
            Open the folder in your browser and copy the ID from the URL.
            Example URL:
                https://drive.google.com/drive/folders/1AbCDeFgHiJKlMnOp
            The folder ID is the long token after "/folders/".
    """
    def __init__(
        self,
        folder_id: str,
        oauth_client_secrets: Path | None = None,
        token_path: Path | None = None,
        scopes: list[str] | None = None,
        api_key_env_var: str | None = None,
    ):
        """
        Initialize a GoogleDriveAssetLoader.

        Args:
            folder_id: Google Drive folder ID to treat as the asset root.
            oauth_client_secrets: Path to OAuth client secrets JSON. Required for
                accessing private or user-shared files.
            token_path: Optional path for the cached OAuth token (refresh token).
                If omitted, an OS-specific config directory is used.
            scopes: Optional OAuth scopes. Defaults to Drive read-only.
            api_key_env_var: Optional environment variable name containing a Drive
                API key. Used for public/link-shared files when OAuth is not provided.
        """
        self._folder_id = folder_id
        self._oauth_client_secrets = oauth_client_secrets
        if self._oauth_client_secrets is None:
            default_secrets = _default_config_base() / "pygmu2" / "client_secrets.json"
            if default_secrets.exists():
                self._oauth_client_secrets = default_secrets
        self._token_path = token_path or (self._default_token_dir() / "gdrive_token.json")
        self._scopes = scopes or [
            "https://www.googleapis.com/auth/drive.readonly"
        ]
        self._api_key_env_var = api_key_env_var

        if self._oauth_client_secrets is None and self._api_key_env_var is None:
            raise AssetLoadFailed(
                "GoogleDriveAssetLoader requires oauth_client_secrets or api_key_env_var. "
                f"Expected default secrets at "
                f"{_default_config_base() / 'pygmu2' / 'client_secrets.json'}"
            )

    def list_remote_assets(self, wildcard_spec: str) -> list[str]:
        logger.debug(
            "Listing Google Drive assets for folder_id=%s spec=%r",
            self._folder_id,
            wildcard_spec,
        )
        assets = self._list_remote_assets_with_ids(wildcard_spec)
        logger.info(
            "Found %d matching Google Drive assets for spec=%r",
            len(assets),
            wildcard_spec,
        )
        return [name for name, _ in assets]

    def load_remote_asset(self, wildcard_spec: str, cache_dir: Path) -> Path | None:
        logger.debug(
            "Loading Google Drive asset for folder_id=%s spec=%r",
            self._folder_id,
            wildcard_spec,
        )
        assets = self._list_remote_assets_with_ids(wildcard_spec)
        if not assets:
            logger.warning(
                "No Google Drive assets matched spec=%r in folder_id=%s",
                wildcard_spec,
                self._folder_id,
            )
            return None

        name, file_id = assets[0]
        local_path = cache_dir / name

        candidate = Path(name)
        if candidate.is_absolute():
            raise AssetLoadFailed(f"refusing to write absolute path: {name}")
        if ".." in candidate.parts:
            raise AssetLoadFailed(f"refusing to write path with '..': {name}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            logger.info("Using cached asset at %s", local_path)
            return local_path

        logger.info("Downloading %r to %s", name, local_path)
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        try:
            self._download_file(url, local_path)
        except AssetLoadFailed as exc:
            logger.error("Download failed for %r: %s", name, exc)
            raise

        return local_path

    def _list_remote_assets_with_ids(self, wildcard_spec: str) -> list[tuple[str, str]]:
        # We intentionally avoid caching listings; AssetManager.load_asset only reaches
        # this path on local cache misses, so Drive calls are already limited.
        prefix, pattern = self._split_spec(wildcard_spec)
        start_folder_id, start_prefix = self._resolve_prefix_folder(prefix)
        if start_folder_id is None:
            return []

        results: list[tuple[str, str]] = []
        for item in self._list_folder_items(start_folder_id):
            name = item.get("name")
            file_id = item.get("id")
            mime_type = item.get("mimeType", "")
            if not name or not file_id:
                continue
            if mime_type == "application/vnd.google-apps.folder":
                continue
            if mime_type.startswith("application/vnd.google-apps"):
                # Skip Google Docs / Sheets / etc. (binary-only support).
                continue
            if fnmatch.fnmatchcase(name, pattern):
                rel_name = f"{start_prefix}{name}" if start_prefix else name
                results.append((rel_name, file_id))

        results.sort(key=lambda x: x[0].casefold())
        return results

    def _list_folder_items(self, folder_id: str) -> list[dict]:
        base_url = "https://www.googleapis.com/drive/v3/files"
        query_base = f"'{folder_id}' in parents and trashed = false"
        fields = "nextPageToken,files(id,name,mimeType)"

        page_token = None
        items: list[dict] = []

        while True:
            params = {
                "q": query_base,
                "fields": fields,
                "pageSize": 1000,
            }
            if page_token:
                params["pageToken"] = page_token

            payload = self._get_json(base_url, params)
            items.extend(payload.get("files", []))

            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        return items

    def _split_spec(self, wildcard_spec: str) -> tuple[str, str]:
        """
        Split wildcard spec into a fixed folder prefix and filename pattern.

        Example:
            "a/b/c*.wav" -> ("a/b/", "c*.wav")
            "*.wav" -> ("", "*.wav")
        """
        wildcard_spec = wildcard_spec.lstrip("/")
        parts = wildcard_spec.split("/")
        if len(parts) == 1:
            return "", parts[0]
        return "/".join(parts[:-1]) + "/", parts[-1]

    def _resolve_prefix_folder(self, prefix: str) -> tuple[str | None, str]:
        """
        Resolve a folder prefix like "a/b/" to a Drive folder ID.

        Returns:
            (folder_id, normalized_prefix_with_trailing_slash)
            If not found, (None, "").
        """
        if not prefix:
            return self._folder_id, ""

        current_folder = self._folder_id
        segments = [p for p in prefix.split("/") if p]
        for segment in segments:
            next_folder = self._find_child_folder(current_folder, segment)
            if next_folder is None:
                return None, ""
            current_folder = next_folder
        return current_folder, "/".join(segments) + "/"

    def _find_child_folder(self, parent_folder_id: str, name: str) -> str | None:
        """
        Find a direct child folder by name under parent_folder_id.
        """
        for item in self._list_folder_items(parent_folder_id):
            if item.get("mimeType") != "application/vnd.google-apps.folder":
                continue
            if item.get("name") == name:
                return item.get("id")
        return None

    def _get_json(self, base_url: str, params: dict) -> dict:
        if self._oauth_client_secrets is not None:
            session = self._get_authorized_session()
            response = session.get(base_url, params=params)
            if response.status_code >= 400:
                raise AssetLoadFailed(
                    f"Drive API request failed ({response.status_code}): {response.text}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise AssetLoadFailed(f"invalid JSON from Drive API: {exc}") from exc

        api_key = self._require_api_key()
        params = dict(params)
        params["key"] = api_key
        url = f"{base_url}?{urlencode(params)}"
        try:
            ssl_context = self._create_ssl_context()
            with request.urlopen(url, context=ssl_context) as response:
                return json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise AssetLoadFailed(f"failed to list assets: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise AssetLoadFailed(f"invalid JSON from Drive API: {exc}") from exc

    def _download_file(self, base_url: str, dest: Path) -> None:
        if self._oauth_client_secrets is not None:
            session = self._get_authorized_session()
            response = session.get(base_url, params={"alt": "media"}, stream=True)
            if response.status_code >= 400:
                raise AssetLoadFailed(
                    f"Drive download failed ({response.status_code}): {response.text}"
                )
            with dest.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return

        api_key = self._require_api_key()
        query = urlencode({"alt": "media", "key": api_key})
        download_url = f"{base_url}?{query}"
        try:
            ssl_context = self._create_ssl_context()
            with request.urlopen(download_url, context=ssl_context) as response:
                with dest.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
        except URLError as exc:
            raise AssetLoadFailed(f"failed to download: {exc}") from exc

    def _require_api_key(self) -> str:
        api_key = os.environ.get(self._api_key_env_var or "")
        if not api_key:
            raise AssetLoadFailed(
                f"missing Google Drive API key in env var {self._api_key_env_var!r}"
            )
        return api_key

    def _get_authorized_session(self):
        try:
            from google.auth.transport.requests import Request, AuthorizedSession
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:
            raise AssetLoadFailed(
                "Google OAuth libraries are not installed. "
                "Install google-auth, google-auth-oauthlib, and requests."
            ) from exc

        creds = None
        if self._token_path.exists():
            creds = Credentials.from_authorized_user_file(
                str(self._token_path), self._scopes
            )

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        if not creds or not creds.valid:
            if self._oauth_client_secrets is None:
                raise AssetLoadFailed("oauth_client_secrets is required for OAuth")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self._oauth_client_secrets), self._scopes
            )
            creds = flow.run_local_server(port=0)
            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_path.write_text(creds.to_json(), encoding="utf-8")

        return AuthorizedSession(creds)

    @staticmethod
    def _default_token_dir() -> Path:
        """Get OS-specific default token directory."""
        return _default_config_base() / "pygmu2" / "gdrive_oauth"

class GithubUserContentAssetLoader(AssetLoader):
    def __init__(
        self,
        owner: str,
        repo: str,
        branch: str = "main",
        root_path: str = "",
    ):
        self._owner = owner
        self._repo = repo
        self._branch = branch
        self._root_path = root_path.strip("/")

    def list_remote_assets(self, wildcard_spec: str) -> list[str]:
        logger.debug(
            "Listing GitHub assets for %s/%s ref=%s root=%r spec=%r",
            self._owner,
            self._repo,
            self._branch,
            self._root_path,
            wildcard_spec,
        )
        assets = self._list_remote_assets_with_urls(wildcard_spec)
        logger.info(
            "Found %d matching GitHub assets for spec=%r",
            len(assets),
            wildcard_spec,
        )
        return [name for name, _ in assets]

    def load_remote_asset(self, wildcard_spec: str, cache_dir: Path) -> Path | None:
        logger.debug(
            "Loading GitHub asset for %s/%s ref=%s root=%r spec=%r",
            self._owner,
            self._repo,
            self._branch,
            self._root_path,
            wildcard_spec,
        )
        assets = self._list_remote_assets_with_urls(wildcard_spec)
        if not assets:
            logger.warning(
                "No GitHub assets matched spec=%r in %s/%s",
                wildcard_spec,
                self._owner,
                self._repo,
            )
            return None

        name, download_url = assets[0]
        local_path = cache_dir / name

        candidate = Path(name)
        if candidate.is_absolute():
            raise AssetLoadFailed(f"refusing to write absolute path: {name}")
        if ".." in candidate.parts:
            raise AssetLoadFailed(f"refusing to write path with '..': {name}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            logger.info("Using cached asset at %s", local_path)
            return local_path

        logger.info("Downloading %r to %s", name, local_path)
        try:
            ssl_context = self._create_ssl_context()
            with request.urlopen(download_url, context=ssl_context) as response:
                with local_path.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
        except URLError as exc:
            logger.error("Download failed for %r: %s", name, exc)
            raise AssetLoadFailed(f"failed to download {name!r}: {exc}") from exc

        return local_path

    def _list_remote_assets_with_urls(
        self, wildcard_spec: str
    ) -> list[tuple[str, str]]:
        base_url = f"https://api.github.com/repos/{self._owner}/{self._repo}/contents"
        if self._root_path:
            base_url = f"{base_url}/{quote(self._root_path)}"
        query = urlencode({"ref": self._branch})
        url = f"{base_url}?{query}"

        try:
            ssl_context = self._create_ssl_context()
            with request.urlopen(url, context=ssl_context) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            logger.error("Failed to list GitHub assets: %s", exc)
            raise AssetLoadFailed(f"failed to list assets: {exc}") from exc
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON from GitHub API: %s", exc)
            raise AssetLoadFailed(f"invalid JSON from GitHub API: {exc}") from exc

        if not isinstance(payload, list):
            logger.error("Unexpected GitHub API response for URL %r", url)
            raise AssetLoadFailed("unexpected GitHub API response")

        results: list[tuple[str, str]] = []
        for item in payload:
            if item.get("type") != "file":
                continue
            name = item.get("name")
            download_url = item.get("download_url")
            if not name or not download_url:
                continue
            if fnmatch.fnmatchcase(name, wildcard_spec):
                results.append((name, download_url))

        results.sort(key=lambda x: x[0].casefold())
        return results
