"""
Tests for AssetManager and its loaders — device-free and network-free.

Network boundaries are faked: AssetManager is exercised with an in-memory
loader; GoogleDriveAssetLoader's Drive API calls are stubbed at
_list_folder_items/_download_file; GithubUserContentAssetLoader's HTTP is
stubbed at request.urlopen.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import fnmatch
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import pygmu2.asset_manager as am
from pygmu2.asset_manager import (
    AssetLoader,
    AssetLoadFailed,
    AssetManager,
    AssetNotFound,
    GithubUserContentAssetLoader,
    GoogleDriveAssetLoader,
)


class FakeLoader(AssetLoader):
    """In-memory loader: a dict of relative-name -> bytes."""

    def __init__(self, assets: dict[str, bytes]):
        self.assets = assets
        self.load_calls = 0
        self.list_calls = 0

    def list_remote_assets(self, wildcard_spec: str) -> list[str]:
        self.list_calls += 1
        return sorted(
            name for name in self.assets if fnmatch.fnmatchcase(name, wildcard_spec)
        )

    def load_remote_asset(self, wildcard_spec: str, cache_dir: Path):
        self.load_calls += 1
        matches = self.list_remote_assets(wildcard_spec)
        if not matches:
            return None
        name = matches[0]
        dest = cache_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.assets[name])
        return dest


@pytest.fixture
def loader():
    return FakeLoader(
        {
            "drums/kick.wav": b"KICK",
            "drums/snare.wav": b"SNARE",
            "loops/groove_01.wav": b"G1",
            "loops/groove_02.wav": b"G2",
        }
    )


@pytest.fixture
def manager(tmp_path, loader):
    return AssetManager(cache_dir=tmp_path / "cache", asset_loader=loader)


class TestAssetManager:
    def test_cache_miss_fetches_and_caches(self, manager, loader):
        path = manager.load_asset("drums/kick.wav")
        assert path.read_bytes() == b"KICK"
        assert path.is_relative_to(manager.cache_path())
        assert loader.load_calls == 1
        # second load is a cache hit — the loader is not consulted again
        again = manager.load_asset("drums/kick.wav")
        assert again == path
        assert loader.load_calls == 1

    def test_wildcard_returns_first_match_sorted(self, manager):
        path = manager.load_asset("loops/groove_0?.wav")
        assert path.name == "groove_01.wav"

    def test_force_deletes_cache_and_refetches(self, manager, loader):
        path = manager.load_asset("drums/snare.wav")
        path.write_bytes(b"LOCALLY MODIFIED")
        refreshed = manager.load_asset("drums/snare.wav", force=True)
        assert refreshed.read_bytes() == b"SNARE"
        assert loader.load_calls == 2

    def test_no_match_raises_asset_not_found(self, manager):
        with pytest.raises(AssetNotFound):
            manager.load_asset("no_such_thing.wav")

    def test_no_loader_and_no_cache_raises(self, tmp_path):
        lonely = AssetManager(cache_dir=tmp_path / "c2")
        with pytest.raises(AssetLoadFailed, match="not configured"):
            lonely.load_asset("anything.wav")

    def test_no_loader_still_serves_cached_files(self, tmp_path):
        cache = tmp_path / "c3"
        cache.mkdir()
        (cache / "hit.wav").write_bytes(b"X")
        lonely = AssetManager(cache_dir=cache)
        assert lonely.load_asset("hit.wav").read_bytes() == b"X"

    def test_locate_local_asset_absolute_path(self, manager, tmp_path):
        f = tmp_path / "outside.wav"
        f.write_bytes(b"OUT")
        assert manager.locate_local_asset(str(f)) == f
        assert manager.locate_local_asset(str(tmp_path / "absent.wav")) is None

    def test_list_and_has_cached_assets(self, manager):
        assert not manager.has_cached_asset("drums/*.wav")
        manager.load_asset("drums/kick.wav")
        manager.load_asset("drums/snare.wav")
        cached = manager.list_cached_assets("drums/*.wav")
        assert [p.name for p in cached] == ["kick.wav", "snare.wav"]
        assert manager.has_cached_asset("drums/*.wav")

    def test_list_remote_assets(self, manager):
        names = manager.list_remote_assets("loops/*.wav")
        assert names == [Path("loops/groove_01.wav"), Path("loops/groove_02.wav")]

    def test_list_remote_assets_without_loader_raises(self, tmp_path):
        lonely = AssetManager(cache_dir=tmp_path / "c4")
        with pytest.raises(AssetLoadFailed):
            lonely.list_remote_assets("*.wav")

    def test_clear_cache_refuses_non_default_dir(self, manager):
        manager.load_asset("drums/kick.wav")
        with pytest.raises(AssetLoadFailed, match="refusing to clear"):
            manager.clear_cache()
        # and the cached file survives the refusal
        assert manager.has_cached_asset("drums/kick.wav")


class TestGoogleDriveAssetLoader:
    """Drive API stubbed at _list_folder_items / _download_file."""

    ROOT = "root-folder-id"

    @pytest.fixture(autouse=True)
    def hermetic_config(self, tmp_path, monkeypatch):
        # Keep the loader from discovering a real client_secrets.json on
        # the developer's machine.
        monkeypatch.setattr(am, "_default_config_base", lambda: tmp_path / "cfg")

    def _loader(self, monkeypatch):
        monkeypatch.setenv("FAKE_DRIVE_KEY", "test-key")
        return GoogleDriveAssetLoader(
            folder_id=self.ROOT, api_key_env_var="FAKE_DRIVE_KEY"
        )

    FOLDERS = {
        ROOT: [
            {
                "id": "sub-id",
                "name": "sub",
                "mimeType": "application/vnd.google-apps.folder",
            },
            {"id": "f1", "name": "b_second.wav", "mimeType": "audio/wav"},
            {"id": "f2", "name": "a_first.wav", "mimeType": "audio/wav"},
            {
                "id": "f3",
                "name": "notes.gdoc",
                "mimeType": "application/vnd.google-apps.document",
            },
            {"id": "f4", "name": "readme.txt", "mimeType": "text/plain"},
        ],
        "sub-id": [
            {"id": "f5", "name": "inner.wav", "mimeType": "audio/wav"},
        ],
    }

    def test_requires_some_authentication(self):
        with pytest.raises(AssetLoadFailed, match="requires"):
            GoogleDriveAssetLoader(folder_id=self.ROOT)

    def test_missing_api_key_env_raises(self, monkeypatch):
        monkeypatch.setenv("FAKE_DRIVE_KEY", "test-key")
        loader = GoogleDriveAssetLoader(
            folder_id=self.ROOT, api_key_env_var="FAKE_DRIVE_KEY"
        )
        monkeypatch.delenv("FAKE_DRIVE_KEY")
        with pytest.raises(AssetLoadFailed, match="missing Google Drive API key"):
            loader._require_api_key()

    def test_split_spec(self, monkeypatch):
        loader = self._loader(monkeypatch)
        assert loader._split_spec("a/b/c*.wav") == ("a/b/", "c*.wav")
        assert loader._split_spec("*.wav") == ("", "*.wav")
        assert loader._split_spec("/lead/slash.wav") == ("lead/", "slash.wav")

    def test_listing_filters_folders_and_google_docs(self, monkeypatch):
        loader = self._loader(monkeypatch)
        with patch.object(
            loader, "_list_folder_items", side_effect=lambda fid: self.FOLDERS[fid]
        ):
            names = loader.list_remote_assets("*")
        # sorted, no folder, no Google-Docs entry
        assert names == ["a_first.wav", "b_second.wav", "readme.txt"]

    def test_listing_resolves_subfolder_prefix(self, monkeypatch):
        loader = self._loader(monkeypatch)
        with patch.object(
            loader, "_list_folder_items", side_effect=lambda fid: self.FOLDERS[fid]
        ):
            assert loader.list_remote_assets("sub/*.wav") == ["sub/inner.wav"]
            assert loader.list_remote_assets("no_such_dir/*.wav") == []

    def test_load_remote_asset_downloads_first_match(self, monkeypatch, tmp_path):
        loader = self._loader(monkeypatch)
        downloads = []

        def fake_download(url, dest):
            downloads.append(url)
            dest.write_bytes(b"AUDIO")

        with (
            patch.object(
                loader, "_list_folder_items", side_effect=lambda fid: self.FOLDERS[fid]
            ),
            patch.object(loader, "_download_file", side_effect=fake_download),
        ):
            path = loader.load_remote_asset("*.wav", tmp_path)
        assert path == tmp_path / "a_first.wav"
        assert path.read_bytes() == b"AUDIO"
        assert downloads == ["https://www.googleapis.com/drive/v3/files/f2"]

    def test_load_remote_asset_uses_existing_cache_file(self, monkeypatch, tmp_path):
        loader = self._loader(monkeypatch)
        (tmp_path / "a_first.wav").write_bytes(b"CACHED")
        with (
            patch.object(
                loader, "_list_folder_items", side_effect=lambda fid: self.FOLDERS[fid]
            ),
            patch.object(loader, "_download_file") as dl,
        ):
            path = loader.load_remote_asset("*.wav", tmp_path)
        assert path.read_bytes() == b"CACHED"
        dl.assert_not_called()

    @pytest.mark.parametrize("evil", ["../escape.wav", "/abs/path.wav"])
    def test_load_remote_asset_refuses_traversal(self, monkeypatch, tmp_path, evil):
        loader = self._loader(monkeypatch)
        with patch.object(
            loader,
            "_list_remote_assets_with_ids",
            return_value=[(evil, "fid")],
        ):
            with pytest.raises(AssetLoadFailed, match="refusing to write"):
                loader.load_remote_asset("*", tmp_path)


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


class TestGithubUserContentAssetLoader:
    PAYLOAD = [
        {"type": "file", "name": "loop_b.wav", "download_url": "https://dl/loop_b.wav"},
        {"type": "file", "name": "loop_a.wav", "download_url": "https://dl/loop_a.wav"},
        {"type": "dir", "name": "subdir", "download_url": None},
        {"type": "file", "name": "readme.md", "download_url": "https://dl/readme.md"},
    ]

    def _loader(self):
        return GithubUserContentAssetLoader(owner="o", repo="r", branch="main")

    def test_listing_filters_and_sorts(self):
        with patch.object(
            am.request,
            "urlopen",
            return_value=_FakeResponse(json.dumps(self.PAYLOAD).encode()),
        ):
            names = self._loader().list_remote_assets("loop_*.wav")
        assert names == ["loop_a.wav", "loop_b.wav"]

    def test_non_list_payload_raises(self):
        with patch.object(
            am.request,
            "urlopen",
            return_value=_FakeResponse(json.dumps({"message": "Not Found"}).encode()),
        ):
            with pytest.raises(AssetLoadFailed, match="unexpected GitHub API response"):
                self._loader().list_remote_assets("*.wav")

    def test_load_remote_asset_downloads_first_match(self, tmp_path):
        listing = _FakeResponse(json.dumps(self.PAYLOAD).encode())
        audio = _FakeResponse(b"WAVBYTES")
        with patch.object(am.request, "urlopen", side_effect=[listing, audio]):
            path = self._loader().load_remote_asset("loop_*.wav", tmp_path)
        assert path == tmp_path / "loop_a.wav"
        assert path.read_bytes() == b"WAVBYTES"

    def test_load_remote_asset_no_match_returns_none(self, tmp_path):
        with patch.object(
            am.request,
            "urlopen",
            return_value=_FakeResponse(json.dumps(self.PAYLOAD).encode()),
        ):
            assert self._loader().load_remote_asset("*.flac", tmp_path) is None


class TestOAuthTokenRecovery:
    """A stale refresh token (invalid_grant) must fall back to the
    interactive flow instead of crashing (device-free: google-auth
    objects are mocked)."""

    def test_refresh_failure_reauthorizes(self, tmp_path, monkeypatch):
        google = pytest.importorskip("google.auth.exceptions")

        monkeypatch.setattr(am, "_default_config_base", lambda: tmp_path / "cfg")
        secrets = tmp_path / "client_secrets.json"
        secrets.write_text("{}")
        token = tmp_path / "gdrive_token.json"
        token.write_text('{"stale": true}')
        loader = GoogleDriveAssetLoader(
            folder_id="root",
            oauth_client_secrets=secrets,
            token_path=token,
        )

        from unittest.mock import MagicMock

        stale_creds = MagicMock()
        stale_creds.expired = True
        stale_creds.refresh_token = "r"
        stale_creds.refresh.side_effect = google.RefreshError("invalid_grant")

        fresh_creds = MagicMock()
        fresh_creds.valid = True
        fresh_creds.to_json.return_value = '{"fresh": true}'
        flow = MagicMock()
        flow.run_local_server.return_value = fresh_creds

        with (
            patch(
                "google.oauth2.credentials.Credentials.from_authorized_user_file",
                return_value=stale_creds,
            ),
            patch(
                "google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
                return_value=flow,
            ),
            patch("google.auth.transport.requests.AuthorizedSession") as session_cls,
        ):
            session = loader._get_authorized_session()

        # interactive flow ran, fresh token written, session built on it
        flow.run_local_server.assert_called_once()
        assert token.read_text() == '{"fresh": true}'
        session_cls.assert_called_once_with(fresh_creds)
        assert session is session_cls.return_value
