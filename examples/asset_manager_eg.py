"""
AssetManager: fetch audio from a remote folder through a local cache.

The model
---------
An AssetManager is a local cache directory in front of one remote folder
(Google Drive or GitHub, via a pluggable loader):

    manager.load_asset("drums/kick*.wav")  ->  local Path

- The spec is a path relative to the remote folder. The last component is
  an fnmatch pattern ('*' = any run, '?' = one character); the leading
  components name a literal subfolder.
- If a cached file already matches, it is returned with NO network
  traffic. Otherwise the first remote match (alphabetical) is downloaded
  into the cache and returned. Pass force=True to re-download.
- list_remote_assets(spec) lists every remote match without downloading;
  list_cached_assets(spec) / has_cached_asset(spec) inspect the cache.

The cache lives at manager.cache_path() — by default:
    macOS:   ~/Library/Caches/pygmu2/asset_cache
    Linux:   ~/.cache/pygmu2/asset_cache
    Windows: %LOCALAPPDATA%/pygmu2/asset_cache

Google Drive authentication
---------------------------
Two modes, chosen by the GoogleDriveAssetLoader arguments:

- OAuth (private or shared folders): create OAuth "Desktop app"
  credentials in Google Cloud Console (enable the Drive API first) and
  save the downloaded JSON as:
      macOS:   ~/Library/Application Support/pygmu2/client_secrets.json
      Linux:   ~/.config/pygmu2/client_secrets.json
  The loader finds it there automatically — no argument needed.
  FIRST RUN OPENS A BROWSER WINDOW to authorize; the resulting token is
  cached (gdrive_oauth/gdrive_token.json next to the secrets), so later
  runs are silent. Requires: google-auth, google-auth-oauthlib, requests.

- API key (public / link-shared folders only): no OAuth setup at all.
  Pass api_key_env_var="MY_DRIVE_KEY" and export a Drive API key in that
  environment variable.

GitHub needs no authentication for public repos.

Troubleshooting
---------------
"invalid_grant: Token has been expired or revoked" — delete the cached
token and re-run (a browser window will open to re-authorize):

    macOS:   rm ~/Library/Application\\ Support/pygmu2/gdrive_oauth/gdrive_token.json
    Linux:   rm ~/.config/pygmu2/gdrive_oauth/gdrive_token.json
    Windows: del %APPDATA%\\pygmu2\\gdrive_oauth\\gdrive_token.json

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import soundfile as sf

from pygmu2.asset_manager import (
    AssetManager,
    GoogleDriveAssetLoader,
    GithubUserContentAssetLoader,
)
import pygmu2 as pg
from examples_helper import run_demos

SAMPLE_RATE = 44100
pg.set_sample_rate(SAMPLE_RATE)


def fetch_and_play(manager, asset_spec):
    """The whole AssetManager story in one function: list the remote
    matches, fetch one through the cache, show the cache working, and
    play the file."""
    print(f"cache directory: {manager.cache_path()}")

    remote = manager.list_remote_assets(asset_spec)
    print(f"remote matches for {asset_spec!r}: {len(remote)}")
    for name in remote:
        print(f"  - {name}")

    path = manager.load_asset(asset_spec)
    print(f"fetched (first match, cached locally): {path}")

    # A second load is served from the cache — no network traffic.
    # (force=True would delete the cached copy and re-download.)
    assert manager.load_asset(asset_spec) == path
    print("second load_asset() served from cache")

    file_rate = sf.info(path).samplerate
    if file_rate == SAMPLE_RATE:
        print("playing...")
        pg.play(pg.WavReaderPE(str(path)))
    else:
        print(
            f"not playing: file is {file_rate} Hz but the session is "
            f"{SAMPLE_RATE} Hz (WavReaderPE would raise; resample or "
            "match the session rate to play it)"
        )


# ------------------------------------------------------------------------------
# Demos
# ------------------------------------------------------------------------------


def demo_google_drive_oauth():
    """Works as-is IF you have access to this shared folder and OAuth
    client secrets in the default location (see module docstring).
    The '?' matches one character, so this spec matches e.g. N2_10.wav
    ... N2_19.wav inside the GiantFish/SegmentedVoice subfolder."""
    print("=== Google Drive (OAuth): list, fetch, cache, play ===")
    loader = GoogleDriveAssetLoader(
        folder_id="1qX5s1KCxAodHIA2sxxiHgybAHY_52LQn",
        # oauth client secrets are auto-discovered from the default
        # location; pass oauth_client_secrets=Path(...) to override.
    )
    manager = AssetManager(asset_loader=loader)
    fetch_and_play(manager, "GiantFish/SegmentedVoice/N2_1?.wav")


def demo_google_drive_api_key():
    """TEMPLATE — edit before running. The zero-OAuth path: works only
    for folders shared as 'anyone with the link', using a Drive API key
    from an environment variable."""
    print("=== Google Drive (API key): public folder template ===")
    folder_id = "YOUR_PUBLIC_FOLDER_ID"  # from the folder URL: /folders/<id>
    if folder_id == "YOUR_PUBLIC_FOLDER_ID":
        print("Edit demo_google_drive_api_key() first: set folder_id and")
        print("export a Drive API key, e.g.  export MY_DRIVE_KEY=AIza...")
        return
    loader = GoogleDriveAssetLoader(
        folder_id=folder_id,
        api_key_env_var="MY_DRIVE_KEY",
    )
    manager = AssetManager(asset_loader=loader)
    fetch_and_play(manager, "*.wav")


def demo_github():
    """Fetch from a public GitHub repo — no authentication needed."""
    print("=== GitHub (public repo): list, fetch, cache, play ===")
    loader = GithubUserContentAssetLoader(
        owner="tomandandy",
        repo="go",
        branch="main",
        root_path="",  # optional subdirectory inside the repo
    )
    manager = AssetManager(asset_loader=loader)
    fetch_and_play(manager, "SOBR_136_Full_Drum_Loop_*.wav")


DEMOS = [
    (
        "Google Drive via OAuth: list, fetch through cache, play",
        demo_google_drive_oauth,
    ),
    (
        "Google Drive via API key (template for public folders)",
        demo_google_drive_api_key,
    ),
    ("GitHub public repo: list, fetch through cache, play", demo_github),
]

if __name__ == "__main__":
    run_demos(DEMOS)
