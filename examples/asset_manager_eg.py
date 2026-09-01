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
- load_assets(spec) (plural) fetches EVERY match, mirroring any
  directory prefix in the spec into the cache — e.g. "snares/*.wav"
  appears as snares/ inside the cache directory.
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

GitHub needs no authentication for public repos, but the API allows
only 60 listing calls/hour per IP unauthenticated. Export a GitHub
token as GITHUB_TOKEN to raise that to 5000/hour (any token works for
public repos — no scopes needed). Downloads themselves are uncounted.

Troubleshooting
---------------
A stale/revoked OAuth token ("invalid_grant") is handled automatically:
the loader discards the cached token and re-opens the browser to
re-authorize. If auth misbehaves in some other way, deleting the cached
token by hand forces a completely fresh start:

    macOS:   rm ~/Library/Application\\ Support/pygmu2/gdrive_oauth/gdrive_token.json
    Linux:   rm ~/.config/pygmu2/gdrive_oauth/gdrive_token.json
    Windows: del %APPDATA%\\pygmu2\\gdrive_oauth\\gdrive_token.json

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import json
import os
import time
from urllib import request

import soundfile as sf

from pygmu2.asset_manager import (
    AssetManager,
    GoogleDriveAssetLoader,
    GithubUserContentAssetLoader,
)
import pygmu2 as pg
from examples_helper import run_demos

# The session sample rate is deliberately NOT set here: PEs capture the
# rate at construction, so nothing needs it until we build the playback
# graph — and by then we know the fetched file's actual rate and can
# adopt it (see fetch_and_play).


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

    # Adopt the file's own sample rate as the session rate, then build
    # the graph. (WavReaderPE raises on a rate mismatch, so setting the
    # rate BEFORE construction is the whole trick.)
    file_rate = sf.info(path).samplerate
    pg.set_sample_rate(file_rate)
    print(f"playing at the file's rate ({file_rate} Hz)...")
    pg.play(pg.WavReaderPE(str(path)))


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


def play_all(paths):
    """Audition a list of local .wav files back to back, letting each
    file set the session rate before its graph is built."""
    for path in paths:
        file_rate = sf.info(path).samplerate
        pg.set_sample_rate(file_rate)
        print(f"playing {path.parent.name}/{path.name} ({file_rate} Hz)")
        pg.play(pg.WavReaderPE(str(path)))


def demo_github_kiks():
    """Fetch EVERY .wav in the repo's one_shots/kiks folder with one
    load_assets() call, then play them one after another. The 'kiks/'
    spec prefix is mirrored into the cache, so the files land in
    asset_cache/kiks/. Second run: everything cache-served, no network."""
    print("=== GitHub: fetch the whole kiks/ folder, play them all ===")
    loader = GithubUserContentAssetLoader(
        owner="tomandandy",
        repo="go",
        branch="main",
        root_path="one_shots",
    )
    manager = AssetManager(asset_loader=loader)
    print(f"cache directory: {manager.cache_path()}")
    paths = manager.load_assets("kiks/*.wav")
    print(f"fetched {len(paths)} kicks into {manager.cache_path() / 'kiks'}")
    play_all(paths)


def demo_github_snares():
    """Same as the kiks demo, for one_shots/snares — the remote folder
    appears as snares/ inside the local cache directory."""
    print("=== GitHub: mirror the snares/ folder, play them all ===")
    loader = GithubUserContentAssetLoader(
        owner="tomandandy",
        repo="go",
        branch="main",
        root_path="one_shots",
    )
    manager = AssetManager(asset_loader=loader)
    print(f"cache directory: {manager.cache_path()}")
    paths = manager.load_assets("snares/*.wav")
    print(f"fetched {len(paths)} snares into {manager.cache_path() / 'snares'}")
    play_all(paths)


def demo_github_token():
    """Check your GitHub API setup: token presence and current rate
    budget. GITHUB_TOKEN is used AUTOMATICALLY by
    GithubUserContentAssetLoader — no code changes — so this demo just
    verifies it and reports what GitHub says your quota is.

    Getting a token: github.com -> Settings -> Developer settings ->
    Personal access tokens -> Fine-grained tokens -> Generate new token.
    For public repos, no repository access or permissions are needed —
    the default read-only public token is enough. Then:

        export GITHUB_TOKEN=github_pat_...

    (The /rate_limit endpoint itself is free — this demo never spends
    any of your quota.)"""
    print("=== GitHub token check: API rate budget ===")
    token = os.environ.get("GITHUB_TOKEN")
    print(f"GITHUB_TOKEN: {'set' if token else 'not set'}")

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = request.Request("https://api.github.com/rate_limit", headers=headers)
    with request.urlopen(req) as resp:
        core = json.loads(resp.read())["resources"]["core"]
    reset_at = time.strftime("%H:%M:%S", time.localtime(core["reset"]))
    print(
        f"listing budget: {core['remaining']} of {core['limit']} "
        f"calls remaining this hour (resets at {reset_at})"
    )
    if not token:
        print("Export a token to raise the limit from 60 to 5000/hour")
        print("(see this demo's docstring for where to get one).")


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
    ("GitHub kiks/ folder: mirror into cache, play back to back", demo_github_kiks),
    ("GitHub snares/ folder: mirror into cache, play back to back", demo_github_snares),
    ("GitHub token check: show your API rate budget (free call)", demo_github_token),
]

if __name__ == "__main__":
    run_demos(DEMOS)
