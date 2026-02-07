"""
demo_asset_mgr.py

Simple example showing AssetManager with Google Drive and GitHub loaders.
Fill in the required parameters before running.
"""

from pathlib import Path

from pygmu2.asset_manager import (

    AssetManager,
    GoogleDriveAssetLoader,
    GithubUserContentAssetLoader,
    _default_config_base,
)
import pygmu2 as pg
pg.set_sample_rate(44100)


def demo_google_drive_giant_fish():
    print("\n=== Google Drive Giant Fish Demo ===")
    folder_id = "1qX5s1KCxAodHIA2sxxiHgybAHY_52LQn"
    oauth_client_secrets = (
        _default_config_base() / "pygmu2" / "client_secrets.json"
    )
    loader = GoogleDriveAssetLoader(
        folder_id=folder_id,
        oauth_client_secrets=oauth_client_secrets,
    )
    # Token cache path defaults to an OS-specific config directory.
    manager = AssetManager(asset_loader=loader)

    asset_spec = "GiantFish/SegmentedVoice/N2_1?.wav"
    # List all matching assets (returns cache-relative paths)
    remote_assets = manager.list_remote_assets(asset_spec)
    print(f"Google Drive matches for {asset_spec!r}: {len(remote_assets)}")
    for asset in remote_assets:
        print(f"  - {asset}")
    # Return the first matching asset, reload even if in cache.
    path = manager.load_asset(asset_spec, force=True)
    print(f"Google Drive Giant Fish selected asset: {path}")

def demo_google_drive():
    print("\n=== Google Drive Demo ===")
    # TODO: Fill in your Google Drive folder ID.
    folder_id = "1idxVO258Lbs_5c97MGnal8W3mdp6T2YL"
    # TODO: Path to your OAuth client secrets JSON (downloaded from Google Cloud).
    # How to create:
    #   1) In Google Cloud Console, create/select a project.
    #   2) Enable the Google Drive API.
    #   3) Create OAuth client ID credentials (Desktop app).
    #   4) Download the client JSON and point to it here.
    # Note: This is NOT the token cache file (gdrive_token.json). It is the
    # client secrets file you download from Google Cloud Console.
    # If OAuth libraries are missing, install: google-auth, google-auth-oauthlib, requests
    # You can omit this parameter if you store the secrets at:
    #   _default_config_base()/pygmu2/client_secrets.json
    oauth_client_secrets = (
        _default_config_base() / "pygmu2" / "client_secrets.json"
    )
    loader = GoogleDriveAssetLoader(
        folder_id=folder_id,
        oauth_client_secrets=oauth_client_secrets,
    )
    # Token cache path defaults to an OS-specific config directory.
    manager = AssetManager(asset_loader=loader)

    asset_spec = "multi_samples/Anklung_Hit/Anklung_Hit*.wav"
    # List all matching assets (returns cache-relative paths)
    remote_assets = manager.list_remote_assets(asset_spec)
    print(f"Google Drive matches for {asset_spec!r}: {len(remote_assets)}")
    for asset in remote_assets:
        print(f"  - {asset}")
    # Return the first matching asset, loading and caching as needed.
    path = manager.load_asset(asset_spec)
    print(f"Google Drive selected asset: {path}")


def demo_github():
    print("\n=== GitHub Demo ===")
    # TODO: Fill in your GitHub repo information.
    owner = "tomandandy"
    repo = "go"
    branch = "main"
    root_path = ""  # optional subdirectory inside the repo

    loader = GithubUserContentAssetLoader(
        owner=owner,
        repo=repo,
        branch=branch,
        root_path=root_path,
    )
    manager = AssetManager(asset_loader=loader)

    asset_spec = "SOBR_136_Full_Drum_Loop_*.wav"
    # List all matching assets
    remote_assets = manager.list_remote_assets(asset_spec)
    print(f"GitHub matches for {asset_spec!r}: {len(remote_assets)}")
    for asset in remote_assets:
        print(f"  - {asset}")
    # Return the first matching asset, loading and caching as needed.
    path = manager.load_asset(asset_spec)
    print(f"GitHub selected asset: {path}")


if __name__ == "__main__":
    demo_google_drive_giant_fish()
    demo_google_drive()
    demo_github()
