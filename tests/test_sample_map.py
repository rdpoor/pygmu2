"""
Tests for AudioLibrary and Strudel JSON support.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import os
from pathlib import Path

import pytest

from pygmu2 import AudioLibrary


STRUDEL_JSON_URL = "https://software.tomandandy.com/strudel.json"


@pytest.mark.skipif(
    os.environ.get("PYGMU2_NETWORK_TESTS") != "1",
    reason="Set PYGMU2_NETWORK_TESTS=1 to enable network smoke tests.",
)
def test_strudel_audio_library_smoke(tmp_path):
    cache_dir = tmp_path / "cache"
    audio_lib = AudioLibrary.from_url(STRUDEL_JSON_URL, cache_dir=cache_dir)

    local_path = Path(audio_lib.resolve("grime80"))
    assert local_path.exists()
    assert local_path.stat().st_size > 0
