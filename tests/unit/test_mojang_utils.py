import json
from pathlib import Path
from typing import Any

import pytest
import requests

from mcio_remote import util


def test_mojang_version_functions(
    fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with open(fixtures_dir / "version_manifest_v2.json") as f:
        mock_manifest = json.load(f)
    with open(fixtures_dir / "1.21.3.json") as f:
        mock_version_details = json.load(f)

    # Use requests-mock?
    class MockResponse:
        def __init__(self, json_data: dict[str, Any]) -> None:
            self.json_data = json_data

        def json(self) -> dict[str, Any]:
            return self.json_data

        def raise_for_status(self) -> None:
            pass

    def mock_requests_get(url: str) -> MockResponse:
        if "version_manifest" in url:
            return MockResponse(mock_manifest)
        elif "1.21.3" in url:
            return MockResponse(mock_version_details)
        raise RuntimeError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", mock_requests_get)

    # Test mojang_get_version_manifest
    manifest = util.mojang_get_version_manifest()
    assert manifest == mock_manifest

    # Test mojang_get_version_info
    version_info = util.mojang_get_version_info("1.21.3")
    assert version_info["id"] == "1.21.3"

    # Test version not found
    with pytest.raises(ValueError, match="Version not found: 1.0.0"):
        util.mojang_get_version_info("1.0.0")

    # Test mojang_get_version_details
    version_details = util.mojang_get_version_details("1.21.3")
    assert version_details["id"] == "1.21.3"
    assert "downloads" in version_details
