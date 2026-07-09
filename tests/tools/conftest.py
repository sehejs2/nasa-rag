"""Shared fixtures for tool tests. All network access is mocked via respx."""

from __future__ import annotations

import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _default_nasa_api_key(monkeypatch):
    """Give NASA_API_KEY a working-looking value by default.

    Tests that specifically exercise the missing-key path override this with
    their own monkeypatch.setattr(settings, "NASA_API_KEY", "").
    """
    monkeypatch.setattr(settings, "NASA_API_KEY", "test-nasa-key")
