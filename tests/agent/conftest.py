"""Fixtures for agent tests.

Stubs the four live NASA tool executions so routing tests don't depend on
NASA API uptime or rate limits - we're testing routing decisions, not NASA's
infrastructure. search_documents is deliberately left real so retrieval-routed
cases hit the actual local DB (and real embeddings/rerank calls).
"""

from __future__ import annotations

import dataclasses

import pytest

from app.tools import registry
from app.tools.base import ToolResult

_STUB_DATA = {
    "apod": {
        "title": "Stub APOD",
        "date": "2026-01-01",
        "explanation": "stub explanation",
        "media_type": "image",
        "url": "https://example.com/apod.jpg",
    },
    "iss_now": {
        "latitude": 0.0,
        "longitude": 0.0,
        "timestamp": 0,
        "position_description": "0.00°N, 0.00°E",
    },
    "mars_rover_photos": {
        "photos": [
            {
                "img_src": "https://example.com/mars.jpg",
                "camera": "NAVCAM",
                "earth_date": "2026-01-01",
                "rover": "Perseverance",
            }
        ]
    },
    "jwst_images": {
        "items": [
            {
                "title": "Stub JWST",
                "description": "stub description",
                "date_created": "2026-01-01",
                "image_url": "https://example.com/jwst.jpg",
            }
        ]
    },
}


def _make_stub_run(tool_name: str, data: dict):
    async def stub_run(**kwargs):
        return ToolResult(tool_name=tool_name, ok=True, data=data, latency_ms=1.0, called_with=kwargs)

    return stub_run


@pytest.fixture(autouse=True)
def _stub_live_tools(monkeypatch):
    for name, data in _STUB_DATA.items():
        spec = registry._TOOLS[name]
        monkeypatch.setitem(registry._TOOLS, name, dataclasses.replace(spec, run=_make_stub_run(name, data)))
