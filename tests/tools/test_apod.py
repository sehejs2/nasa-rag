from __future__ import annotations

import httpx
import respx

from app.config import settings
from app.tools import apod
from app.tools.apod import APOD_URL


@respx.mock
async def test_happy_path_builds_url_and_parses_response():
    route = respx.get(APOD_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "title": "Test Nebula",
                "date": "2026-07-01",
                "explanation": "A lovely nebula.",
                "media_type": "image",
                "url": "https://apod.nasa.gov/apod/image/test.jpg",
                "service_version": "v1",
            },
        )
    )

    result = await apod.run(date="2026-07-01")

    assert result.ok is True
    assert result.tool_name == "apod"
    assert result.data == {
        "title": "Test Nebula",
        "date": "2026-07-01",
        "explanation": "A lovely nebula.",
        "media_type": "image",
        "url": "https://apod.nasa.gov/apod/image/test.jpg",
    }
    assert result.called_with == {"date": "2026-07-01"}

    request = route.calls.last.request
    assert request.url.params["date"] == "2026-07-01"
    assert request.url.params["api_key"] == "test-nasa-key"


@respx.mock
async def test_date_omitted_when_not_given():
    route = respx.get(APOD_URL).mock(
        return_value=httpx.Response(
            200,
            json={"title": "T", "date": "2026-07-08", "explanation": "E", "media_type": "image", "url": "u"},
        )
    )

    result = await apod.run()

    assert result.ok is True
    assert "date" not in route.calls.last.request.url.params


async def test_missing_nasa_api_key(monkeypatch):
    monkeypatch.setattr(settings, "NASA_API_KEY", "")

    result = await apod.run()

    assert result.ok is False
    assert "NASA_API_KEY" in result.error


@respx.mock
async def test_404_does_not_retry():
    route = respx.get(APOD_URL).mock(return_value=httpx.Response(404, json={"error": "not found"}))

    result = await apod.run(date="1900-01-01")

    assert result.ok is False
    assert route.call_count == 1
