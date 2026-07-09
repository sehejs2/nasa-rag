from __future__ import annotations

import httpx
import respx

from app.config import settings
from app.tools import mars_rover_photos
from app.tools.mars_rover_photos import MARS_PHOTOS_BASE_URL


def _photo(rover: str = "Perseverance", camera: str = "NAVCAM") -> dict:
    return {
        "img_src": "https://example.com/photo.jpg",
        "earth_date": "2026-07-01",
        "camera": {"name": camera},
        "rover": {"name": rover},
    }


@respx.mock
async def test_earth_date_takes_precedence_over_sol():
    route = respx.get(f"{MARS_PHOTOS_BASE_URL}/perseverance/photos").mock(
        return_value=httpx.Response(200, json={"photos": [_photo()]})
    )

    result = await mars_rover_photos.run(rover="perseverance", earth_date="2026-07-01", sol=1000)

    assert result.ok is True
    request = route.calls.last.request
    assert request.url.params["earth_date"] == "2026-07-01"
    assert "sol" not in request.url.params
    assert result.data["photos"] == [
        {
            "img_src": "https://example.com/photo.jpg",
            "camera": "NAVCAM",
            "earth_date": "2026-07-01",
            "rover": "Perseverance",
        }
    ]


@respx.mock
async def test_sol_used_when_no_earth_date():
    route = respx.get(f"{MARS_PHOTOS_BASE_URL}/curiosity/photos").mock(
        return_value=httpx.Response(200, json={"photos": [_photo(rover="Curiosity")]})
    )

    result = await mars_rover_photos.run(rover="curiosity", sol=1000)

    assert result.ok is True
    assert route.calls.last.request.url.params["sol"] == "1000"


@respx.mock
async def test_no_date_args_uses_latest_photos_endpoint():
    route = respx.get(f"{MARS_PHOTOS_BASE_URL}/perseverance/latest_photos").mock(
        return_value=httpx.Response(200, json={"latest_photos": [_photo()]})
    )

    result = await mars_rover_photos.run(rover="perseverance")

    assert result.ok is True
    assert route.called
    assert len(result.data["photos"]) == 1


@respx.mock
async def test_camera_param_passed_when_given():
    route = respx.get(f"{MARS_PHOTOS_BASE_URL}/perseverance/photos").mock(
        return_value=httpx.Response(200, json={"photos": [_photo()]})
    )

    await mars_rover_photos.run(rover="perseverance", earth_date="2026-07-01", camera="NAVCAM")

    assert route.calls.last.request.url.params["camera"] == "NAVCAM"


@respx.mock
async def test_photos_truncated_to_five():
    photos = [_photo() for _ in range(8)]
    respx.get(f"{MARS_PHOTOS_BASE_URL}/perseverance/photos").mock(
        return_value=httpx.Response(200, json={"photos": photos})
    )

    result = await mars_rover_photos.run(rover="perseverance", earth_date="2026-07-01")

    assert len(result.data["photos"]) == 5


async def test_missing_nasa_api_key(monkeypatch):
    monkeypatch.setattr(settings, "NASA_API_KEY", "")

    result = await mars_rover_photos.run(rover="perseverance")

    assert result.ok is False
    assert "NASA_API_KEY" in result.error
