from __future__ import annotations

import httpx
import respx

from app.tools import jwst_images
from app.tools.jwst_images import IMAGES_SEARCH_URL


def _item(title: str = "Test", description: str = "", rel: str = "preview") -> dict:
    return {
        "data": [{"title": title, "description": description, "date_created": "2026-01-01T00:00:00Z"}],
        "links": [{"href": "https://example.com/img.jpg", "rel": rel}],
    }


@respx.mock
async def test_happy_path_and_description_truncation():
    route = respx.get(IMAGES_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json={"collection": {"items": [_item(description="A" * 400)]}}
        )
    )

    result = await jwst_images.run(query="Carina Nebula")

    assert result.ok is True
    request = route.calls.last.request
    assert request.url.params["q"] == "Carina Nebula"
    assert request.url.params["keywords"] == "JWST"

    item = result.data["items"][0]
    assert item["title"] == "Test"
    assert len(item["description"]) == 303  # 300 chars + "..."
    assert item["description"].endswith("...")
    assert item["image_url"] == "https://example.com/img.jpg"
    assert item["date_created"] == "2026-01-01T00:00:00Z"


@respx.mock
async def test_short_description_not_truncated():
    respx.get(IMAGES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"collection": {"items": [_item(description="short")]}})
    )

    result = await jwst_images.run(query="galaxy")

    assert result.data["items"][0]["description"] == "short"


@respx.mock
async def test_limit_truncates_items():
    items = [_item(title=f"item-{i}") for i in range(10)]
    respx.get(IMAGES_SEARCH_URL).mock(return_value=httpx.Response(200, json={"collection": {"items": items}}))

    result = await jwst_images.run(query="galaxy", limit=3)

    assert len(result.data["items"]) == 3


@respx.mock
async def test_falls_back_to_first_link_when_no_preview_rel():
    respx.get(IMAGES_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"collection": {"items": [_item(rel="alternate")]}})
    )

    result = await jwst_images.run(query="galaxy")

    assert result.data["items"][0]["image_url"] == "https://example.com/img.jpg"
