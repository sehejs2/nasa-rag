"""Astronomy Picture of the Day (api.nasa.gov/planetary/apod). Needs NASA_API_KEY."""

from __future__ import annotations

from app.tools.base import ToolResult, request_json, require_nasa_api_key, run_tool

APOD_URL = "https://api.nasa.gov/planetary/apod"

NAME = "apod"
DESCRIPTION = (
    "Get NASA's Astronomy Picture of the Day (APOD): a curated astronomy image or "
    "video with an expert-written explanation. Use for 'picture of the day', "
    "'today's APOD', or questions about a specific past day's featured image."
)
PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "date": {
            "type": "string",
            "description": (
                "Date of the APOD to fetch, in YYYY-MM-DD format. Omit for today's picture."
            ),
            "pattern": r"^\d{4}-\d{2}-\d{2}$",
        }
    },
    "required": [],
    "additionalProperties": False,
}


async def _fetch(date: str | None) -> dict:
    api_key = require_nasa_api_key()
    params: dict[str, str] = {"api_key": api_key}
    if date:
        params["date"] = date

    payload = await request_json("GET", APOD_URL, params=params)
    return {
        "title": payload.get("title", ""),
        "date": payload.get("date", ""),
        "explanation": payload.get("explanation", ""),
        "media_type": payload.get("media_type", ""),
        "url": payload.get("url", ""),
    }


async def run(date: str | None = None) -> ToolResult:
    return await run_tool(NAME, {"date": date}, lambda: _fetch(date))
