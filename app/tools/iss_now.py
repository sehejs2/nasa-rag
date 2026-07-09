"""Current ISS position (api.open-notify.org/iss-now.json). No API key needed."""

from __future__ import annotations

from app.tools.base import ToolResult, request_json, run_tool

ISS_NOW_URL = "http://api.open-notify.org/iss-now.json"

NAME = "iss_now"
DESCRIPTION = (
    "Get the current real-time latitude/longitude position of the International "
    "Space Station. Use for 'where is the ISS right now' style questions; it has "
    "no memory of past or future positions and takes no parameters."
)
PARAMETERS: dict = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}


def _describe_position(lat: float, lon: float) -> str:
    """Plain lat/lon + hemisphere, e.g. '10.00°N, 50.00°E' - no geocoding dependency."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}"


async def _fetch() -> dict:
    payload = await request_json("GET", ISS_NOW_URL)
    position = payload["iss_position"]
    latitude = float(position["latitude"])
    longitude = float(position["longitude"])
    return {
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": payload["timestamp"],
        "position_description": _describe_position(latitude, longitude),
    }


async def run() -> ToolResult:
    return await run_tool(NAME, {}, _fetch)
