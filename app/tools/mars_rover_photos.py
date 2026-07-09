"""Mars Rover Photos (api.nasa.gov/mars-photos). Needs NASA_API_KEY.

Note: NASA's mars-photos backend (community-run, proxied through api.nasa.gov)
has a history of intermittent upstream outages. This implementation follows
NASA's documented API contract; if the upstream is down, ToolResult(ok=False)
surfaces that cleanly rather than crashing.
"""

from __future__ import annotations

from app.tools.base import ToolResult, request_json, require_nasa_api_key, run_tool

MARS_PHOTOS_BASE_URL = "https://api.nasa.gov/mars-photos/api/v1/rovers"
MAX_PHOTOS = 5

NAME = "mars_rover_photos"
DESCRIPTION = (
    "Get photos taken by NASA's Perseverance or Curiosity Mars rovers on a given "
    "Earth date or mission sol (Martian day). Use for 'show me photos from Mars' "
    "or what a specific rover saw on a specific day; omit both date fields to get "
    "the rover's most recent photos instead. Returns imagery only, not science "
    "findings or mission history - use search_documents for those."
)
PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "rover": {
            "type": "string",
            "enum": ["perseverance", "curiosity"],
            "description": "Which Mars rover to query.",
        },
        "earth_date": {
            "type": "string",
            "description": (
                "Earth date the photos were taken, in YYYY-MM-DD format. "
                "Takes precedence over sol if both are given."
            ),
            "pattern": r"^\d{4}-\d{2}-\d{2}$",
        },
        "sol": {
            "type": "integer",
            "description": (
                "Martian sol (mission day number) the photos were taken, as an "
                "alternative to earth_date."
            ),
            "minimum": 0,
        },
        "camera": {
            "type": "string",
            "description": (
                "Optional camera abbreviation to filter by (e.g. NAVCAM, MAST, FHAZ). "
                "Rover-specific; omit to include all cameras."
            ),
        },
    },
    "required": ["rover"],
    "additionalProperties": False,
}


async def _fetch(rover: str, earth_date: str | None, sol: int | None, camera: str | None) -> dict:
    api_key = require_nasa_api_key()
    params: dict[str, str | int] = {"api_key": api_key}
    if camera:
        params["camera"] = camera

    if earth_date:
        url = f"{MARS_PHOTOS_BASE_URL}/{rover}/photos"
        params["earth_date"] = earth_date
    elif sol is not None:
        url = f"{MARS_PHOTOS_BASE_URL}/{rover}/photos"
        params["sol"] = sol
    else:
        url = f"{MARS_PHOTOS_BASE_URL}/{rover}/latest_photos"

    payload = await request_json("GET", url, params=params)
    photos = payload.get("photos") or payload.get("latest_photos") or []
    trimmed = [
        {
            "img_src": photo["img_src"],
            "camera": photo["camera"]["name"],
            "earth_date": photo["earth_date"],
            "rover": photo["rover"]["name"],
        }
        for photo in photos[:MAX_PHOTOS]
    ]
    return {"photos": trimmed}


async def run(
    rover: str,
    earth_date: str | None = None,
    sol: int | None = None,
    camera: str | None = None,
) -> ToolResult:
    return await run_tool(
        NAME,
        {"rover": rover, "earth_date": earth_date, "sol": sol, "camera": camera},
        lambda: _fetch(rover, earth_date, sol, camera),
    )
