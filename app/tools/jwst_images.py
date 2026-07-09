"""JWST-related imagery/metadata search via the NASA Image and Video Library.

NASA exposes no simple "JWST science" API - JWST's raw science data lives in
the Space Telescope Science Institute's MAST archive, which requires domain
expertise (proposal IDs, instrument modes) to query meaningfully. This tool
instead searches the public NASA Image and Video Library (images-api.nasa.gov,
no API key needed) for JWST-tagged media and press metadata - useful for
"show me a JWST picture of X" style requests. Substantive JWST science
questions (discoveries, findings, mission details) are answered by the RAG
corpus (Phase 3 retrieval), not this tool.
"""

from __future__ import annotations

from app.tools.base import ToolResult, request_json, run_tool

IMAGES_SEARCH_URL = "https://images-api.nasa.gov/search"
DEFAULT_LIMIT = 5
DESCRIPTION_TRUNCATE_CHARS = 300

NAME = "jwst_images"
DESCRIPTION = (
    "Search NASA's public image/video library for JWST-related pictures and their "
    "captions (e.g. 'show me a JWST photo of the Carina Nebula'). Returns media and "
    "caption metadata only, not scientific findings - for JWST discoveries or "
    "science questions, prefer document retrieval over this tool."
)
PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search text, e.g. a target name or topic (e.g. 'Carina Nebula', 'exoplanet').",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "minimum": 1,
            "maximum": 20,
            "default": DEFAULT_LIMIT,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


def _truncate(text: str, limit: int = DESCRIPTION_TRUNCATE_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


async def _fetch(query: str, limit: int) -> dict:
    payload = await request_json("GET", IMAGES_SEARCH_URL, params={"q": query, "keywords": "JWST"})
    raw_items = payload["collection"]["items"][:limit]

    items = []
    for item in raw_items:
        data = item["data"][0]
        links = item.get("links", [])
        preview = next((link["href"] for link in links if link.get("rel") == "preview"), None)
        if preview is None and links:
            preview = links[0]["href"]
        items.append(
            {
                "title": data.get("title", ""),
                "description": _truncate(data.get("description", "")),
                "date_created": data.get("date_created", ""),
                "image_url": preview,
            }
        )
    return {"items": items}


async def run(query: str, limit: int = DEFAULT_LIMIT) -> ToolResult:
    return await run_tool(NAME, {"query": query, "limit": limit}, lambda: _fetch(query, limit))
