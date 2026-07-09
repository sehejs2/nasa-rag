from __future__ import annotations

import httpx
import respx

from app.tools import registry


def test_get_openai_tools_schemas_are_valid():
    tools = registry.get_openai_tools()

    assert len(tools) == 4
    names = set()
    for tool in tools:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert isinstance(params["properties"], dict)
        assert "required" in params
        assert isinstance(params["required"], list)
        names.add(fn["name"])

    assert names == {"apod", "iss_now", "mars_rover_photos", "jwst_images"}


async def test_execute_unknown_tool_name():
    result = await registry.execute("not_a_real_tool", {})

    assert result.ok is False
    assert "Unknown tool" in result.error


async def test_execute_missing_required_arg():
    result = await registry.execute("mars_rover_photos", {})

    assert result.ok is False
    assert "rover" in result.error


async def test_execute_invalid_enum_value():
    result = await registry.execute("mars_rover_photos", {"rover": "opportunity"})

    assert result.ok is False


async def test_execute_wrong_type():
    result = await registry.execute("jwst_images", {"query": "galaxy", "limit": "five"})

    assert result.ok is False


async def test_execute_invalid_date_pattern():
    result = await registry.execute("apod", {"date": "07-01-2026"})

    assert result.ok is False


async def test_execute_no_args_defaults_to_empty_dict():
    result = await registry.execute("mars_rover_photos", None)

    assert result.ok is False
    assert "rover" in result.error


@respx.mock
async def test_execute_happy_path_routes_to_tool():
    respx.get("http://api.open-notify.org/iss-now.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "timestamp": 1,
                "message": "success",
                "iss_position": {"latitude": "1.0", "longitude": "2.0"},
            },
        )
    )

    result = await registry.execute("iss_now", {})

    assert result.ok is True
    assert result.tool_name == "iss_now"
