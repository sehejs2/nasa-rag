from __future__ import annotations

import httpx
import pytest
import respx

from app.tools import iss_now
from app.tools.iss_now import ISS_NOW_URL


@respx.mock
async def test_happy_path_parses_southern_western_position():
    respx.get(ISS_NOW_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "timestamp": 1783552953,
                "message": "success",
                "iss_position": {"longitude": "-109.1065", "latitude": "-26.7049"},
            },
        )
    )

    result = await iss_now.run()

    assert result.ok is True
    assert result.tool_name == "iss_now"
    assert result.called_with == {}
    assert result.data["latitude"] == pytest.approx(-26.7049)
    assert result.data["longitude"] == pytest.approx(-109.1065)
    assert result.data["timestamp"] == 1783552953
    assert result.data["position_description"] == "26.70°S, 109.11°W"


@respx.mock
async def test_northern_eastern_position_description():
    respx.get(ISS_NOW_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "timestamp": 1,
                "message": "success",
                "iss_position": {"longitude": "50.0", "latitude": "10.0"},
            },
        )
    )

    result = await iss_now.run()

    assert result.data["position_description"] == "10.00°N, 50.00°E"


@respx.mock
async def test_timeout_retries_then_ok_false():
    route = respx.get(ISS_NOW_URL).mock(side_effect=httpx.TimeoutException("boom"))

    result = await iss_now.run()

    assert result.ok is False
    assert route.call_count == 3
