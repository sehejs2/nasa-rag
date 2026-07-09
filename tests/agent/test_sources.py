"""Unit tests for build_sources: dedup, ordering, and failed-tool exclusion."""

from __future__ import annotations

from app.agent.models import AgentTrace, TokenUsage, ToolCallRecord
from app.agent.sources import build_sources
from app.retrieval.models import RetrievedChunk


def _chunk(chunk_id: str, rank: int, title: str = "Title", section: str | None = "Intro") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=f"text for {chunk_id}",
        title=title,
        source_url=f"https://example.com/{chunk_id}",
        source_family="nasa_general",
        section=section,
        vector_score=0.9,
        rerank_score=8.0,
        rank=rank,
    )


def _tool_call(
    tool_name: str,
    call_id: str,
    *,
    iteration: int = 1,
    result_ok: bool = True,
    arguments: dict | None = None,
    result_data: dict | None = None,
) -> ToolCallRecord:
    return ToolCallRecord(
        iteration=iteration,
        tool_name=tool_name,
        arguments=arguments or {},
        result_ok=result_ok,
        result_summary="summary",
        latency_ms=1.0,
        call_id=call_id,
        result_data=result_data,
    )


def _trace(**overrides) -> AgentTrace:
    defaults = dict(
        query="q",
        route="direct",
        tool_calls=[],
        retrieved_chunks=[],
        draft_answer="",
        iterations_used=1,
        stopped_reason="model_finished",
        total_latency_ms=1.0,
        token_usage=TokenUsage(),
    )
    defaults.update(overrides)
    return AgentTrace(**defaults)


def test_chunks_deduplicated_by_chunk_id():
    trace = _trace(
        retrieved_chunks=[
            _chunk("c1", rank=1),
            _chunk("c2", rank=2),
            _chunk("c1", rank=3),  # duplicate chunk_id from a second search_documents call
        ]
    )

    sources = build_sources(trace)

    assert [s.ref_id for s in sources] == ["c1", "c2"]
    assert [s.number for s in sources] == [1, 2]


def test_chunks_ordered_by_rank_not_list_order():
    trace = _trace(retrieved_chunks=[_chunk("c2", rank=2), _chunk("c1", rank=1)])

    sources = build_sources(trace)

    assert [s.ref_id for s in sources] == ["c1", "c2"]


def test_failed_tool_calls_excluded():
    trace = _trace(
        tool_calls=[
            _tool_call("iss_now", "call-1", result_ok=True, result_data={"latitude": 1.0}),
            _tool_call("apod", "call-2", result_ok=False, result_data=None),
        ]
    )

    sources = build_sources(trace)

    assert len(sources) == 1
    assert sources[0].ref_id == "call-1"


def test_search_documents_tool_calls_produce_no_tool_source():
    trace = _trace(
        tool_calls=[_tool_call("search_documents", "call-1", arguments={"query": "x"}, result_data={"chunks": []})],
        retrieved_chunks=[_chunk("c1", rank=1)],
    )

    sources = build_sources(trace)

    assert len(sources) == 1
    assert sources[0].kind == "chunk"


def test_chunks_come_before_tools_and_numbering_is_contiguous():
    trace = _trace(
        retrieved_chunks=[_chunk("c1", rank=1)],
        tool_calls=[_tool_call("iss_now", "call-1", result_data={"latitude": 1.0})],
    )

    sources = build_sources(trace)

    assert [s.kind for s in sources] == ["chunk", "tool"]
    assert [s.number for s in sources] == [1, 2]


def test_tools_ordered_by_call_order():
    trace = _trace(
        tool_calls=[
            _tool_call("apod", "call-1", result_data={"url": "https://example.com/apod.jpg"}),
            _tool_call("iss_now", "call-2", result_data={"latitude": 1.0}),
        ]
    )

    sources = build_sources(trace)

    assert [s.ref_id for s in sources] == ["call-1", "call-2"]


def test_apod_url_extracted_from_result_data():
    trace = _trace(
        tool_calls=[_tool_call("apod", "call-1", result_data={"url": "https://apod.nasa.gov/apod/image/x.jpg"})]
    )

    sources = build_sources(trace)

    assert sources[0].url == "https://apod.nasa.gov/apod/image/x.jpg"


def test_mars_rover_photos_url_extracted_from_first_photo():
    trace = _trace(
        tool_calls=[
            _tool_call(
                "mars_rover_photos",
                "call-1",
                result_data={"photos": [{"img_src": "https://example.com/mars.jpg", "camera": "NAVCAM"}]},
            )
        ]
    )

    sources = build_sources(trace)

    assert sources[0].url == "https://example.com/mars.jpg"


def test_jwst_images_url_extracted_from_first_item():
    trace = _trace(
        tool_calls=[
            _tool_call(
                "jwst_images",
                "call-1",
                result_data={"items": [{"image_url": "https://example.com/jwst.jpg", "title": "T"}]},
            )
        ]
    )

    sources = build_sources(trace)

    assert sources[0].url == "https://example.com/jwst.jpg"


def test_iss_now_has_no_url():
    trace = _trace(tool_calls=[_tool_call("iss_now", "call-1", result_data={"latitude": 1.0, "longitude": 2.0})])

    sources = build_sources(trace)

    assert sources[0].url is None
