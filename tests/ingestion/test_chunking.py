from pathlib import Path

from app.ingestion.chunking import (
    MAX_CHUNK_TOKENS,
    chunk_document,
    split_into_sentences,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

DOC_ID = "voyager-overview-test"
TITLE = "Voyager Mission Overview"
SOURCE_URL = "https://example.com/voyager"
SOURCE_FAMILY = "nasa_general"


def _chunk_fixture(filename: str):
    text = (FIXTURES_DIR / filename).read_text(encoding="utf-8")
    return chunk_document(DOC_ID, text, TITLE, SOURCE_URL, SOURCE_FAMILY)


def test_chunks_respect_token_budget():
    chunks = _chunk_fixture("structured_long.txt")
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.token_count <= MAX_CHUNK_TOKENS


def test_no_mid_sentence_splits():
    chunks = _chunk_fixture("structured_long.txt")
    for chunk in chunks:
        text = chunk.text.strip()
        assert text[0].isupper() or text[0].isdigit()
        assert text[-1] in ".!?\"'”)"


def test_overlap_exists_between_adjacent_chunks():
    chunks = _chunk_fixture("structured_long.txt")
    assert len(chunks) >= 2
    for first, second in zip(chunks, chunks[1:]):
        sentences_first = split_into_sentences(first.text)
        sentences_second = split_into_sentences(second.text)
        assert sentences_first[-1] == sentences_second[0]


def test_deterministic_chunk_ids():
    chunks_a = _chunk_fixture("structured_long.txt")
    chunks_b = _chunk_fixture("structured_long.txt")
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]

    # chunk_id depends on doc_id + index, not on content alone.
    text = (FIXTURES_DIR / "structured_long.txt").read_text(encoding="utf-8")
    other_doc_chunks = chunk_document("a-different-doc-id", text, TITLE, SOURCE_URL, SOURCE_FAMILY)
    assert other_doc_chunks[0].chunk_id != chunks_a[0].chunk_id


def test_heading_metadata_is_captured():
    chunks = _chunk_fixture("structured_long.txt")
    headings = [c.metadata.section_heading for c in chunks]
    assert "Voyager Mission Overview" in headings
    assert "Saturn and Beyond" in headings


def test_no_headings_yields_null_section_heading():
    chunks = _chunk_fixture("no_headings.txt")
    assert len(chunks) >= 1
    assert all(c.metadata.section_heading is None for c in chunks)


def test_chunk_metadata_fields():
    chunks = _chunk_fixture("structured_long.txt")
    for chunk in chunks:
        assert chunk.doc_id == DOC_ID
        assert chunk.metadata.title == TITLE
        assert chunk.metadata.source_url == SOURCE_URL
        assert chunk.metadata.source_family == SOURCE_FAMILY
