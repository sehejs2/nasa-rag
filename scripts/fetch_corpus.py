"""Fetch a corpus of public NASA documents into data/raw/ for Phase 1 chunking.

Sources (three families, see CLAUDE.md):
  - jwst: JWST science releases from science.nasa.gov (webbtelescope.org content
    has been folded into science.nasa.gov's WordPress site).
  - mars: Perseverance/Curiosity mission updates from science.nasa.gov.
  - nasa_general: general NASA newsroom articles (nasa.gov RSS feeds) plus a
    handful of longer technical report PDFs from the NASA Technical Reports
    Server (ntrs.nasa.gov).

Candidates are discovered dynamically via public NASA APIs/feeds (no scraping
of search-result HTML), converted to clean text, and recorded in
data/raw/manifest.json. Re-running the script skips URLs already present in
the manifest, so it is safe (and cheap) to re-run to top up the corpus.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"

USER_AGENT = (
    "nasa-rag-corpus-fetcher/0.1 "
    "(educational portfolio project; contact: sehej2247@gmail.com)"
)
REQUEST_DELAY_SECONDS = 0.75
REQUEST_TIMEOUT_SECONDS = 30
MIN_EXTRACTED_CHARS = 200
MAX_PDF_PAGES = 60

TARGET_COUNTS = {
    "jwst": 20,
    "mars": 20,
    "nasa_general_html": 15,
    "nasa_general_pdf": 8,
}

JWST_QUERIES = [
    "James Webb Space Telescope",
    "Webb telescope galaxy",
    "Webb telescope exoplanet",
    "Webb telescope nebula",
    "Webb Space Telescope discovery",
    "Webb telescope star formation",
]
JWST_URL_MARKERS = ["/missions/webb/", "webb-telescope", "webb-space-telescope"]

MARS_QUERIES = [
    "Perseverance rover Mars",
    "Curiosity rover Mars",
    "Mars Sample Return",
    "Mars 2020 mission",
    "Mars helicopter Ingenuity",
    "Mars rover science",
]
MARS_URL_MARKERS = ["mars", "curiosity", "perseverance"]

NASA_GENERAL_FEEDS = [
    "https://www.nasa.gov/feed/",
    "https://www.nasa.gov/news-release/feed/",
]
NASA_GENERAL_FEED_PAGES = 4

NTRS_QUERIES = [
    "NASA mission report",
    "space technology report",
    "aeronautics research report",
    "spacecraft systems report",
    "NASA history fact sheet",
    "satellite mission summary",
]

SCIENCE_SEARCH_URL = "https://science.nasa.gov/wp-json/wp/v2/posts"
NTRS_SEARCH_URL = "https://ntrs.nasa.gov/api/citations/search"
NTRS_BASE_URL = "https://ntrs.nasa.gov"


@dataclass
class Candidate:
    url: str
    title: str
    source_family: str  # jwst | mars | nasa_general
    kind: str  # html | pdf


def log(message: str) -> None:
    print(message, file=sys.stderr)


def slugify(title: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:max_len].strip("-") or "doc"


def stable_id(url: str, title: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{slugify(title)}-{digest}"


def load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return []


def save_manifest(entries: list[dict]) -> None:
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2))


def collect_science_candidates(
    client: httpx.Client, queries: list[str], family: str, url_markers: list[str]
) -> dict[str, str]:
    """Search science.nasa.gov's public WP-JSON API and keep on-topic links."""
    found: dict[str, str] = {}
    for query in queries:
        try:
            resp = client.get(
                SCIENCE_SEARCH_URL,
                params={"search": query, "per_page": 30, "_fields": "id,link,title"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log(f"[warn] science.nasa.gov search failed for {query!r}: {exc}")
            continue
        for item in resp.json():
            link = item["link"]
            title = html.unescape(item["title"]["rendered"])
            if any(marker in link.lower() for marker in url_markers) and link not in found:
                found[link] = title
        time.sleep(REQUEST_DELAY_SECONDS)
    return found


def collect_nasa_feed_candidates(
    client: httpx.Client, feeds: list[str], pages: int
) -> dict[str, str]:
    """Pull article links out of NASA's public RSS feeds."""
    found: dict[str, str] = {}
    for feed in feeds:
        for page in range(1, pages + 1):
            target = feed if page == 1 else f"{feed}?paged={page}"
            try:
                resp = client.get(target, timeout=REQUEST_TIMEOUT_SECONDS)
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
            except (httpx.HTTPError, ET.ParseError) as exc:
                log(f"[warn] feed fetch/parse failed for {target}: {exc}")
                continue
            for item in root.iter("item"):
                link_el = item.find("link")
                title_el = item.find("title")
                if link_el is not None and link_el.text:
                    link = link_el.text.strip()
                    title = (title_el.text or "").strip() if title_el is not None else ""
                    found.setdefault(link, title)
            time.sleep(REQUEST_DELAY_SECONDS)
    return found


def collect_ntrs_pdf_candidates(
    client: httpx.Client, queries: list[str]
) -> dict[str, tuple[str, str]]:
    """Search NTRS for citations with a downloadable PDF. Returns url -> (title, id)."""
    found: dict[str, tuple[str, str]] = {}
    for query in queries:
        try:
            resp = client.get(
                NTRS_SEARCH_URL,
                params={"q": query, "page[size]": 50},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log(f"[warn] NTRS search failed for {query!r}: {exc}")
            continue
        for result in resp.json().get("results", []):
            if not result.get("downloadsAvailable"):
                continue
            for download in result.get("downloads", []):
                if download.get("mimetype") == "application/pdf":
                    pdf_url = NTRS_BASE_URL + download["links"]["pdf"]
                    found.setdefault(pdf_url, (result["title"], str(result["id"])))
                    break
        time.sleep(REQUEST_DELAY_SECONDS)
    return found


def extract_html_text(raw_html: str) -> tuple[str, str]:
    """Convert an article page into (title, structure-aware text).

    Headings are kept as markdown-style '#' prefixed lines so the chunker
    (Phase 1B) can detect section boundaries; everything else is paragraph text.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    title = re.sub(r"\s*[-|]\s*NASA.*$", "", title).strip()

    article = soup.find("article") or soup.find(class_="entry-content") or soup.body
    if article is None:
        return title, ""

    for junk in article.find_all(["script", "style", "nav", "aside", "form", "figure", "footer"]):
        junk.decompose()

    lines: list[str] = []
    for el in article.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if len(text) < 2:
            continue
        if el.name[0] == "h" and el.name[1].isdigit():
            level = int(el.name[1])
            lines.append(f"{'#' * level} {text}")
        else:
            lines.append(text)
    return title, "\n\n".join(lines)


def extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def fetch_and_store(
    client: httpx.Client, candidate: Candidate, existing_urls: set[str]
) -> dict | None:
    if candidate.url in existing_urls:
        return None

    try:
        resp = client.get(
            candidate.url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log(f"[skip] request failed for {candidate.url}: {exc}")
        return None

    title = candidate.title
    if candidate.kind == "html":
        extracted_title, text = extract_html_text(resp.text)
        title = extracted_title or title
    else:
        try:
            reader = PdfReader(io.BytesIO(resp.content))
        except Exception as exc:  # noqa: BLE001 - malformed PDFs vary widely
            log(f"[skip] unreadable PDF at {candidate.url}: {exc}")
            return None
        if len(reader.pages) > MAX_PDF_PAGES:
            log(f"[skip] PDF too long ({len(reader.pages)} pages) at {candidate.url}")
            return None
        text = extract_pdf_text(resp.content)

    if len(text) < MIN_EXTRACTED_CHARS:
        log(f"[skip] extracted text too short for {candidate.url}")
        return None

    doc_id = stable_id(candidate.url, title)
    filename = f"{doc_id}.txt"
    (RAW_DIR / filename).write_text(text, encoding="utf-8")

    return {
        "id": doc_id,
        "title": title,
        "source_url": candidate.url,
        "source_family": candidate.source_family,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "local_filename": filename,
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    existing_urls = {entry["source_url"] for entry in manifest}
    existing_ids = {entry["id"] for entry in manifest}

    client = httpx.Client(headers={"User-Agent": USER_AGENT})

    log("Discovering candidates...")
    jwst_found = collect_science_candidates(client, JWST_QUERIES, "jwst", JWST_URL_MARKERS)
    mars_found = collect_science_candidates(client, MARS_QUERIES, "mars", MARS_URL_MARKERS)
    general_found = collect_nasa_feed_candidates(client, NASA_GENERAL_FEEDS, NASA_GENERAL_FEED_PAGES)
    pdf_found = collect_ntrs_pdf_candidates(client, NTRS_QUERIES)

    candidates: list[Candidate] = []
    for url, title in list(jwst_found.items())[: TARGET_COUNTS["jwst"]]:
        candidates.append(Candidate(url, title, "jwst", "html"))
    for url, title in list(mars_found.items())[: TARGET_COUNTS["mars"]]:
        candidates.append(Candidate(url, title, "mars", "html"))
    for url, title in list(general_found.items())[: TARGET_COUNTS["nasa_general_html"]]:
        candidates.append(Candidate(url, title, "nasa_general", "html"))
    for url, (title, _ntrs_id) in list(pdf_found.items())[: TARGET_COUNTS["nasa_general_pdf"]]:
        candidates.append(Candidate(url, title, "nasa_general", "pdf"))

    log(f"Found {len(candidates)} candidates ({len(existing_urls)} already in manifest).")

    fetched_this_run = 0
    for candidate in candidates:
        if candidate.url in existing_urls:
            continue
        entry = fetch_and_store(client, candidate, existing_urls)
        time.sleep(REQUEST_DELAY_SECONDS)
        if entry is None:
            continue
        if entry["id"] in existing_ids:
            # Extremely unlikely hash collision across different URLs; skip rather than overwrite.
            log(f"[skip] duplicate id {entry['id']} for {candidate.url}")
            continue
        manifest.append(entry)
        existing_urls.add(entry["source_url"])
        existing_ids.add(entry["id"])
        fetched_this_run += 1
        save_manifest(manifest)  # incremental save so a crash mid-run doesn't lose progress
        log(f"[ok] {entry['source_family']:>12} | {entry['title'][:70]}")

    by_family: dict[str, int] = {}
    for entry in manifest:
        by_family[entry["source_family"]] = by_family.get(entry["source_family"], 0) + 1

    log("")
    log(f"Fetched {fetched_this_run} new document(s) this run.")
    log(f"Manifest now has {len(manifest)} document(s) total: {by_family}")


if __name__ == "__main__":
    main()
