"""
Phase 1 — Extractor

Three extraction paths, in order of preference for any given URL:

  1. URL ends in .pdf  → free direct HTTP GET → PyMuPDF parse
  2. URL points at a non-PDF resource we can fetch directly (HTML, .txt,
     .json, .csv) → free direct HTTP GET → store raw + best-effort text
  3. JS-rendered or paywalled HTML → Tavily Extract (rotates keys)

"Free download" means: we always attempt a plain GET first, so when a URL
points at a file that's freely available on the open web we never burn a
Tavily credit. Tavily Extract is reserved for pages that require JS rendering
or block direct scrapers.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import httpx

from shared.logger import get_logger
from shared.tavily import TavilyAllKeysExhausted, async_tavily_post

logger = get_logger("phase1.extractor")

TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

_PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}
_DOWNLOADABLE_PREFIXES = (
    "application/pdf",
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml",
)

# Minimum content length to consider a document useful
MIN_CONTENT_CHARS = 150

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GeorgiaEVIntel/1.0; "
        "+https://github.com/anthropics/) Python-httpx"
    ),
    "Accept": "*/*",
}


@dataclass
class ExtractedDocument:
    """Result of extracting text from one URL."""
    url: str
    company_name: str
    content_type: str           # "pdf" / "html" / "json" / "csv" / "txt"
    text: str                   # Extracted clean text
    title: str = ""
    word_count: int = 0
    char_count: int = 0
    content_hash: str = ""      # SHA-256 of raw bytes (or text if no bytes)
    raw_bytes: bytes = field(default_factory=bytes)
    raw_bytes_size: int = 0
    extraction_method: str = "" # "pymupdf" / "tavily_extract" / "direct_html" / "direct_text"
    error: str = ""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_pdf_url(url: str) -> bool:
    url_lower = url.lower()
    return url_lower.endswith(".pdf") or "/pdf/" in url_lower


def _is_pdf_bytes(content: bytes) -> bool:
    return content[:4] == b"%PDF"


def _is_pdf_content_type(content_type: str) -> bool:
    return content_type.lower().split(";")[0].strip() in _PDF_CONTENT_TYPES


def _is_downloadable(content_type: str) -> bool:
    ct = (content_type or "").lower()
    return any(ct.startswith(prefix) for prefix in _DOWNLOADABLE_PREFIXES)


def extract_pdf_bytes(pdf_bytes: bytes, url: str) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf") from exc

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: list[str] = []
        for page in doc:
            text = page.get_text("text")
            if text and text.strip():
                pages.append(text.strip())
        doc.close()
        return "\n\n".join(pages)
    except Exception as exc:
        logger.warning("PyMuPDF failed for %s: %s", url, exc)
        return ""


def _strip_html(content: bytes) -> tuple[str, str]:
    """
    Best-effort HTML → text without an external dep.
    Returns (title, text).
    """
    try:
        text = content.decode("utf-8", errors="ignore")
    except Exception:
        return "", ""

    title = ""
    title_open = text.lower().find("<title")
    if title_open != -1:
        title_close = text.find(">", title_open)
        title_end = text.lower().find("</title>", title_close)
        if title_close != -1 and title_end != -1:
            title = text[title_close + 1: title_end].strip()

    # Drop script/style blocks
    import re
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text,
                     flags=re.IGNORECASE | re.DOTALL)
    # Strip remaining tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return title, cleaned


async def http_download(
    url: str,
    timeout: float = 30.0,
    max_bytes: int = 50 * 1024 * 1024,
) -> tuple[bytes, str]:
    """
    Free, direct HTTP download. Returns (bytes, content_type).
    Raises httpx.HTTPStatusError on non-2xx.
    """
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=DEFAULT_HEADERS,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        if len(response.content) > max_bytes:
            raise ValueError(f"File exceeds {max_bytes} bytes — refusing download")
        return response.content, response.headers.get("content-type", "") or ""


async def tavily_extract(url: str) -> dict[str, Any]:
    """
    Tavily Extract API call. Rotates keys via shared.tavily.
    Raises TavilyAllKeysExhausted when every key is disabled.
    """
    data = await async_tavily_post(
        TAVILY_EXTRACT_URL,
        {"urls": [url]},
        timeout=60.0,
    )
    results = data.get("results") or []
    if results:
        return results[0]
    failed = data.get("failed_results") or []
    if failed:
        raise ValueError(f"Tavily extract failed for {url}: {failed[0].get('error', 'unknown')}")
    raise ValueError(f"Tavily extract returned no results for {url}")


async def extract_document(
    url: str,
    company_name: str,
    force_pdf: bool = False,
) -> ExtractedDocument:
    """
    Main entry point: extract text + raw bytes from any URL.
    Free direct HTTP first; Tavily Extract only if the page can't be parsed.
    """
    is_pdf_hint = force_pdf or _is_pdf_url(url)
    if is_pdf_hint:
        return await _extract_via_direct(url, company_name, prefer_pdf=True)
    return await _extract_via_direct(url, company_name, prefer_pdf=False)


async def _extract_via_direct(
    url: str, company_name: str, prefer_pdf: bool,
) -> ExtractedDocument:
    """Try a plain HTTP GET; if it works, parse it; otherwise fall through to Tavily."""
    try:
        raw_bytes, content_type = await http_download(url)
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("Direct download failed for %s (%s) — falling back to Tavily Extract", url, exc)
        return await _extract_via_tavily(url, company_name)

    looks_like_pdf = _is_pdf_bytes(raw_bytes) or _is_pdf_content_type(content_type) or prefer_pdf

    if looks_like_pdf and _is_pdf_bytes(raw_bytes):
        text = extract_pdf_bytes(raw_bytes, url)
        if text and len(text) >= MIN_CONTENT_CHARS:
            content_hash = _sha256(raw_bytes)
            logger.info("  [OK] PDF %d words extracted (%s)", len(text.split()), url[:80])
            return ExtractedDocument(
                url=url,
                company_name=company_name,
                content_type="pdf",
                text=text,
                word_count=len(text.split()),
                char_count=len(text),
                content_hash=content_hash,
                raw_bytes=raw_bytes,
                raw_bytes_size=len(raw_bytes),
                extraction_method="pymupdf",
            )
        logger.info("PDF parsed empty for %s — trying Tavily Extract instead", url)
        return await _extract_via_tavily(url, company_name)

    # Non-PDF: simple HTML / text extraction
    if _is_downloadable(content_type) or content_type == "":
        ct_lower = (content_type or "").lower()
        if ct_lower.startswith("text/html") or b"<html" in raw_bytes[:200].lower():
            title, body = _strip_html(raw_bytes)
            if body and len(body) >= MIN_CONTENT_CHARS:
                content_hash = _sha256(raw_bytes)
                logger.info("  [OK] HTML(direct) %d words extracted (%s)", len(body.split()), url[:80])
                return ExtractedDocument(
                    url=url,
                    company_name=company_name,
                    content_type="html",
                    text=body,
                    title=title,
                    word_count=len(body.split()),
                    char_count=len(body),
                    content_hash=content_hash,
                    raw_bytes=raw_bytes,
                    raw_bytes_size=len(raw_bytes),
                    extraction_method="direct_html",
                )
            # Direct HTML scrape too thin — Tavily will likely render JS better
            return await _extract_via_tavily(url, company_name)

        # Plain-text style content (json/csv/txt)
        try:
            text = raw_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            text = ""
        if text and len(text) >= MIN_CONTENT_CHARS:
            content_hash = _sha256(raw_bytes)
            kind = "json" if "json" in content_type else "csv" if "csv" in content_type else "txt"
            logger.info("  [OK] %s(direct) %d words (%s)", kind.upper(), len(text.split()), url[:80])
            return ExtractedDocument(
                url=url,
                company_name=company_name,
                content_type=kind,
                text=text,
                word_count=len(text.split()),
                char_count=len(text),
                content_hash=content_hash,
                raw_bytes=raw_bytes,
                raw_bytes_size=len(raw_bytes),
                extraction_method="direct_text",
            )

    # Anything else — let Tavily try to make sense of it
    return await _extract_via_tavily(url, company_name)


async def _extract_via_tavily(url: str, company_name: str) -> ExtractedDocument:
    logger.info("[TAVILY] %s", url[:100])
    try:
        result = await tavily_extract(url)
    except TavilyAllKeysExhausted:
        # Surface to the pipeline so it can checkpoint and stop.
        raise
    except Exception as exc:
        return ExtractedDocument(
            url=url, company_name=company_name, content_type="html", text="",
            error=str(exc)[:200], extraction_method="tavily_extract",
        )

    text = (result.get("raw_content") or "").strip()
    if not text or len(text) < MIN_CONTENT_CHARS:
        return ExtractedDocument(
            url=url,
            company_name=company_name,
            content_type="html",
            text="",
            error=f"Tavily extract returned short/empty content (len={len(text)})",
            extraction_method="tavily_extract",
        )

    content_bytes = text.encode("utf-8")
    content_hash = _sha256(content_bytes)
    logger.info("  [OK] HTML(tavily) %d words (%s)", len(text.split()), url[:80])
    return ExtractedDocument(
        url=url,
        company_name=company_name,
        content_type="html",
        text=text,
        word_count=len(text.split()),
        char_count=len(text),
        content_hash=content_hash,
        raw_bytes=content_bytes,
        raw_bytes_size=len(content_bytes),
        extraction_method="tavily_extract",
    )
