"""
Phase 1 — Document metadata builder.

Produces a metadata dict that fits the RAG_Data_Management_Framework.xlsx
Document_Registry sheet (Document_Type, Category, Sub_Category, Source,
Date_Published, Page_Count, Language, Contains_Tables, Contains_Images,
Confidentiality, Quality_Score, Notes, ...).

Inputs:
  - The Tavily search hit (URL, title, snippet, score, query)
  - The ExtractedDocument (raw bytes for PDFs, plain text for HTML)

Output: a dict suitable for `Document(**metadata)` plus a few extras consumed
by the storage layer (file_name / file_extension / mime).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

# Category taxonomy aligned with RAG_Data_Management_Framework → Category_Taxonomy
CATEGORY_RULES = [
    # (predicate(url, title, text), category, sub_category, document_type)
    (lambda u, t, x: "sec.gov" in u or "10-k" in t or "10-q" in t or "8-k" in t,
     "Financial Data", "SEC Filing", "PDF"),
    (lambda u, t, x: "annual report" in t or "annual-report" in u,
     "Financial Data", "Annual Reports", "PDF"),
    (lambda u, t, x: "press release" in t or "pressrelease" in u or "/press/" in u,
     "Industry Reports", "Press Release", "HTML"),
    (lambda u, t, x: "energy.gov" in u or "doe.gov" in u,
     "Industry Reports", "Energy", "PDF"),
    (lambda u, t, x: "georgia.org" in u or "selectgeorgia" in u or "gdecd" in u,
     "Industry Reports", "Economic Development", "HTML"),
    (lambda u, t, x: any(d in u for d in ("reuters.com", "bloomberg.com", "autonews.com", "wsj.com")),
     "News & Media", "News Article", "HTML"),
    (lambda u, t, x: u.endswith(".pdf") or "/pdf/" in u,
     "Industry Reports", "Automotive", "PDF"),
    (lambda u, t, x: "battery" in t or "ev " in t or "electric vehicle" in t,
     "Industry Reports", "EV Battery", "HTML"),
]

DEFAULT_CATEGORY = ("Web Content", "Company Websites", "HTML")

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_DATE_RE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
_TABLE_HINTS = ("<table", "table of contents", "| ---", "\t\t")
_IMAGE_HINTS = ("<img ", ".png", ".jpg", ".jpeg", "figure ", "image:")


def _slugify(text: str) -> str:
    return _SLUG_RE.sub("_", (text or "").lower()).strip("_")[:80]


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _file_name_from_url(url: str, fallback: str) -> str:
    try:
        path = urlparse(url).path
        leaf = path.rsplit("/", 1)[-1]
        if leaf and "." in leaf:
            return leaf[:200]
    except Exception:
        pass
    return f"{_slugify(fallback) or 'document'}"


def _extension_for(content_type: str, url: str) -> str:
    """
    Resolve a canonical file extension. Prefers `content_type` (set by the
    extractor — "pdf"/"html"/"json"/"csv"/"txt") over the URL hint, so a
    .pdf URL that fell back to Tavily Extract (HTML text) is correctly
    tagged as .html instead of .pdf.
    """
    ct = (content_type or "").lower()
    if ct:
        if "pdf" in ct:
            return ".pdf"
        if "html" in ct:
            return ".html"
        if "json" in ct:
            return ".json"
        if "csv" in ct:
            return ".csv"
        if "txt" in ct or "text" in ct:
            return ".txt"
    # Fall back to URL hint only when content_type is empty/unknown.
    if url.lower().endswith(".pdf"):
        return ".pdf"
    return ".txt"


def _mime_for_extension(extension: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json",
        ".csv": "text/csv",
        ".txt": "text/plain; charset=utf-8",
    }.get(extension, "application/octet-stream")


def _classify(url: str, title: str, text: str) -> tuple[str, str, str]:
    url_l = (url or "").lower()
    title_l = (title or "").lower()
    text_l = (text or "")[:2000].lower()
    for predicate, cat, sub, doctype in CATEGORY_RULES:
        try:
            if predicate(url_l, title_l + " " + text_l, text_l):
                return cat, sub, doctype
        except Exception:
            continue
    return DEFAULT_CATEGORY


def _sniff_published_date(text: str) -> datetime | None:
    if not text:
        return None
    match = _DATE_RE.search(text[:5000])
    if not match:
        return None
    try:
        y, m, d = (int(x) for x in match.groups())
        if 1990 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
            return datetime(y, m, d)
    except (ValueError, TypeError):
        return None
    return None


def _quality_score(word_count: int, has_title: bool, has_date: bool, relevance: float) -> float:
    """
    0..100 score reflecting how useful the doc is for the KB.
    Inputs are easy to compute and align with the RAG framework Quality_Score
    column (0..100).
    """
    score = 40.0  # baseline for any extracted doc
    if word_count >= 300:
        score += 15
    if word_count >= 1000:
        score += 10
    if word_count >= 5000:
        score += 5
    if has_title:
        score += 10
    if has_date:
        score += 5
    score += min(15.0, max(0.0, relevance * 15.0))
    return round(min(score, 100.0), 1)


def _document_uid(url: str, content_hash: str) -> str:
    seed = (content_hash or url).encode("utf-8")
    digest = hashlib.sha256(seed).hexdigest()[:10].upper()
    return f"DOC_{digest}"


def build_document_metadata(
    *,
    extracted: Any,
    company: dict[str, Any] | None,
    search_hit: dict[str, Any] | None,
    b2_bucket: str,
) -> dict[str, Any]:
    """
    Build the metadata dict for a Document row.

    Args:
        extracted: ExtractedDocument from extractor.py.
        company: company dict from kb_loader (id, company_name, ...).
        search_hit: original Tavily search result dict (with title/snippet/score/query).
        b2_bucket: name of the B2 bucket the file lands in.
    """
    company = company or {}
    hit = search_hit or {}

    url = extracted.url
    title = (hit.get("title") or "").strip() or extracted.url
    snippet = (hit.get("snippet") or "").strip()
    text_preview = extracted.text or ""

    category, sub_category, document_type = _classify(url, title, text_preview)

    extension = _extension_for(extracted.content_type, url)
    file_name = _file_name_from_url(url, fallback=title or "document")
    if extension and not file_name.endswith(extension):
        file_name = f"{file_name.rsplit('.', 1)[0]}{extension}"

    raw_size = extracted.raw_bytes_size or len((extracted.text or "").encode("utf-8"))
    contains_tables = any(h in text_preview.lower() for h in _TABLE_HINTS)
    contains_images = any(h in text_preview.lower() for h in _IMAGE_HINTS) or extension == ".pdf"
    page_count = None
    if extension == ".pdf" and extracted.raw_bytes:
        try:
            import fitz
            doc = fitz.open(stream=extracted.raw_bytes, filetype="pdf")
            page_count = doc.page_count
            doc.close()
        except Exception:
            page_count = None

    date_published = _sniff_published_date(text_preview)
    quality = _quality_score(
        word_count=extracted.word_count,
        has_title=bool(title and title != extracted.url),
        has_date=date_published is not None,
        relevance=hit.get("score", 0.0) or 0.0,
    )

    return {
        # External-facing
        "document_uid": _document_uid(url, extracted.content_hash),
        "document_name": title[:500],
        "document_type": document_type,
        "category": category,
        "sub_category": sub_category,
        # Linkage
        "company_id": company.get("id"),
        "company_name": company.get("company_name") or extracted.company_name,
        # Provenance
        "source": _domain(url) or "tavily",
        "source_url": url,
        "search_query": (hit.get("query") or "")[:500] or None,
        "search_query_family": hit.get("family"),
        "relevance_score": float(hit.get("score") or 0.0),
        # File / B2
        "b2_bucket": b2_bucket,
        "file_name": file_name[:500],
        "file_extension": extension,
        "content_type": _mime_for_extension(extension),
        "content_hash_sha256": extracted.content_hash,
        "file_size_bytes": raw_size,
        "file_size_mb": round(raw_size / (1024 * 1024), 4) if raw_size else None,
        # Body
        "page_count": page_count,
        "word_count": extracted.word_count,
        "char_count": extracted.char_count,
        "language": "English",
        "contains_tables": contains_tables,
        "contains_images": contains_images,
        # Dates
        "date_published": date_published,
        "date_acquired": datetime.utcnow(),
        "date_processed": datetime.utcnow(),
        "downloaded_at": datetime.utcnow(),
        "extracted_at": datetime.utcnow(),
        # Governance
        "confidentiality": "Public",
        "retention_period": "Permanent",
        "owner": "Georgia EV Intelligence Pipeline",
        # Pipeline state
        "extraction_method": extracted.extraction_method or None,
        "extraction_status": "extracted",
        "processing_status": "Completed",
        "quality_score": quality,
        "notes": (snippet[:500] or None),
    }


def metadata_for_failed(
    *,
    url: str,
    error: str,
    company: dict[str, Any] | None,
    search_hit: dict[str, Any] | None,
    b2_bucket: str,
) -> dict[str, Any]:
    """Compact metadata row for a failed extraction (no body fields)."""
    company = company or {}
    hit = search_hit or {}
    title = (hit.get("title") or url)[:500]
    extension = _extension_for("", url)
    return {
        "document_uid": _document_uid(url, ""),
        "document_name": title,
        "document_type": extension.lstrip(".").upper() or "UNKNOWN",
        "category": "Unknown",
        "sub_category": "Unknown",
        "company_id": company.get("id"),
        "company_name": company.get("company_name"),
        "source": _domain(url) or "tavily",
        "source_url": url,
        "search_query": (hit.get("query") or "")[:500] or None,
        "search_query_family": hit.get("family"),
        "relevance_score": float(hit.get("score") or 0.0),
        "b2_bucket": b2_bucket,
        "file_name": _file_name_from_url(url, fallback=title),
        "file_extension": extension,
        "content_type": _mime_for_extension(extension),
        "extraction_method": None,
        "extraction_status": "failed",
        "processing_status": "Failed",
        "extraction_error": error[:500] if error else None,
        "date_acquired": datetime.utcnow(),
        "downloaded_at": datetime.utcnow(),
        "language": "English",
        "confidentiality": "Public",
        "retention_period": "Permanent",
        "owner": "Georgia EV Intelligence Pipeline",
        "quality_score": 0.0,
    }
