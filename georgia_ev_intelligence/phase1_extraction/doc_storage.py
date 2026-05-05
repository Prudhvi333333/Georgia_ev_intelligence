"""
Phase 1 — Document Storage

Saves extracted documents to:
  - Backblaze B2  (raw content / PDF bytes / direct HTML / Tavily text)
  - PostgreSQL    (rich metadata row per document — schema mirrors the
                   RAG_Data_Management_Framework Document_Registry sheet)

Deduplication keyed on SHA-256 content hash (same content from a different URL
is reused, B2 isn't double-uploaded).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from shared.config import Config
from shared.db import Document, get_session
from shared.logger import get_logger
from shared.storage import key_exists, make_document_key, upload_bytes
from phase1_extraction.extractor import ExtractedDocument
from phase1_extraction.metadata import build_document_metadata, metadata_for_failed

logger = get_logger("phase1.doc_storage")


def _build_b2_key(extracted: ExtractedDocument, extension: str) -> str | None:
    if not extracted.content_hash:
        return None
    ext = extension.lstrip(".") or ("pdf" if extracted.content_type == "pdf" else "txt")
    return make_document_key(extracted.company_name, extracted.content_hash, ext)


def _content_for_upload(extracted: ExtractedDocument) -> tuple[bytes, str]:
    """Return (bytes_to_upload, mime). Prefer raw_bytes; fall back to text."""
    if extracted.raw_bytes:
        if extracted.content_type == "pdf":
            return extracted.raw_bytes, "application/pdf"
        if extracted.content_type == "html":
            return extracted.raw_bytes, "text/html; charset=utf-8"
        if extracted.content_type == "json":
            return extracted.raw_bytes, "application/json"
        if extracted.content_type == "csv":
            return extracted.raw_bytes, "text/csv"
        return extracted.raw_bytes, "text/plain; charset=utf-8"
    return extracted.text.encode("utf-8"), "text/plain; charset=utf-8"


def save_document(
    extracted: ExtractedDocument,
    company: dict[str, Any] | None,
    search_hit: dict[str, Any] | None = None,
) -> int | None:
    """
    Save an extracted document to B2 + PostgreSQL.

    Args:
        extracted   : ExtractedDocument from extractor.py.
        company     : kb_loader company dict (id, company_name, ...).
        search_hit  : Tavily search result that found this URL (carries
                      title, snippet, score, query, family). Optional but
                      recommended — drives metadata provenance.

    Returns the saved document_id on success, None on failure.
    """
    if not extracted.text or extracted.error:
        logger.debug("Skipping empty/failed extraction for %s", extracted.url)
        return None

    cfg = Config.get()
    bucket = cfg.b2_bucket

    metadata = build_document_metadata(
        extracted=extracted,
        company=company,
        search_hit=search_hit,
        b2_bucket=bucket,
    )

    session = get_session()
    try:
        # Dedup by SHA-256 content hash
        if extracted.content_hash:
            existing_by_hash = (
                session.query(Document)
                .filter_by(content_hash_sha256=extracted.content_hash)
                .first()
            )
            if existing_by_hash:
                logger.debug(
                    "Duplicate content for %s (hash matches doc %d) — skipping",
                    extracted.url, existing_by_hash.id,
                )
                return existing_by_hash.id

        # Upload to B2 (idempotent — head_object before put)
        b2_key = _build_b2_key(extracted, metadata["file_extension"])
        if b2_key:
            if not key_exists(b2_key):
                content_bytes, mime = _content_for_upload(extracted)
                upload_bytes(content_bytes, b2_key, mime)
                logger.info(
                    "B2 upload OK [%s] %.1fKB → %s",
                    extracted.content_type.upper(), len(content_bytes) / 1024, b2_key,
                )
            else:
                logger.info("B2 key already exists (dedup skipped): %s", b2_key)
        metadata["b2_key"] = b2_key

        # Upsert by source_url
        existing_by_url = (
            session.query(Document).filter_by(source_url=extracted.url).first()
        )
        if existing_by_url:
            for field, value in metadata.items():
                if value is None and getattr(existing_by_url, field, None) is not None:
                    continue
                setattr(existing_by_url, field, value)
            existing_by_url.updated_at = datetime.utcnow()
            session.commit()
            logger.info("DB updated document #%d → %s", existing_by_url.id, extracted.url[:80])
            return existing_by_url.id

        doc = Document(**metadata)
        session.add(doc)
        session.commit()
        session.refresh(doc)
        logger.info(
            "DB saved document #%d [%s/%s] %d words → %s",
            doc.id, metadata["category"], metadata["sub_category"],
            extracted.word_count, extracted.url[:80],
        )
        return doc.id

    except Exception as exc:
        session.rollback()
        logger.error("Failed to save document for %s: %s", extracted.url, exc)
        return None
    finally:
        session.close()


def mark_document_failed(
    url: str,
    error: str,
    company: dict[str, Any] | None = None,
    search_hit: dict[str, Any] | None = None,
) -> None:
    """Record a failed extraction attempt with full RAG-framework metadata."""
    cfg = Config.get()
    bucket = cfg.b2_bucket
    metadata = metadata_for_failed(
        url=url, error=error, company=company,
        search_hit=search_hit, b2_bucket=bucket,
    )

    session = get_session()
    try:
        existing = session.query(Document).filter_by(source_url=url).first()
        if existing:
            existing.extraction_status = "failed"
            existing.extraction_error = (error or "")[:500]
            existing.processing_status = "Failed"
        else:
            session.add(Document(**metadata))
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Could not record failed document %s: %s", url, exc)
    finally:
        session.close()


def get_document_count_for_company(company_name: str) -> int:
    """How many successfully extracted documents does this company have?"""
    session = get_session()
    try:
        return (
            session.query(Document)
            .filter_by(company_name=company_name, extraction_status="extracted")
            .count()
        )
    finally:
        session.close()
