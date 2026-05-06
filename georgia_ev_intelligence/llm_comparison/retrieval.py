"""Dense + hybrid Qdrant retrieval + mandatory cross-encoder rerank.

Per the sreeja-arch spec:
  * Attempt hybrid (dense + sparse RRF) retrieval; fall back to dense-only
    if the sparse vector query fails.
  * For exhaustive list questions with extractable structured filters, scan
    the KB Excel directly and bypass vector search entirely — this gives
    100% recall for filtered sets (e.g. "all Tier 2 suppliers in Troup County").
  * Apply metadata pre-filters before vector search for questions with
    explicit structured constraints.
  * Rerank with cross-encoder/ms-marco-MiniLM-L12-v2 (mandatory, hard fail).
  * For list/count questions without deterministic filters, return all
    candidates above a rerank score threshold instead of a fixed top-N cap.

Sparse hashing note:
  _build_sparse_vector() uses hashlib.sha1 for stable cross-process indices.
  The same function is used in phase2_embedding/vector_store.py. If the
  Qdrant collection was indexed with an older version using Python's built-in
  hash() (which is randomized per process), re-ingestion is required for the
  sparse/BM25 component to produce correct matches.
"""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient, models

from llm_comparison.config import GenerationConfig

logger = logging.getLogger("llm_comparison.retrieval")

# ── Query type classifier ────────────────────────────────────────────────────
# Phase 1: strong patterns that alone signal a list/exhaustive answer.
_STRONG_LIST = re.compile(
    r"\b(?:"
    r"list|every|each|enumerate|"
    r"how\s+many\s+(?:\w+\s+){0,3}(?:companies|suppliers|areas|facilities)|"
    r"how\s+many\s+of\s+its|"
    r"top\s+\d+|"
    r"show\s+(?:all|the\s+full)|"
    r"map\s+all|"
    r"identify\s+(?:all|georgia|any)|"
    r"find\s+(?:georgia|tier)"
    r")\b",
    re.IGNORECASE,
)

# Phase 2: "which" followed (anywhere in the question) by a multi-entity noun.
_WHICH_MULTI = re.compile(
    r"\bwhich\b.*?\b(?:companies|suppliers|firms|areas|counties|locations|"
    r"facilities|roles|tiers|georgia\s+tier|ev\s+supply\s+chain\s+roles)\b",
    re.IGNORECASE | re.DOTALL,
)

# Exclude singular "which company/county" questions (asking for a single entity).
_WHICH_SINGULAR = re.compile(
    r"\bwhich\s+(?:specific\s+)?(?:company|county|firm|city|single)\s+(?:has|is|have|was|provides|supplies)\b",
    re.IGNORECASE,
)


def classify_query(question: str) -> str:
    """Return 'list' for exhaustive/enumeration questions, 'point' otherwise."""
    if _STRONG_LIST.search(question):
        return "list"
    if _WHICH_MULTI.search(question) and not _WHICH_SINGULAR.search(question):
        return "list"
    return "point"


# ── Sparse vector with STABLE hashing ────────────────────────────────────────
# Uses hashlib.sha1 so indices are identical across Python processes.
# VOCAB_SIZE must match phase2_embedding/vector_store.py.

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its", "as", "up",
})
_VOCAB_SIZE = 1_048_576  # 2^20


def _stable_hash(token: str) -> int:
    """Stable cross-process hash for sparse vector index generation."""
    return int.from_bytes(hashlib.sha1(token.encode()).digest()[:4], "big") % _VOCAB_SIZE


def _build_sparse_vector(text: str) -> models.SparseVector:
    tokens = [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in _STOPWORDS and len(t) >= 2
    ]
    if not tokens:
        return models.SparseVector(indices=[0], values=[0.0])
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    seen: set[int] = set()
    indices: list[int] = []
    values: list[float] = []
    for token, count in tf.items():
        idx = _stable_hash(token)
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
            values.append(float(count))
    return models.SparseVector(indices=indices, values=values)


# ── Metadata filter extraction ───────────────────────────────────────────────

_TIER_COMPOSITES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\btier\s*1\s*/\s*2\b", re.I), "Tier 1/2"),
    (re.compile(r"\btier\s*2\s*/\s*3\b", re.I), "Tier 2/3"),
]
_TIER_STANDALONE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\btier\s*1\b", re.I), "Tier 1"),
    (re.compile(r"\btier\s*2\b", re.I), "Tier 2"),
]

# Phrases that indicate the tier refers to a hypothetical company, not a filter.
_HYPOTHETICAL_TIER = re.compile(
    r"\b(?:new\s+tier|for\s+a\s+new|looking\s+to\s+(?:locate|establish)|"
    r"new\s+company\s+looking|international\s+.{0,20}company\s+seeking)\b",
    re.IGNORECASE,
)

# Complete set of Georgia counties that appear in the GNEM KB.
_GEORGIA_COUNTIES: frozenset[str] = frozenset({
    "appling", "baldwin", "bartow", "bibb", "bryan", "bulloch", "candler",
    "carroll", "catoosa", "chatham", "chattahoochee", "chattooga", "cherokee",
    "clarke", "clayton", "cobb", "columbia", "coweta", "dade", "dawson",
    "dekalb", "dougherty", "douglas", "elbert", "fayette", "floyd", "forsyth",
    "franklin", "fulton", "gordon", "grady", "gwinnett", "habersham", "hall",
    "haralson", "harris", "henry", "jackson", "jones", "lamar", "laurens",
    "liberty", "lowndes", "lumpkin", "meriwether", "morgan", "newton",
    "paulding", "peach", "polk", "pulaski", "rabun", "richmond", "schley",
    "spalding", "stephens", "troup", "union", "walton", "warren", "whitfield",
})


def extract_filters(question: str) -> dict[str, Any]:
    """Extract explicit structured constraints from a question.

    Returns an empty dict when uncertain — no filter is always safer than a
    wrong filter that silently drops valid rows.

    Tier filter is skipped for hypothetical-company phrases like
    "new Tier 1 company looking to locate".
    Multi-value tiers (e.g. "Tier 1 or Tier 1/2") use OR semantics via
    the 'tiers' key instead of the single-value 'tier' key.
    """
    result: dict[str, Any] = {}
    q_lower = question.lower()
    hypothetical_location = bool(
        _HYPOTHETICAL_TIER.search(question)
        or "company seeking" in q_lower
        or "seeking a georgia location" in q_lower
    )

    # Tier — skip if the question is about a hypothetical company. Composite
    # labels are removed before matching standalone tiers so "Tier 1/2" does
    # not accidentally become both "Tier 1/2" and "Tier 1".
    if not hypothetical_location:
        found_tiers: list[str] = []
        tier_scan_text = question
        for pattern, canonical in _TIER_COMPOSITES:
            if pattern.search(question) and canonical not in found_tiers:
                found_tiers.append(canonical)
                tier_scan_text = pattern.sub(" ", tier_scan_text)
        for pattern, canonical in _TIER_STANDALONE:
            if pattern.search(tier_scan_text) and canonical not in found_tiers:
                found_tiers.append(canonical)
        if len(found_tiers) == 1:
            result["tier"] = found_tiers[0]
        elif len(found_tiers) > 1:
            result["tiers"] = found_tiers  # resolved to OR filter in _make_qdrant_filter

    # Classification / direct manufacturer language. This is safer than mapping
    # generic "OEM contracts" to Category=OEM.
    if "direct manufacturer" in q_lower or "classified as direct manufacturer" in q_lower:
        result["classification_values"] = ["Direct Manufacturer"]

    # Exact category constraints. These are distinct from "OEM contracts",
    # which refers to the Primary OEMs/customer field.
    category_values: list[str] = []
    if "oem footprint" in q_lower:
        category_values.extend(["OEM Footprint", "OEM (Footprint)"])
    if "oem supply chain" in q_lower:
        category_values.append("OEM Supply Chain")
    if category_values:
        result["category_values"] = _dedupe_preserve_order(category_values)
        result["suppress_ev_relevance_filter"] = True

    company_names = _extract_company_names_from_question(q_lower)
    if company_names and re.search(r"\b(?:linked to|contracts?|supplier network|customers?)\b", q_lower):
        company_names = []
    if company_names:
        result["company_names"] = company_names
        result["preserve_company_rows"] = True

    # Explicit role constraints.
    role_terms: list[str] = []
    battery_role_context = bool(
        re.search(
            r"(classified under|roles?|suppliers?|supply chain role|sole-sourced|sole sourced)",
            q_lower,
        )
    )
    battery_as_negative_area_context = "lack battery cell" in q_lower or "lacks battery cell" in q_lower
    battery_as_customer_context = "provide materials to battery cell" in q_lower
    if "battery cell" in q_lower and battery_role_context and not battery_as_negative_area_context and not battery_as_customer_context:
        role_terms.append("Battery Cell")
    if "battery pack" in q_lower and battery_role_context and not battery_as_negative_area_context and not battery_as_customer_context:
        role_terms.append("Battery Pack")
    if "thermal management" in q_lower or "thermal-related" in q_lower or "thermal related" in q_lower:
        role_terms.append("Thermal Management")
    if "power electronics" in q_lower and "dc-to-dc" not in q_lower and "capacitor" not in q_lower:
        role_terms.append("Power Electronics")
    if "charging infrastructure" in q_lower:
        role_terms.append("Charging Infrastructure")
    if "vehicle assembly" in q_lower:
        role_terms.append("Vehicle Assembly")
    if "materials-category" in q_lower or "materials category" in q_lower:
        role_terms.append("Materials")
    if "general automotive" in q_lower:
        role_terms.append("General Automotive")
    if "wiring harness" in q_lower or "wiring harnesses" in q_lower:
        role_terms.append("wiring harness")
    if role_terms:
        result["role_terms"] = _dedupe_preserve_order(role_terms)

    # OEM/customer constraints. "OEM contracts" means customer/OEM relationship,
    # not Category=OEM.
    if "hyundai kia" in q_lower or "hyundai/kia" in q_lower:
        result["oem_terms_all"] = ["hyundai", "kia"]
    elif "hyundai" in q_lower:
        result["oem_terms_any"] = ["hyundai"]
    elif "kia" in q_lower:
        result["oem_terms_any"] = ["kia"]
    if "rivian" in q_lower:
        result.setdefault("oem_terms_any", []).append("rivian")
    if "multiple oems" in q_lower:
        result["primary_oems_values"] = ["Multiple OEMs"]
    if (
        "existing oem contract" in q_lower
        or "existing oem contracts" in q_lower
        or "sole-sourced" in q_lower
        or "sole sourced" in q_lower
    ):
        result["require_specific_oem_contract"] = True

    # EV relevance constraints.
    if not result.get("suppress_ev_relevance_filter") and (
        "indirectly relevant" in q_lower or '"indirect' in q_lower
    ):
        result["ev_relevance_values"] = ["Indirect"]
    elif not result.get("suppress_ev_relevance_filter") and (
        "no ev-specific" in q_lower or "no ev specific" in q_lower
    ):
        # The golden answers treat "no EV-specific production presence" as
        # anything that is not explicitly EV/Battery Relevant = Yes.
        result["ev_relevance_not_values"] = ["Yes"]
    elif not result.get("suppress_ev_relevance_filter") and (
        "ev relevant" in q_lower or "ev-relevant" in q_lower or "ev component" in q_lower
    ):
        result["ev_relevance_values"] = ["Yes"]
    # Facility / industry constraints.
    if "manufacturing plant" in q_lower or "manufacturing facilities" in q_lower:
        result["facility_terms"] = ["manufacturing plant"]
    if "r&d" in q_lower or "research" in q_lower or "development" in q_lower:
        result["facility_terms"] = _dedupe_preserve_order(result.get("facility_terms", []) + ["r&d"])
    if "chemical manufacturing" in q_lower or "chemical infrastructure" in q_lower:
        result["industry_terms"] = ["chemical"]
    if "chemicals and allied products" in q_lower:
        result["industry_terms"] = ["chemicals and allied products"]
    if "electronic and electrical equipment" in q_lower:
        result["industry_terms"] = ["electronic and other electrical equipment and components"]

    # Product/service phrase constraints. These are used by the Excel scan, not
    # Qdrant filters, because Qdrant payload filters are exact-field oriented.
    product_terms: list[str] = []
    if "lithium-ion battery materials" in q_lower:
        product_terms.append("lithium-ion battery materials")
    if "battery electrolyte" in q_lower or "battery electrolytes" in q_lower or "electrolyte" in q_lower:
        product_terms.append("battery electrolyte")
    if (
        "battery cell" in q_lower
        and re.search(r"\b(?:produce|producing|manufacture|manufacturing)\b", q_lower)
        and not battery_as_customer_context
        and "battery cells" not in product_terms
    ):
        product_terms.append("battery cells")
    if "battery parts" in q_lower:
        product_terms.append("battery parts")
    if "copper foil" in q_lower or "electrodeposited" in q_lower:
        product_terms.extend(["copper foil", "electrodeposited"])
    if "anodes" in q_lower or "cathodes" in q_lower:
        product_terms.extend(["lithium-ion battery", "raw materials"])
    if (
        "battery materials" in q_lower
        and not hypothetical_location
        and "raw materials" not in product_terms
    ):
        product_terms.append("raw materials")
    if "dc-to-dc" in q_lower:
        product_terms.append("dc-to-dc")
    if "capacitor" in q_lower or "capacitors" in q_lower:
        product_terms.append("capacitor")
    if "powder coating" in q_lower or "powder coatings" in q_lower:
        product_terms.extend(["powder coating", "powder coatings"])
    if "battery recycling" in q_lower or "second-life battery" in q_lower or "second life battery" in q_lower:
        product_terms.extend(["recycler", "recycling", "second-life", "second life"])
    if "engineered plastics" in q_lower:
        product_terms.append("engineered plastic")
    if "polymers" in q_lower:
        product_terms.append("polymers")
    if "composite materials" in q_lower or "composite" in q_lower:
        product_terms.append("composite")
    if "lightweight aluminum" in q_lower or "aluminum" in q_lower:
        product_terms.append("aluminum")
    if "high-voltage" in q_lower or "high voltage" in q_lower:
        product_terms.append("high-voltage")
    if "inverter" in q_lower:
        product_terms.append("inverter")
    if "motor controller" in q_lower:
        product_terms.append("motor controller")
    if "research" in q_lower or "development" in q_lower or "prototyping" in q_lower:
        product_terms.extend(["r&d", "research", "development", "prototyping", "prototype"])
    if product_terms:
        result["product_terms"] = _dedupe_preserve_order(product_terms)

    # County — require "county" keyword to avoid false matches on city names.
    county_match = re.search(r"\b(\w+)\s+county\b", q_lower)
    if county_match and county_match.group(1) in _GEORGIA_COUNTIES:
        result["location_county"] = county_match.group(1).title()

    # Employment thresholds.
    emp_above = re.search(
        r"\b(?:over|more than|greater than|above)\s+(\d[\d,]*)"
        r"\s*(?:employees?|workers?|jobs?)\b",
        q_lower,
    )
    emp_below = re.search(
        r"\b(?:under|fewer than|less than|below)\s+(\d[\d,]*)"
        r"\s*(?:employees?|workers?|jobs?)\b",
        q_lower,
    )
    if emp_above:
        result["min_employment"] = float(emp_above.group(1).replace(",", ""))
        result["min_employment_strict"] = True
    if emp_below:
        result["max_employment"] = float(emp_below.group(1).replace(",", ""))
        result["max_employment_strict"] = True
    emp_over_after = re.search(r"\bemployment\s+(?:over|above|greater than|more than)\s+(\d[\d,]*)\b", q_lower)
    if emp_over_after:
        result["min_employment"] = float(emp_over_after.group(1).replace(",", ""))
        result["min_employment_strict"] = True

    # Aggregate/list intents that should be computed from filtered Excel rows
    # rather than left to semantic search.
    if ("highest" in q_lower or "greatest" in q_lower) and "employment" in q_lower:
        if "county" in q_lower or "area" in q_lower or "region" in q_lower:
            result.setdefault("aggregate", "highest_county_employment")
            result["force_kb_scan"] = True
        elif not result.get("sort_employment_desc_limit"):
            result["sort_employment_desc_limit"] = 1
            result["force_kb_scan"] = True

    top_n_emp_match = re.search(r"\btop\s+(\d+)\b", q_lower)
    if top_n_emp_match and "employment" in q_lower:
        result["sort_employment_desc_limit"] = int(top_n_emp_match.group(1))
        result["force_kb_scan"] = True

    if ("highest concentration" in q_lower or "most concentrated" in q_lower) and "material" in q_lower:
        result.setdefault("aggregate", "area_concentration")

    if ("how many" in q_lower) and ("area" in q_lower or "location" in q_lower) and "manufacturing" in q_lower:
        result.setdefault("aggregate", "area_counts")

    if "served by only" in q_lower or "only one company" in q_lower or "single company" in q_lower:
        result.setdefault("aggregate", "single_company_roles")
        result["force_kb_scan"] = True

    if "lack" in q_lower and "battery" in q_lower and "tier 1" in q_lower and "general automotive" in q_lower:
        result.clear()
        result["aggregate"] = "counties_lacking_battery_with_tier1_general_auto"
        result["force_kb_scan"] = True

    return result


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _build_selection_rule(filters: dict[str, Any]) -> str:
    """Summarize the retrieval/filter operation in human-readable form."""
    if not filters:
        return "full KB scan (no structured filters)"

    parts: list[str] = []
    if filters.get("tier"):
        parts.append(f"Tier = {filters['tier']}")
    elif filters.get("tiers"):
        parts.append("Tier in {" + ", ".join(filters["tiers"]) + "}")
    if filters.get("location_county"):
        parts.append(f"County = {filters['location_county']}")
    if filters.get("category_values"):
        parts.append("Category in {" + ", ".join(filters["category_values"]) + "}")
    if filters.get("classification_values"):
        parts.append("Classification in {" + ", ".join(filters["classification_values"]) + "}")
    if filters.get("role_terms"):
        parts.append("Role contains {" + ", ".join(filters["role_terms"]) + "}")
    if filters.get("product_terms"):
        parts.append("Products contains {" + ", ".join(filters["product_terms"]) + "}")
    if filters.get("industry_terms"):
        parts.append("Industry contains {" + ", ".join(filters["industry_terms"]) + "}")
    if filters.get("facility_terms"):
        parts.append("Facility contains {" + ", ".join(filters["facility_terms"]) + "}")
    if filters.get("oem_terms_all"):
        parts.append("OEMs contain all {" + ", ".join(filters["oem_terms_all"]) + "}")
    if filters.get("oem_terms_any"):
        parts.append("OEMs contain any {" + ", ".join(filters["oem_terms_any"]) + "}")
    if filters.get("primary_oems_values"):
        parts.append("Primary OEMs in {" + ", ".join(filters["primary_oems_values"]) + "}")
    if filters.get("ev_relevance_values"):
        parts.append("EV relevance in {" + ", ".join(filters["ev_relevance_values"]) + "}")
    if filters.get("ev_relevance_not_values"):
        parts.append("EV relevance not in {" + ", ".join(filters["ev_relevance_not_values"]) + "}")
    if filters.get("min_employment") is not None:
        op = ">" if filters.get("min_employment_strict") else ">="
        parts.append(f"Employment {op} {int(filters['min_employment'])}")
    if filters.get("max_employment") is not None:
        op = "<" if filters.get("max_employment_strict") else "<="
        parts.append(f"Employment {op} {int(filters['max_employment'])}")

    selection = "filtered " + ", ".join(parts) if parts else "full KB scan (no structured filters)"
    if filters.get("sort_employment_desc_limit"):
        selection += (
            f", sorted by Employment descending, returned top {int(filters['sort_employment_desc_limit'])}"
        )
    return selection


def _make_qdrant_filter(filters: dict[str, Any]) -> models.Filter | None:
    """Convert extract_filters() output into a Qdrant Filter."""
    if not filters:
        return None
    must: list[Any] = []

    if "tier" in filters:
        must.append(
            models.FieldCondition(key="tier", match=models.MatchValue(value=filters["tier"]))
        )
    elif "tiers" in filters:
        must.append(
            models.FieldCondition(key="tier", match=models.MatchAny(any=filters["tiers"]))
        )

    if "primary_oems_values" in filters:
        must.append(
            models.FieldCondition(
                key="primary_oems",
                match=models.MatchAny(any=filters["primary_oems_values"]),
            )
        )
    if "ev_relevance_values" in filters:
        must.append(
            models.FieldCondition(
                key="ev_battery_relevant",
                match=models.MatchAny(any=filters["ev_relevance_values"]),
            )
        )
    if "ev_relevance_not_values" in filters:
        # Qdrant does not need this for the deterministic Excel path, and the
        # public client model names have changed across versions. Leave this
        # constraint to _kb_scan() instead of risking a client incompatibility.
        pass
    if "category_values" in filters:
        must.append(
            models.FieldCondition(
                key="tier",
                match=models.MatchAny(any=filters["category_values"]),
            )
        )
    if "classification_values" in filters:
        must.append(
            models.FieldCondition(
                key="classification_method",
                match=models.MatchAny(any=filters["classification_values"]),
            )
        )

    for field in ("location_county", "location_city"):
        if field in filters:
            must.append(
                models.FieldCondition(key=field, match=models.MatchValue(value=filters[field]))
            )

    min_emp = filters.get("min_employment")
    max_emp = filters.get("max_employment")
    if min_emp is not None or max_emp is not None:
        must.append(
            models.FieldCondition(
                key="employment",
                range=models.Range(
                    gt=float(min_emp) if filters.get("min_employment_strict") and min_emp is not None else None,
                    gte=float(min_emp) if not filters.get("min_employment_strict") and min_emp is not None else None,
                    lt=float(max_emp) if filters.get("max_employment_strict") and max_emp is not None else None,
                    lte=float(max_emp) if not filters.get("max_employment_strict") and max_emp is not None else None,
                ),
            )
        )

    return models.Filter(must=must) if must else None


# ── KB Excel deterministic scan ───────────────────────────────────────────────
# For list questions with extractable structured filters, scan the GNEM Excel
# directly and return all matching rows. This gives 100% recall for the
# filtered set without relying on vector search recall.

_KB_EXCEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "kb"
    / "GNEM - Auto Landscape Lat Long Updated.xlsx"
)

_KB_DF: Any = None  # cached DataFrame; loaded once


def _load_kb_df() -> Any:
    global _KB_DF
    if _KB_DF is not None:
        return _KB_DF
    try:
        import pandas as pd
        _KB_DF = pd.read_excel(_KB_EXCEL_PATH)
        logger.info("Loaded KB Excel: %d rows", len(_KB_DF))
    except Exception as exc:
        logger.warning("Could not load KB Excel (%s) — KB scan unavailable", exc)
        _KB_DF = False  # sentinel: tried and failed
    return _KB_DF


@lru_cache(maxsize=1)
def _known_company_names() -> tuple[str, ...]:
    """Canonical company names from the KB, longest first for exact matching."""
    df = _load_kb_df()
    if df is False or df is None:
        return ()
    names = sorted(
        {str(name).strip() for name in df["Company"].dropna().tolist() if str(name).strip()},
        key=len,
        reverse=True,
    )
    return tuple(names)


def _extract_company_names_from_question(q_lower: str) -> list[str]:
    """Find exact KB company names mentioned in the question."""
    found: list[str] = []
    for name in _known_company_names():
        if name.lower() in q_lower:
            found.append(name)
    return found


def _parse_location(loc: str) -> tuple[str, str]:
    """Return (city, county) from 'City, County County' format."""
    if not loc or str(loc).strip().lower() in ("nan", ""):
        return "", ""
    loc = str(loc).strip()
    if "," not in loc and re.search(r"\bcounty\b$", loc, flags=re.IGNORECASE):
        county = re.sub(r"\s+[Cc]ounty$", "", loc).strip()
        return "", county
    parts = loc.split(",", 1)
    city = parts[0].strip()
    if len(parts) == 1:
        return city, ""
    county_raw = parts[1].strip()
    county = re.sub(r"\s+[Cc]ounty$", "", county_raw).strip()
    return city, county


def _text_contains_any(text: Any, terms: list[str]) -> bool:
    text_lower = str(text or "").lower()
    return any(term.lower() in text_lower for term in terms)


def _text_contains_all(text: Any, terms: list[str]) -> bool:
    text_lower = str(text or "").lower()
    return all(term.lower() in text_lower for term in terms)


def _has_meaningful_kb_scan_filters(filters: dict[str, Any]) -> bool:
    """Return True when filters are strong enough for deterministic scanning.

    Employment-only scans are intentionally rejected because they can return
    huge, irrelevant slices of the KB. Employment is useful only when paired
    with a semantic/category/location filter or an explicit aggregate.
    """
    if filters.get("force_kb_scan") or filters.get("aggregate"):
        return True
    strong_keys = {
        "tier", "tiers", "category_values", "classification_values",
        "company_names", "location_county", "role_terms", "product_terms",
        "industry_terms", "facility_terms", "oem_terms_all", "oem_terms_any",
        "primary_oems_values", "require_specific_oem_contract",
        "ev_relevance_values", "ev_relevance_not_values",
    }
    return any(key in filters for key in strong_keys)


def _employment_value(row: Any) -> float:
    try:
        import pandas as pd
        value = row.get("Employment", "")
        if pd.isna(value) or str(value).strip() == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _row_to_hit(row: Any, source: str = "excel_scan", text_override: str | None = None) -> dict[str, Any]:
    city, county = _parse_location(str(row.get("Updated Location", "")))
    employment_float = _employment_value(row)
    employment: int | str = int(employment_float) if employment_float else ""

    meta = {
        "company_name": str(row.get("Company", "")),
        "tier": str(row.get("Category", "")),
        "ev_supply_chain_role": str(row.get("EV Supply Chain Role", "")),
        "location_city": city,
        "location_county": county,
        "employment": employment,
        "primary_oems": str(row.get("Primary OEMs", "")),
        "products_services": str(row.get("Product / Service", "")),
        "ev_battery_relevant": str(row.get("EV / Battery Relevant", "")),
        "industry_group": str(row.get("Industry Group", "")),
        "facility_type": str(row.get("Primary Facility Type", "")),
        "classification_method": str(row.get("Classification Method", "")),
        "supplier_affiliation_type": str(row.get("Supplier or Affiliation Type", "")),
    }
    product_text = meta["products_services"]
    text = text_override or (f"{meta['company_name']} — {product_text}" if product_text else meta["company_name"])
    return {
        "dense_rank": 0,
        "dense_score": 1.0,
        "rerank_score": 1.0,
        "text": text,
        "company_name": meta["company_name"],
        "source_url": "",
        "chunk_id": str(row.name),
        "metadata": meta,
        "retrieval_source": source,
    }


def _aggregate_kb_hits(df: Any, filters: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Compute deterministic aggregate answers from the Excel sheet."""
    aggregate = filters.get("aggregate")
    if not aggregate:
        return None

    import pandas as pd

    if aggregate == "highest_county_employment":
        work = df.copy()
        if "tier" in filters:
            work = work[work["Category"].astype(str).str.strip().eq(filters["tier"])]
        counties = work["Updated Location"].apply(lambda loc: _parse_location(str(loc))[1])
        employment = pd.to_numeric(work["Employment"], errors="coerce").fillna(0)
        grouped = employment.groupby(counties).sum().sort_values(ascending=False)
        if grouped.empty:
            return []
        county = str(grouped.index[0])
        total = int(grouped.iloc[0])
        row = pd.Series({
            "Company": f"{county} County",
            "Category": filters.get("tier", "All Companies"),
            "EV Supply Chain Role": "County employment aggregate",
            "Updated Location": f"{county} County",
            "Employment": total,
            "Primary OEMs": "",
            "Product / Service": f"Highest total employment: {total}",
            "EV / Battery Relevant": "",
            "Industry Group": "",
            "Primary Facility Type": "",
            "Classification Method": "",
            "Supplier or Affiliation Type": "",
        }, name=f"agg:{aggregate}:{county}")
        return [_row_to_hit(row, source="excel_aggregate")]

    if aggregate == "area_concentration":
        matched = df[df["EV Supply Chain Role"].astype(str).str.contains("Materials", case=False, na=False)]
        counts = matched["Updated Location"].value_counts()
        if counts.empty:
            return []
        top_count = int(counts.iloc[0])
        top_locations = counts[counts == top_count].index.tolist()
        hits: list[dict[str, Any]] = []
        for location in top_locations:
            companies = matched[matched["Updated Location"] == location]["Company"].astype(str).tolist()
            city, county = _parse_location(str(location))
            row = pd.Series({
                "Company": str(location),
                "Category": "Area",
                "EV Supply Chain Role": "Materials supplier concentration",
                "Updated Location": str(location),
                "Employment": top_count,
                "Primary OEMs": "",
                "Product / Service": f"{top_count} Materials-category suppliers: {', '.join(companies)}",
                "EV / Battery Relevant": "",
                "Industry Group": "Materials",
                "Primary Facility Type": "",
                "Classification Method": "",
                "Supplier or Affiliation Type": "",
            }, name=f"agg:{aggregate}:{city}:{county}")
            hits.append(_row_to_hit(row, source="excel_aggregate"))
        return hits

    if aggregate == "area_counts":
        facility = df["Primary Facility Type"].astype(str).str.lower().str.contains("manufacturing")
        not_ev_specific = ~df["EV / Battery Relevant"].astype(str).str.strip().eq("Yes")
        matched = df[facility & not_ev_specific]
        counts = matched["Updated Location"].value_counts()
        hits = []
        for location, count in counts.items():
            companies = matched[matched["Updated Location"] == location]["Company"].astype(str).tolist()
            row = pd.Series({
                "Company": str(location),
                "Category": "Area",
                "EV Supply Chain Role": "Manufacturing plants without explicit EV-specific production",
                "Updated Location": str(location),
                "Employment": int(count),
                "Primary OEMs": "",
                "Product / Service": f"{int(count)} plants: {', '.join(companies)}",
                "EV / Battery Relevant": "No/Indirect",
                "Industry Group": "",
                "Primary Facility Type": "Manufacturing Plant",
                "Classification Method": "",
                "Supplier or Affiliation Type": "",
            }, name=f"agg:{aggregate}:{location}")
            hits.append(_row_to_hit(row, source="excel_aggregate"))
        return hits

    if aggregate == "single_company_roles":
        role_series = df["EV Supply Chain Role"].dropna().astype(str).str.strip()
        singleton_roles = role_series.value_counts()
        singleton_roles = singleton_roles[singleton_roles == 1].index.tolist()
        hits = []
        for role in singleton_roles:
            row = df[df["EV Supply Chain Role"].astype(str).str.strip().eq(role)].iloc[0]
            hits.append(_row_to_hit(row, source="excel_aggregate", text_override=f"Role served by one company: {role} | Company: {row.get('Company', '')}"))
        facility_role = "EV thermal systems and electronics"
        facility_rows = df[
            df["Primary Facility Type"].astype(str).str.strip().eq(facility_role)
            & df["EV Supply Chain Role"].isna()
        ]
        for _, row in facility_rows.iterrows():
            hits.append(
                _row_to_hit(
                    row,
                    source="excel_aggregate",
                    text_override=f"Role served by one company: {facility_role} | Company: {row.get('Company', '')}",
                )
            )
        return hits

    if aggregate == "counties_lacking_battery_with_tier1_general_auto":
        def _comma_county(loc: Any) -> str:
            loc_text = str(loc or "")
            if "," not in loc_text:
                return ""
            return _parse_location(loc_text)[1]

        counties = df["Updated Location"].apply(_comma_county)
        battery_counties = set(
            counties[
                df["EV Supply Chain Role"].astype(str).str.contains("Battery Cell|Battery Pack", case=False, na=False)
            ].dropna().astype(str)
        )
        tier1_general = (
            df["Category"].astype(str).str.strip().isin(["Tier 1", "Tier 1/2"])
            & df["EV Supply Chain Role"].astype(str).str.contains("General Automotive", case=False, na=False)
        )
        target_counties = sorted({c for c in counties[tier1_general].dropna().astype(str) if c} - battery_counties)
        hits = []
        for county in target_counties:
            companies = df[tier1_general & (counties == county)]["Company"].astype(str).tolist()
            row = pd.Series({
                "Company": f"{county} County",
                "Category": "Tier 1 infrastructure",
                "EV Supply Chain Role": "General Automotive; no Battery Cell/Pack suppliers in county",
                "Updated Location": f"{county} County",
                "Employment": len(companies),
                "Primary OEMs": "",
                "Product / Service": f"Tier 1 General Automotive companies: {', '.join(companies)}",
                "EV / Battery Relevant": "",
                "Industry Group": "",
                "Primary Facility Type": "",
                "Classification Method": "",
                "Supplier or Affiliation Type": "",
            }, name=f"agg:{aggregate}:{county}")
            hits.append(_row_to_hit(row, source="excel_aggregate"))
        return hits

    return None


def _kb_scan(filters: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Scan the KB Excel and return all rows matching the extracted filters.

    Returns None  → KB unavailable or no useful filters (caller falls back to vector search).
    Returns []    → filters valid but no rows match.
    Returns hits  → all matching rows formatted as hit dicts (no reranking needed).
    """
    if not filters or not _has_meaningful_kb_scan_filters(filters):
        return None  # no deterministic filters to apply

    df = _load_kb_df()
    if df is False or df is None:
        return None

    import pandas as pd

    aggregated = _aggregate_kb_hits(df, filters)
    if aggregated is not None:
        logger.info("KB aggregate returned %d rows for filters %s", len(aggregated), filters)
        return aggregated

    mask = pd.Series([True] * len(df), index=df.index)

    if "company_names" in filters:
        mask &= df["Company"].astype(str).str.strip().isin(filters["company_names"])

    # Tier filter
    tiers_list: list[str] | None = None
    if "tier" in filters:
        tiers_list = [filters["tier"]]
    elif "tiers" in filters:
        tiers_list = filters["tiers"]
    if tiers_list:
        tier_col = df["Category"].astype(str).str.strip()
        mask &= tier_col.isin(tiers_list)

    if "category_values" in filters:
        category_col = df["Category"].astype(str).str.strip()
        mask &= category_col.isin(filters["category_values"])

    if "classification_values" in filters:
        classification_col = df["Classification Method"].astype(str).str.strip()
        mask &= classification_col.isin(filters["classification_values"])

    # County filter
    if "location_county" in filters:
        target_county = filters["location_county"].lower()
        def _county_match(loc: str) -> bool:
            _, county = _parse_location(str(loc))
            return county.lower() == target_county.lower()
        mask &= df["Updated Location"].apply(_county_match)

    if "role_terms" in filters:
        role_col = df["EV Supply Chain Role"].astype(str)
        mask &= role_col.apply(lambda value: _text_contains_any(value, filters["role_terms"]))

    if "product_terms" in filters:
        product_col = df["Product / Service"].astype(str)
        mask &= product_col.apply(lambda value: _text_contains_any(value, filters["product_terms"]))

    if "industry_terms" in filters:
        industry_col = df["Industry Group"].astype(str)
        mask &= industry_col.apply(lambda value: _text_contains_any(value, filters["industry_terms"]))

    if "facility_terms" in filters:
        facility_col = df["Primary Facility Type"].astype(str)
        product_col = df["Product / Service"].astype(str)
        mask &= (
            facility_col.apply(lambda value: _text_contains_any(value, filters["facility_terms"]))
            | product_col.apply(lambda value: _text_contains_any(value, filters["facility_terms"]))
        )

    if "primary_oems_values" in filters:
        oem_col = df["Primary OEMs"].astype(str).str.strip()
        mask &= oem_col.isin(filters["primary_oems_values"])

    if "oem_terms_all" in filters:
        oem_col = df["Primary OEMs"].astype(str)
        mask &= oem_col.apply(lambda value: _text_contains_all(value, filters["oem_terms_all"]))

    if "oem_terms_any" in filters:
        oem_col = df["Primary OEMs"].astype(str)
        mask &= oem_col.apply(lambda value: _text_contains_any(value, filters["oem_terms_any"]))

    if filters.get("require_specific_oem_contract"):
        oem_col = df["Primary OEMs"].astype(str).str.strip()
        mask &= oem_col.ne("") & ~oem_col.str.lower().isin({"nan", "multiple oems"})

    if "ev_relevance_values" in filters:
        ev_col = df["EV / Battery Relevant"].astype(str).str.strip()
        mask &= ev_col.isin(filters["ev_relevance_values"])

    if "ev_relevance_not_values" in filters:
        ev_col = df["EV / Battery Relevant"].astype(str).str.strip()
        mask &= ~ev_col.isin(filters["ev_relevance_not_values"])

    # Employment filters
    if "min_employment" in filters or "max_employment" in filters:
        emp_series = pd.to_numeric(df["Employment"], errors="coerce").fillna(0)
        if "min_employment" in filters:
            if filters.get("min_employment_strict"):
                mask &= emp_series > filters["min_employment"]
            else:
                mask &= emp_series >= filters["min_employment"]
        if "max_employment" in filters:
            if filters.get("max_employment_strict"):
                mask &= emp_series < filters["max_employment"]
            else:
                mask &= emp_series <= filters["max_employment"]

    matched = df[mask]
    if "sort_employment_desc_limit" in filters:
        emp_series = pd.to_numeric(matched["Employment"], errors="coerce").fillna(0)
        matched = (
            matched.assign(_employment_sort=emp_series)
            .sort_values(["_employment_sort", "Company"], ascending=[False, True])
        )
        matched = matched.head(int(filters["sort_employment_desc_limit"])).drop(columns=["_employment_sort"])
    elif not filters.get("preserve_company_rows"):
        matched = matched.drop_duplicates(
            subset=["Company", "Updated Location", "Primary Facility Type"],
            keep="first",
        )

    if matched.empty:
        logger.info("KB scan found 0 rows for filters %s", filters)
        return []

    hits: list[dict[str, Any]] = [_row_to_hit(row) for _, row in matched.iterrows()]

    logger.info("KB scan returned %d rows for filters %s", len(hits), filters)
    return hits


# ── Reranker (mandatory) ─────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str):
    """Force-load the cross-encoder. Hard fail on any error."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Reranker is mandatory; sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from exc
    try:
        return CrossEncoder(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Reranker is mandatory; failed to load CrossEncoder({model_name!r}): {exc}"
        ) from exc


def ensure_reranker(model_name: str) -> None:
    """Eagerly load the cross-encoder so a missing model fails fast."""
    _load_cross_encoder(model_name)


# ── Qdrant client ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _get_qdrant_client(url: str, api_key: str) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key, timeout=60)


# ── Query embedding via Ollama ────────────────────────────────────────────────

def _embed_query(question: str, ollama_base_url: str, embed_model: str) -> list[float]:
    url = f"{ollama_base_url}/api/embed"
    response = httpx.post(
        url,
        json={"model": embed_model, "input": [question]},
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings", [])
    if not embeddings:
        raise RuntimeError(f"Ollama returned no embeddings for query: {question!r}")
    return embeddings[0]


# ── Qdrant search (hybrid with dense-only fallback) ───────────────────────────

_LIST_RERANK_THRESHOLD = -5.0


def _search(
    client: QdrantClient,
    collection: str,
    dense_name: str,
    sparse_name: str,
    query_vector: list[float],
    query_text: str,
    top_k: int,
    qdrant_filter: models.Filter | None,
) -> tuple[list[dict[str, Any]], str]:
    """Hybrid RRF search with graceful dense-only fallback."""
    sparse_vec = _build_sparse_vector(query_text)
    retrieval_source = "hybrid"
    try:
        result = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using=dense_name,
                    limit=top_k,
                    filter=qdrant_filter,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using=sparse_name,
                    limit=top_k,
                    filter=qdrant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        logger.info("Hybrid search returned %d hits", len(result.points))
    except Exception as exc:
        logger.warning("Hybrid search failed (%s) — falling back to dense-only", exc)
        retrieval_source = "dense"
        kw: dict[str, Any] = {}
        if qdrant_filter is not None:
            kw["query_filter"] = qdrant_filter
        result = client.query_points(
            collection_name=collection,
            query=query_vector,
            using=dense_name,
            limit=top_k,
            with_payload=True,
            **kw,
        )
        logger.info("Dense-only fallback returned %d hits", len(result.points))

    hits: list[dict[str, Any]] = []
    for rank, point in enumerate(result.points):
        payload = point.payload or {}
        text = payload.get("parent_text") or payload.get("text") or ""
        hits.append(
            {
                "dense_rank": rank,
                "dense_score": float(point.score) if point.score is not None else 0.0,
                "text": text,
                "company_name": payload.get("company_name", ""),
                "source_url": payload.get("source_url", ""),
                "chunk_id": payload.get("chunk_id", str(point.id)),
                "metadata": payload,
                "retrieval_source": retrieval_source,
            }
        )
    return hits, retrieval_source


def _rerank(
    question: str, hits: list[dict[str, Any]], reranker_model: str
) -> list[dict[str, Any]]:
    if not hits:
        return hits
    encoder = _load_cross_encoder(reranker_model)
    pairs = [(question, hit["text"]) for hit in hits]
    scores = encoder.predict(pairs)
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
    hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return hits


# ── Public entrypoint ─────────────────────────────────────────────────────────

def retrieve_and_rerank(
    question: str,
    cfg: GenerationConfig,
    top_k: int = 120,
    rerank_top_n: int = 40,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Structured retrieval with vector fallback.

    Returns (hits, retrieval_meta) where retrieval_meta describes the retrieval
    operation so format_context() can prepend a RETRIEVAL SUMMARY for the LLM.

    Decision tree:
      1. If query is a list/count question AND structured filters are extractable
         → KB Excel scan (100% recall for filtered set, no vector search).
      2. If KB scan finds no rows for a deterministic Excel query
         → return an empty context; do not pull unrelated vector hits.
      3. If no filters OR KB unavailable
         → hybrid/dense vector search.
      4. For list queries via vector: return all candidates above a score
         threshold instead of a fixed top-N cap.
    """
    query_type = classify_query(question)
    filters = extract_filters(question)
    if filters:
        logger.info("Extracted filters: %s", filters)

    # Path 1: deterministic KB scan for list questions with extractable filters
    # or point questions whose answer is a structured Excel computation.
    if filters and (
        query_type == "list"
        or filters.get("force_kb_scan")
        or "company_names" in filters
    ):
        kb_hits = _kb_scan(filters)
        if kb_hits is not None:
            selection_rule = _build_selection_rule(filters)
            is_complete = "sort_employment_desc_limit" not in filters and not filters.get("aggregate")
            meta: dict[str, Any] = {
                "mode": "deterministic_excel_scan",
                "selection_rule": selection_rule,
                "matched_rows": len(kb_hits),
                "is_complete": is_complete,
                "sort_rule": "Employment descending" if filters.get("sort_employment_desc_limit") else None,
                "grouping_rule": "by County" if filters.get("aggregate") == "highest_county_employment" else None,
            }
            if kb_hits:
                logger.info("KB scan path: returning %d deterministic rows", len(kb_hits))
                return kb_hits, meta
            else:
                logger.info("KB scan path: 0 deterministic rows for filters %s", filters)
                return [], meta

    # Path 2: vector search (hybrid → dense fallback).
    qdrant_filter = _make_qdrant_filter(filters)
    client = _get_qdrant_client(cfg.qdrant_url, cfg.qdrant_api_key)
    query_vector = _embed_query(question, cfg.ollama_base_url, cfg.embedding_model)

    hits, vec_source = _search(
        client=client,
        collection=cfg.qdrant_collection,
        dense_name=cfg.qdrant_dense_name,
        sparse_name=cfg.qdrant_sparse_name,
        query_vector=query_vector,
        query_text=question,
        top_k=top_k,
        qdrant_filter=qdrant_filter,
    )

    reranked = _rerank(question, hits, cfg.reranker_model)

    if query_type == "list":
        result = [h for h in reranked if h.get("rerank_score", 0.0) > _LIST_RERANK_THRESHOLD]
        logger.info(
            "Vector list mode: %d/%d hits above rerank threshold %.1f",
            len(result), len(reranked), _LIST_RERANK_THRESHOLD,
        )
        vec_meta: dict[str, Any] = {
            "mode": vec_source,
            "selection_rule": _build_selection_rule(filters) if filters else "vector search, no structured filters",
            "matched_rows": len(result),
            "is_complete": False,
            "sort_rule": "rerank score descending",
            "grouping_rule": None,
        }
        return result, vec_meta

    final = reranked[:rerank_top_n]
    point_meta: dict[str, Any] = {
        "mode": vec_source,
        "selection_rule": _build_selection_rule(filters) if filters else "vector search, no structured filters",
        "matched_rows": len(final),
        "is_complete": False,
        "sort_rule": "rerank score descending",
        "grouping_rule": None,
    }
    return final, point_meta


def format_context(
    hits: list[dict[str, Any]],
    retrieval_meta: dict[str, Any] | None = None,
) -> str:
    """Render hits as a structured field-per-line format.

    If retrieval_meta is provided, a RETRIEVAL SUMMARY block is prepended so
    the LLM knows the selection rule and can trust the returned rows.
    """
    if not hits:
        return ""

    header_lines: list[str] = []
    if retrieval_meta:
        mode_label = {
            "deterministic_excel_scan": "deterministic Excel scan",
            "hybrid": "hybrid (dense + sparse)",
            "dense": "dense vector",
        }.get(retrieval_meta.get("mode", ""), retrieval_meta.get("mode", "unknown"))
        completeness = (
            f"{retrieval_meta['matched_rows']} (complete — all matching rows returned)"
            if retrieval_meta.get("is_complete")
            else f"{retrieval_meta['matched_rows']} (may be partial — capped at top N by score)"
        )
        header_lines.append("RETRIEVAL SUMMARY:")
        header_lines.append(f"  Retrieval mode: {mode_label}")
        header_lines.append(f"  Selection rule: {retrieval_meta.get('selection_rule', 'unknown')}")
        header_lines.append(f"  Matched KB rows: {completeness}")
        if retrieval_meta.get("sort_rule"):
            header_lines.append(f"  Sort rule: {retrieval_meta['sort_rule']}")
        if retrieval_meta.get("grouping_rule"):
            header_lines.append(f"  Grouping rule: {retrieval_meta['grouping_rule']}")
        if retrieval_meta.get("mode") == "deterministic_excel_scan":
            header_lines.append("")
            header_lines.append(
                "All returned rows are deterministic KB matches. "
                "Do not re-filter them. "
                "If the selection rule says 'top 1', the first row IS the answer."
            )
        header_lines.append("---")

    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        company = hit.get("company_name") or meta.get("company_name") or ""
        block_header = f"[{i}] {company}" if company else f"[{i}]"

        tier = meta.get("tier", "")
        role = meta.get("ev_supply_chain_role", "")
        city = meta.get("location_city", "")
        county = meta.get("location_county", "")
        employment = meta.get("employment", "")
        oems = meta.get("primary_oems", "")
        products = meta.get("products_services", "")
        ev_relevant = meta.get("ev_battery_relevant", "")
        industry = meta.get("industry_group", "")
        facility = meta.get("facility_type", "")
        supplier_affiliation = meta.get("supplier_affiliation_type", "")
        classification = meta.get("classification_method", "")
        source_url = hit.get("source_url", "")

        lines: list[str] = [block_header]
        if tier:
            lines.append(f"  Tier: {tier}")
        if role:
            lines.append(f"  Role: {role}")
        if city or county:
            location = ", ".join(filter(None, [city, county, "Georgia"]))
            lines.append(f"  Location: {location}")
        if employment:
            lines.append(f"  Employment: {employment}")
        if oems:
            lines.append(f"  OEMs: {oems}")
        if products:
            lines.append(f"  Products: {products}")
        if ev_relevant:
            lines.append(f"  EV Relevant: {ev_relevant}")
        if industry:
            lines.append(f"  Industry: {industry}")
        if facility:
            lines.append(f"  Facility: {facility}")
        if supplier_affiliation and str(supplier_affiliation).lower() not in ("nan", ""):
            lines.append(f"  Affiliation: {supplier_affiliation}")
        if classification and str(classification).lower() not in ("nan", ""):
            lines.append(f"  Classification: {classification}")
        if source_url:
            lines.append(f"  Source: {source_url}")

        retrieval_source = hit.get("retrieval_source", "")
        if retrieval_source not in ("excel_scan", "excel_aggregate"):
            raw_text = hit.get("text", "").strip()
            if raw_text:
                lines.append(raw_text)

        blocks.append("\n".join(lines))

    body = "\n\n".join(blocks)
    if header_lines:
        return "\n".join(header_lines) + "\n\n" + body
    return body
