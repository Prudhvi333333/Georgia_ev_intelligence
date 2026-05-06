"""
Phase 4 — Agent Pipeline (V3 Architecture)

  Step 1: EXTRACT  — Deterministic entity extraction (no LLM)
  Step 2: RETRIEVE — Text-to-SQL (LLM generates SQL) + Deterministic Cypher + Gemma fallback
  Step 3: GENERATE — Single LLM synthesis call (qwen2.5:7b)

WHY TEXT-TO-SQL REPLACES HARDCODED RULES:
  Rules break for edge cases:
    "Tier 1 only"       → needs exact match, not ilike
    "at least 500 emp"  → needs >= filter
    "top 5 by county"   → needs LIMIT 5
    "excluding OEMs"    → needs NOT LIKE filter

  The LLM reads the question + schema → generates correct SQL for ANY phrasing.
  No code changes needed when new question patterns appear.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any

from phase4_agent.entity_extractor import extract, Entities
from phase4_agent.sql_retriever import (
    query_companies,
    full_text_search,
    get_single_supplier_roles,
    aggregate_employment_by_county,
    top_companies_by_employment,
)
from phase4_agent.text_to_sql import text_to_sql
from phase4_agent.text_to_cypher import (
    execute_cypher_safe,
    execute_cypher,
    normalize_cypher_results,
)
from phase4_agent.cypher_builder import build_cypher
from phase4_agent.streaming import stream_answer_collected
# from phase4_agent.formatters import ...   ← available for future use, not active
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase4.agent")

_MAX_LLM_COMPANIES = 15   # capped from 50 — prevents LLM reading overload that causes 'not found' hallucination


def _is_oem_reference(tier_value: str | None) -> bool:
    """True when 'OEM' was extracted as a bare acronym, not as a real tier label."""
    if not tier_value:
        return False
    if tier_value.lower() in ("oem supply chain", "oem footprint", "oem (footprint)"):
        return False
    return tier_value.strip().upper() == "OEM"


def _parse_aggregate_context(context: str) -> list[dict]:
    """
    Parse formatted aggregate context back into row dicts.
    Context lines look like: "Troup County: 2,280 employees (7 companies)"
    """
    import re
    rows = []
    for line in context.splitlines():
        m = re.match(r"\s*(.+?):\s+([\d,]+)\s+employees\s+\((\d+)\s+companies?\)", line)
        if m:
            rows.append({
                "county":           m.group(1).strip(),
                "total_employment": int(m.group(2).replace(",", "")),
                "company_count":    int(m.group(3)),
            })
    return rows


def _parse_company_context(context: str) -> list[dict]:
    """
    Parse pipe-separated company table rows from formatted context.
    Dynamically reads the header row so it doesn't break when new columns are added.
    """
    companies = []
    lines = context.splitlines()
    in_table = False
    headers = []
    
    for line in lines:
        if " | " in line and not in_table:
            # We found the header row
            in_table = True
            headers = [h.strip() for h in line.split("|")]
            continue
            
        if in_table and line.startswith("---"):
            continue
            
        if in_table and "|" in line and not line.startswith("Total:"):
            parts = [p.strip() for p in line.split("|")]
            row_dict = {}
            for i, header in enumerate(headers):
                if i < len(parts):
                    # Handle employment empty string conversion to None
                    val = parts[i]
                    if header.lower() == "employment" and not val:
                        row_dict[header] = None
                    else:
                        row_dict[header] = val
                else:
                    row_dict[header] = ""
            companies.append(row_dict)
            
    return companies


def _sql_row_to_company(row: dict) -> dict:
    """
    Map a raw SQL result row to a company dict for _format_companies.

    WHY SCHEMA-AGNOSTIC (not a hardcoded column whitelist):
      If a new column is added to gev_companies (e.g. latitude, naics_code,
      supplier_tier_2_of, etc.) it is automatically passed through here.
      _format_companies reads specific named keys and ignores extras, so
      new columns are available in context without any code change needed.

      Rule: pass ALL row keys through, coercing None -> empty string.
    """
    return {k: (v if v is not None else "") for k, v in row.items()}


class EVAgent:
    """Georgia EV Supply Chain Intelligence Agent — V3 Architecture."""

    def __init__(self, model_override: str | None = None) -> None:
        cfg = Config.get()
        # model_override lets the evaluator pass 'gemma2:9b', 'qwen2.5:14b' etc.
        # directly without modifying env vars (Config is @lru_cache — env changes are ignored).
        self.llm_model = model_override or cfg.ollama_llm_model
        logger.info("EVAgent initialized | model=%s", self.llm_model)


    # ── Step 3: Generate ──────────────────────────────────────────────────────

    def _generate(self, question: str, context: str) -> str:
        """Synthesize answer — uses self.llm_model (supports model_override for eval)."""
        try:
            return stream_answer_collected(question, context, model=self.llm_model)
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            return f"[LLM unavailable] Retrieved data: {context[:500]}"


    # ── Step 2: Retrieve ──────────────────────────────────────────────────────

    def _retrieve(self, question: str, e: Entities) -> tuple[str, bool]:
        """
        V4 Architecture — priority order:

        A. Text-to-SQL     → LLM generates full SQL for aggregate/county/filter Qs
        B. PgVector Hybrid → Semantic vector similarity search (handles everything else)

        Returns (context_string, cypher_was_used)
        """
        # ── A: Text-to-SQL (Aggregate / Filtering) ────────────────────────────────
        aggregate_signals = ["how many", "count", "top", "highest", "most", "total employment", "county with"]
        is_aggregate = any(sig in question.lower() for sig in aggregate_signals)

        if is_aggregate:
            logger.info("Routing to Text-to-SQL (aggregate question)")
            from phase4_agent.text_to_sql import generate_sql
            sql_query = generate_sql(question)
            if sql_query and sql_query.upper().startswith("SELECT"):
                try:
                    from shared.db import get_session
                    from sqlalchemy import text
                    session = get_session()
                    rows = session.execute(text(sql_query)).fetchall()
                    session.close()
                    
                    if rows:
                        header = list(rows[0]._mapping.keys())
                        lines = [" | ".join(header)]
                        lines.append("-" * 60)
                        for r in rows:
                            lines.append(" | ".join(str(val) if val is not None else "" for val in r))
                        
                        ctx = f"[Text-to-SQL Result]:\\n" + "\\n".join(lines)
                        return ctx, False
                except Exception as exc:
                    logger.warning("Text-to-SQL execution failed: %s", exc)

        # ── B: PgVector Hybrid Search (Semantic Retrieval) ───────────────────────
        logger.info("Routing to PgVector Hybrid Search")
        from phase4_agent.pgvector_retriever import hybrid_search
        hybrid_results = hybrid_search(question)
        if hybrid_results:
            ctx = f"[PgVector Hybrid Search ({len(hybrid_results)} relevant companies)]:\n" + self._format_companies(hybrid_results)
            return ctx, False

        # ── C: Nothing found ───────────────────────────────────────────────────
        return (
            "No relevant data found. The database contains 193 Georgia EV supply chain "
            "companies with tier, role, employment, county, OEM, industry, and facility type data."
        ), False

    # ── Formatting helper ─────────────────────────────────────────────────────

    @staticmethod
    def _format_companies(companies: list[dict]) -> str:
        """Compact pipe-separated table. Dynamically uses keys from the dictionaries."""
        if not companies:
            return "No matching companies found."

        # Dynamically extract headers from the first company dictionary
        keys = list(companies[0].keys())
        header = " | ".join(keys)
        divider = "-" * len(header)
        rows = [f"Total: {len(companies)} companies\n", header, divider]

        for c in companies:
            row_parts = []
            for key in keys:
                val = c.get(key)
                if val is None:
                    row_parts.append("")
                elif isinstance(val, (int, float)):
                    # Format numbers cleanly
                    row_parts.append(str(int(val)) if val == int(val) else str(val))
                else:
                    # Truncate extremely long string fields to prevent LLM overload
                    val_str = str(val).replace('\n', ' ')
                    row_parts.append(val_str[:60] if len(val_str) > 60 else val_str)
            rows.append(" | ".join(row_parts))
            
        return "\n".join(rows)

    # ── Main entry point ──────────────────────────────────────────────────────

    def ask(self, question: str) -> dict[str, Any]:
        """
        Answer a question about the Georgia EV supply chain.

        Architecture:
          Retrieval : 0-1 LLM calls (deterministic SQL/Cypher or Text-to-SQL generation)
          Synthesis : 1 LLM call always (true RAG — LLM reads data and writes answer)

        With streaming enabled, complex 2-call queries (~25s total) feel instant
        because the first token arrives in <2s and the user reads as it generates.

        LLM call count:
          Risk query             → 0 retrieval + 1 synthesis = 1 call  (~10s)
          Deterministic SQL/Neo4j→ 0 retrieval + 1 synthesis = 1 call  (~10s)
          Text-to-SQL (complex)  → 1 SQL gen   + 1 synthesis = 2 calls (~25s, streamed)
        """
        start = time.monotonic()
        logger.info("Question: %s", question[:100])

        # Step 1: Extract entities
        entities = extract(question)
        logger.info(
            "Extracted: tier=%s county=%s oem=%s industry_group=%s role=%s | "
            "aggregate=%s risk=%s oem_dep=%s capacity=%s misalign=%s top_n=%s",
            entities.tier, entities.county, entities.oem, entities.industry_group,
            entities.ev_role or entities.ev_role_list,
            entities.is_aggregate, entities.is_risk_query, entities.is_oem_dependency,
            entities.is_capacity_risk, entities.is_misalignment, entities.is_top_n,
        )

        # Step 2: Retrieve
        context, cypher_used = self._retrieve(question, entities)
        # Count pipe-separated company rows (more accurate than line count)
        retrieved_count = sum(1 for line in context.splitlines() if " | " in line and not line.startswith("Company"))

        # Step 3: Synthesize — LLM reads data and writes the answer (RAG)
        # Exception: SPOF risk query is already a formatted answer (direct list)
        if context.startswith("__DIRECT_ANSWER__:"):
            answer = context[len("__DIRECT_ANSWER__:"):]
        else:
            answer = self._generate(question, context)

        elapsed = time.monotonic() - start
        logger.info("Answered in %.1fs | rows=%d | cypher=%s", elapsed, retrieved_count, cypher_used)

        # Serialize ALL entity fields automatically using dataclasses.asdict().
        # WHY: a hardcoded dict silently drops new fields added to Entities.
        # dataclasses.asdict() always includes every field — zero maintenance.
        entity_dict = dataclasses.asdict(entities)
        entity_dict["cypher_used"] = cypher_used   # pipeline-level flag, not in dataclass

        return {
            "question":          question,
            "answer":            answer,
            "retrieved_context": context,
            "entities":          entity_dict,
            "retrieved_count":   retrieved_count,
            "elapsed_s":         round(elapsed, 1),
        }

