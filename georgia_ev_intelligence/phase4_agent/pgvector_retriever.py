"""
phase4_agent/pgvector_retriever.py
==============================================================
Pgvector Hybrid Search Retriever

Replaces strict SQL text-filtering with semantic vector similarity,
dramatically improving Context Recall for EV Supply Chain questions.
"""
from __future__ import annotations
import json
import urllib.request
from sqlalchemy import text
from shared.db import get_session
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase4.pgvector_retriever")

def get_embedding(prompt: str) -> list[float] | None:
    """Get the 768-dimension vector from Ollama using nomic-embed-text."""
    cfg = Config.get()
    url = f"{cfg.ollama_base_url}/api/embeddings"
    data = json.dumps({"model": "nomic-embed-text", "prompt": prompt}).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode()).get("embedding")
    except Exception as e:
        logger.error(f"Failed to get embedding from Ollama: {e}")
        return None

def hybrid_search(question: str) -> list[dict]:
    """
    Perform a hybrid semantic search using pgvector.
    This fetches the most semantically relevant companies to the user's question,
    eliminating the need for brittle regex or exact-text SQL WHERE clauses.
    """
    logger.info(f"Generating vector embedding for question: {question}")
    vec = get_embedding(question)
    if not vec:
        logger.warning("Falling back to empty result due to embedding failure.")
        return []

    # Calculate L2 distance (<->) or Cosine distance (<=>)
    # We'll use Cosine distance (<=>) for semantic similarity
    query = text("""
        SELECT 
            company_name, 
            tier, 
            ev_supply_chain_role, 
            location_county, 
            employment, 
            industry_group, 
            ev_battery_relevant, 
            facility_type, 
            products_services,
            1 - (embedding <=> CAST(:vec AS vector)) AS similarity_score
        FROM gev_companies 
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:vec AS vector);
    """)

    session = get_session()
    results = []
    try:
        rows = session.execute(query, {"vec": str(vec)})
        for row in rows:
            # Only include highly relevant results (e.g. score > 0.25)
            # You can tweak this threshold to balance Precision and Recall
            if float(row.similarity_score) > 0.25:
                results.append({
                    "company_name": row.company_name or "",
                    "tier": row.tier or "",
                    "ev_supply_chain_role": row.ev_supply_chain_role or "",
                    "location_county": row.location_county or "",
                    "employment": row.employment or "",
                    "industry_group": row.industry_group or "",
                    "ev_battery_relevant": row.ev_battery_relevant or "",
                    "facility_type": row.facility_type or "",
                    "products_services": row.products_services or "",
                    "similarity_score": row.similarity_score,
                })
        logger.info(f"Hybrid search returned {len(results)} highly relevant companies.")
        return results
    finally:
        session.close()
