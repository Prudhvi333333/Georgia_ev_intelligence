"""Mode-specific prompt builders for the four LLM-comparison modes.

Each mode states its name, allowed sources, forbidden sources, the inline
tagging rules, and an output contract. Prompts are deliberately verbose so
the four modes produce measurably different behavior and so the answers can
be audited row-by-row.

Returned prompt is a single string suitable for Ollama's /api/generate
endpoint (which does not accept separate system/user roles). The "system"
section is rendered as a prelude inside the same string.
"""
from __future__ import annotations

VALID_MODES = ("rag_only", "no_rag", "rag_pretrained", "rag_pretrained_web")

INSUFFICIENT_CONTEXT_SENTENCE = (
    "The retrieved context does not contain enough information to answer this question."
)

# Field-focus and exhaustive-list instruction injected into all RAG prompts.
_FIELD_FOCUS = (
    "Field-focus rules:\n"
    "  * Read the question carefully to identify the SPECIFIC attribute requested\n"
    "    (e.g., employment count, tier, location, OEM relationships).\n"
    "  * Answer ONLY that attribute. Do not include other company fields in the answer.\n"
    "  * If the question asks to list ALL matching companies, include EVERY company\n"
    "    from INTERNAL_CONTEXT that satisfies the condition. Do not stop after the first few.\n"
    "  * Copy company names exactly as they appear in INTERNAL_CONTEXT — character for character.\n"
)

# Excel-grounded guardrail injected into modes 3 and 4 only.
# Prevents pretrained/web facts from overwriting or diluting Excel-specific values.
_EXCEL_GUARDRAIL = (
    "  * INTERNAL_CONTEXT is the authoritative source for company names, counts,\n"
    "    roles, tiers, locations, employment, and products. Use pretrained or web\n"
    "    facts ONLY for clearly labelled background that does not change the\n"
    "    Excel-grounded answer.\n"
)

# Retrieval-trust instruction injected into all RAG prompts.
# Tells the LLM to trust the RETRIEVAL SUMMARY prepended by format_context().
_RETRIEVAL_TRUST = (
    "Retrieval trust rules:\n"
    "  * INTERNAL_CONTEXT begins with a RETRIEVAL SUMMARY. Trust every claim in it.\n"
    "  * If RETRIEVAL SUMMARY states a selection rule (e.g., 'sorted by Employment\n"
    "    descending, returned top 1'), the listed company IS the answer. Do not\n"
    "    second-guess, re-rank, or refuse to name it.\n"
    "  * If RETRIEVAL SUMMARY says 'Matched KB rows: N', include all N companies\n"
    "    or locations in your answer. Do not collapse or merge rows.\n"
    "  * If a row is a deterministic KB match, do not re-filter it based on your\n"
    "    own judgment. The selection rule already explains why it was included.\n"
    "  * Do not refuse to answer when a deterministic selection rule is present\n"
    "    and the answer row is in context.\n"
)

# Incomplete-context warning injected into rag_only for exhaustive questions.
_INCOMPLETE_CONTEXT = (
    "  * If the question asks to list ALL matching companies but INTERNAL_CONTEXT\n"
    "    appears to be an incomplete subset, begin your answer with:\n"
    "    \"Note: retrieved context may be incomplete.\"\n"
    "    Then list every company present in INTERNAL_CONTEXT that matches.\n"
)


def _system(mode: str, allowed: list[str], forbidden: list[str], tagging: list[str]) -> str:
    allowed_block = "\n".join(f"  * {item}" for item in allowed)
    forbidden_block = "\n".join(f"  * {item}" for item in forbidden) if forbidden else "  * (none)"
    tagging_block = "\n".join(f"  * {item}" for item in tagging) if tagging else "  * (no tagging required)"
    return (
        f"You are answering questions about Georgia electric vehicle (EV) infrastructure.\n"
        f"Mode: {mode}.\n\n"
        f"Allowed sources:\n{allowed_block}\n\n"
        f"Forbidden sources:\n{forbidden_block}\n\n"
        f"Tagging rules (apply inline, exactly as written):\n{tagging_block}\n\n"
        f"Output contract:\n"
        f"  * Return only the final answer text.\n"
        f"  * No preamble. No chain-of-thought. No source list.\n"
        f"  * Do not restate the question.\n"
    )


def _build_rag_only(question: str, internal_context: str) -> str:
    system = _system(
        mode="rag_only",
        allowed=["INTERNAL_CONTEXT (retrieved from the project knowledge base)"],
        forbidden=[
            "Your pretrained / general knowledge",
            "Web search results",
        ],
        tagging=[],
    )
    extra = (
        f"{_RETRIEVAL_TRUST}\n"
        "Strict instruction: Answer ONLY using INTERNAL_CONTEXT. Do not use any pretrained\n"
        "knowledge. Do not use any web data. If INTERNAL_CONTEXT does not contain enough\n"
        "information to answer the question, your entire answer must be exactly:\n"
        f"\"{INSUFFICIENT_CONTEXT_SENTENCE}\"\n\n"
        f"{_INCOMPLETE_CONTEXT}\n"
        f"{_FIELD_FOCUS}"
    )
    user = (
        f"Question: {question}\n\n"
        f"INTERNAL_CONTEXT:\n{internal_context or '(empty)'}\n"
    )
    return f"{system}\n{extra}\n{user}\nAnswer:"


def _build_no_rag(question: str) -> str:
    system = _system(
        mode="no_rag",
        allowed=["Your pretrained / general knowledge only"],
        forbidden=[
            "INTERNAL_CONTEXT (none is provided)",
            "Web search results",
        ],
        tagging=[],
    )
    user = f"Question: {question}\n"
    return f"{system}\n{user}\nAnswer:"


def _build_rag_pretrained(question: str, internal_context: str) -> str:
    system = _system(
        mode="rag_pretrained",
        allowed=[
            "INTERNAL_CONTEXT (retrieved from the project knowledge base)",
            "Your pretrained / general knowledge",
        ],
        forbidden=["Web search results"],
        tagging=[
            "No tag for facts taken from INTERNAL_CONTEXT.",
            "\"[General knowledge]\" immediately after any fact taken from your pretrained knowledge.",
        ],
    )
    extra = (
        "You MUST use BOTH sources:\n"
        "  * Use INTERNAL_CONTEXT as the primary, authoritative source. Prefer it when sources disagree.\n"
        "  * Actively add complementary facts from your pretrained knowledge that INTERNAL_CONTEXT does NOT cover.\n"
        "  * Tag every pretrained-derived statement inline with [General knowledge].\n"
        "  * If you cannot contribute any pretrained facts beyond INTERNAL_CONTEXT, append exactly this sentence:\n"
        "    \"No additional pretrained facts contributed.\"\n"
        f"{_EXCEL_GUARDRAIL}\n"
        f"{_FIELD_FOCUS}"
    )
    user = (
        f"Question: {question}\n\n"
        f"INTERNAL_CONTEXT:\n{internal_context or '(empty)'}\n"
    )
    return f"{system}\n{extra}\n{user}\nAnswer:"


def _build_rag_pretrained_web(question: str, internal_context: str, web_context: str) -> str:
    system = _system(
        mode="rag_pretrained_web",
        allowed=[
            "INTERNAL_CONTEXT (retrieved from the project knowledge base)",
            "WEB_CONTEXT (live Tavily search results)",
            "Your pretrained / general knowledge",
        ],
        forbidden=[],
        tagging=[
            "No tag for facts taken from INTERNAL_CONTEXT.",
            "\"[Web]\" immediately after any fact taken from WEB_CONTEXT.",
            "\"[General knowledge]\" immediately after any fact taken from your pretrained knowledge.",
        ],
    )
    extra = (
        "You MUST use ALL THREE sources. Combine them with this priority when they disagree:\n"
        "  INTERNAL_CONTEXT > WEB_CONTEXT > pretrained knowledge.\n"
        "  * Use INTERNAL_CONTEXT as the primary authority.\n"
        "  * Use WEB_CONTEXT to add recent or missing facts; tag them [Web].\n"
        "  * Use pretrained knowledge for complementary background; tag those [General knowledge].\n"
        f"{_EXCEL_GUARDRAIL}\n"
        f"{_FIELD_FOCUS}"
    )
    user = (
        f"Question: {question}\n\n"
        f"INTERNAL_CONTEXT:\n{internal_context or '(empty)'}\n\n"
        f"WEB_CONTEXT:\n{web_context or '(empty)'}\n"
    )
    return f"{system}\n{extra}\n{user}\nAnswer:"


def build_prompt(
    mode: str,
    question: str,
    internal_context: str = "",
    web_context: str = "",
) -> str:
    if mode == "rag_only":
        return _build_rag_only(question, internal_context)
    if mode == "no_rag":
        return _build_no_rag(question)
    if mode == "rag_pretrained":
        return _build_rag_pretrained(question, internal_context)
    if mode == "rag_pretrained_web":
        return _build_rag_pretrained_web(question, internal_context, web_context)
    raise ValueError(f"Unknown mode: {mode!r}. Valid modes: {VALID_MODES}")
