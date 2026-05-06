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
        "Strict instruction: Answer ONLY using INTERNAL_CONTEXT. Do not use any pretrained\n"
        "knowledge. Do not use any web data. If INTERNAL_CONTEXT does not contain enough\n"
        "information to answer the question, your entire answer must be exactly:\n"
        f"\"{INSUFFICIENT_CONTEXT_SENTENCE}\"\n"
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
