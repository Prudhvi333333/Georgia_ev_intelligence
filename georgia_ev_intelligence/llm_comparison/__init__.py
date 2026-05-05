"""sreeja-arch LLM comparison + official RAGAS evaluation pipeline.

Self-contained module: reads env vars directly, does not depend on
shared.config / phase4_agent / evaluate.format_runner. The four generation
modes (rag_only, no_rag, rag_pretrained, rag_pretrained_web) and the official
ragas evaluator (judge = Kimi 2.6) live here.
"""
