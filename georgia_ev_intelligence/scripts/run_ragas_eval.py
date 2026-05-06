"""Deprecated compatibility entrypoint.

The old custom Ollama-judge RAGAS-style evaluator was removed on the
sreeja-arch branch. Use scripts/run_llm_evaluation.py, which wires the official
ragas package to the configured Kimi/OpenAI-compatible judge.
"""
from __future__ import annotations

SMOKE_QUESTIONS: list[dict] = []
FIFTY_QUESTIONS: list[dict] = []


def main() -> int:
    raise SystemExit(
        "run_ragas_eval.py has been removed on sreeja-arch. "
        "Run: python -m scripts.run_llm_evaluation --run-id <run_id>"
    )


if __name__ == "__main__":
    main()
