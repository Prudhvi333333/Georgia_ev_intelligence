"""
Phase 1 — Checkpoint / Resume

Persistent run-state so the pipeline can resume from where it stopped — even
after every Tavily key is exhausted, the laptop sleeps, or the process is
killed mid-run.

State is held in a single JSON file at logs/phase1_checkpoint.json:

    {
      "run_id":           "phase1_2026-05-04T12:00:00Z",
      "started_at":       "...",
      "last_updated":     "...",
      "completed_companies": ["Adient", "ACM Georgia LLC", ...],
      "failed_companies":    [{"company": "...", "error": "..."}],
      "completed_urls":      ["https://...", ...]   # docs already saved
    }

The pipeline:
  1. Loads the checkpoint at startup.
  2. Skips any company already in `completed_companies`.
  3. Skips any URL already in `completed_urls` (per-company resume).
  4. After every successful company / URL, atomically rewrites the file.
  5. If TavilyAllKeysExhausted is raised, the file already reflects everything
     that was finished — re-run after adding more keys to .env to continue.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.logger import get_logger

logger = get_logger("phase1.checkpoint")


def _checkpoint_path() -> Path:
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "phase1_checkpoint.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Checkpoint:
    """Thread-safe, file-backed Phase-1 progress tracker."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _checkpoint_path()
        self._lock = threading.Lock()
        self._state: dict[str, Any] = self._load()

    # ── persistence ──────────────────────────────────────────
    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return self._fresh_state()
        try:
            with self._path.open("r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Checkpoint file unreadable (%s); starting fresh.", exc)
            return self._fresh_state()

        # Normalise older / partial files
        state.setdefault("run_id", f"phase1_{_now_iso()}")
        state.setdefault("started_at", _now_iso())
        state.setdefault("completed_companies", [])
        state.setdefault("failed_companies", [])
        state.setdefault("completed_urls", [])
        state.setdefault("stats", {})
        return state

    @staticmethod
    def _fresh_state() -> dict[str, Any]:
        ts = _now_iso()
        return {
            "run_id": f"phase1_{ts}",
            "started_at": ts,
            "last_updated": ts,
            "completed_companies": [],
            "failed_companies": [],
            "completed_urls": [],
            "stats": {},
        }

    def _flush(self) -> None:
        """Atomic write — never leaves a half-written checkpoint on disk."""
        self._state["last_updated"] = _now_iso()
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self._path.parent),
            prefix=".checkpoint_",
            suffix=".tmp",
            delete=False,
        )
        try:
            json.dump(self._state, tmp, indent=2, ensure_ascii=False, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
        finally:
            tmp.close()
        os.replace(tmp.name, self._path)

    # ── queries ──────────────────────────────────────────────
    @property
    def path(self) -> Path:
        return self._path

    @property
    def run_id(self) -> str:
        return str(self._state.get("run_id", ""))

    def is_company_done(self, company_name: str) -> bool:
        with self._lock:
            return company_name in self._state["completed_companies"]

    def is_url_done(self, url: str) -> bool:
        with self._lock:
            return url in self._state["completed_urls"]

    def completed_company_count(self) -> int:
        with self._lock:
            return len(self._state["completed_companies"])

    def completed_url_count(self) -> int:
        with self._lock:
            return len(self._state["completed_urls"])

    # ── mutations ────────────────────────────────────────────
    def mark_url_done(self, url: str) -> None:
        with self._lock:
            if url in self._state["completed_urls"]:
                return
            self._state["completed_urls"].append(url)
            self._flush()

    def mark_company_done(self, company_name: str, stats: dict[str, Any] | None = None) -> None:
        with self._lock:
            if company_name not in self._state["completed_companies"]:
                self._state["completed_companies"].append(company_name)
            if stats:
                self._state["stats"][company_name] = stats
            self._flush()

    def mark_company_failed(self, company_name: str, error: str) -> None:
        with self._lock:
            self._state["failed_companies"] = [
                f for f in self._state["failed_companies"]
                if f.get("company") != company_name
            ]
            self._state["failed_companies"].append({
                "company": company_name,
                "error": error[:500],
                "at": _now_iso(),
            })
            self._flush()

    def reset(self) -> None:
        """Wipe checkpoint and start over — used by `--rerun-all`."""
        with self._lock:
            self._state = self._fresh_state()
            self._flush()
        logger.info("Checkpoint reset — new run id %s", self.run_id)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self._state["run_id"],
                "started_at": self._state["started_at"],
                "last_updated": self._state["last_updated"],
                "completed_companies": len(self._state["completed_companies"]),
                "completed_urls": len(self._state["completed_urls"]),
                "failed_companies": len(self._state["failed_companies"]),
            }


_singleton: Checkpoint | None = None
_singleton_lock = threading.Lock()


def get_checkpoint() -> Checkpoint:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = Checkpoint()
        return _singleton


def reset_checkpoint_singleton() -> None:
    """Tests only — drops the in-process Checkpoint object."""
    global _singleton
    with _singleton_lock:
        _singleton = None
