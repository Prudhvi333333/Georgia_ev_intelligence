"""
Tavily API key rotation.

Reads all configured Tavily keys, observes failed quota/rate/auth responses,
and retries with the next available key in the same process.
"""
from __future__ import annotations

import threading
from typing import Any

import httpx

from collector.shared.config import Config
from collector.shared.logger import get_logger

logger = get_logger("shared.tavily")

_ROTATABLE_STATUS_CODES = {401, 402, 403, 429, 432}
_ROTATABLE_ERROR_MARKERS = (
    "api key",
    "auth",
    "credit",
    "exceed",
    "exhaust",
    "forbidden",
    "insufficient",
    "limit",
    "payment",
    "quota",
    "rate",
    "unauthorized",
)


def _mask_key(key: str) -> str:
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


def _body_text(response: httpx.Response) -> str:
    try:
        return response.text[:1000]
    except Exception:
        return ""


def _is_rotatable_failure(response: httpx.Response) -> bool:
    if response.status_code in _ROTATABLE_STATUS_CODES:
        return True
    if response.status_code < 400:
        return False
    body = _body_text(response).lower()
    return any(marker in body for marker in _ROTATABLE_ERROR_MARKERS)


def _payload_failure_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    parts: list[str] = []
    for key in ("error", "detail", "message", "status"):
        value = payload.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _is_rotatable_payload(payload: Any) -> bool:
    text = _payload_failure_text(payload).lower()
    return bool(text) and any(marker in text for marker in _ROTATABLE_ERROR_MARKERS)


class TavilyKeyObserver:
    """Process-local observer for Tavily key health and rotation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys = Config.get().tavily_api_keys
        self._active_index = 0
        self._disabled: set[str] = set()

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    def active_key(self) -> str:
        with self._lock:
            if not self._keys:
                raise EnvironmentError(
                    "No Tavily API keys configured. Set TAVILY_API_KEYS, "
                    "TAVILY_API_KEY_1..6, or TAVILY_API_KEY in .env."
                )
            for offset in range(len(self._keys)):
                idx = (self._active_index + offset) % len(self._keys)
                key = self._keys[idx]
                if key not in self._disabled:
                    self._active_index = idx
                    return key
            raise RuntimeError("All configured Tavily API keys appear exhausted or invalid.")

    def mark_failed(self, key: str, reason: str) -> None:
        with self._lock:
            if key in self._disabled:
                return
            self._disabled.add(key)
            if key in self._keys:
                self._active_index = (self._keys.index(key) + 1) % len(self._keys)
        logger.warning("Tavily key %s disabled after failure: %s", _mask_key(key), reason[:180])


_observer: TavilyKeyObserver | None = None
_observer_lock = threading.Lock()


def get_tavily_observer() -> TavilyKeyObserver:
    global _observer
    with _observer_lock:
        if _observer is None:
            _observer = TavilyKeyObserver()
        return _observer


def reset_tavily_observer() -> None:
    """Force reload of Tavily keys. Useful after changing .env in tests."""
    global _observer
    with _observer_lock:
        _observer = None


async def async_tavily_post(
    endpoint: str,
    payload: dict[str, Any],
    timeout: float = 60.0,
) -> dict[str, Any]:
    """
    POST to a Tavily endpoint with automatic key rotation.

    endpoint examples:
      - "https://api.tavily.com/search"
      - "https://api.tavily.com/extract"
    """
    observer = get_tavily_observer()
    attempts = max(1, len(observer.keys))
    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        for _ in range(attempts):
            key = observer.active_key()
            try:
                response = await client.post(endpoint, json={**payload, "api_key": key})
                if _is_rotatable_failure(response):
                    observer.mark_failed(key, f"HTTP {response.status_code}: {_body_text(response)}")
                    last_exc = httpx.HTTPStatusError(
                        f"Tavily key failed with HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    continue
                response.raise_for_status()
                data = response.json()
                if _is_rotatable_payload(data):
                    observer.mark_failed(key, _payload_failure_text(data))
                    last_exc = RuntimeError(_payload_failure_text(data))
                    continue
                return data
            except httpx.HTTPStatusError as exc:
                if _is_rotatable_failure(exc.response):
                    observer.mark_failed(
                        key,
                        f"HTTP {exc.response.status_code}: {_body_text(exc.response)}",
                    )
                    last_exc = exc
                    continue
                raise

    if last_exc:
        raise last_exc
    raise RuntimeError("Tavily request failed before a response was received.")


def tavily_post(
    endpoint: str,
    payload: dict[str, Any],
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Synchronous version of async_tavily_post."""
    observer = get_tavily_observer()
    attempts = max(1, len(observer.keys))
    last_exc: Exception | None = None

    with httpx.Client(timeout=timeout) as client:
        for _ in range(attempts):
            key = observer.active_key()
            try:
                response = client.post(endpoint, json={**payload, "api_key": key})
                if _is_rotatable_failure(response):
                    observer.mark_failed(key, f"HTTP {response.status_code}: {_body_text(response)}")
                    last_exc = httpx.HTTPStatusError(
                        f"Tavily key failed with HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    continue
                response.raise_for_status()
                data = response.json()
                if _is_rotatable_payload(data):
                    observer.mark_failed(key, _payload_failure_text(data))
                    last_exc = RuntimeError(_payload_failure_text(data))
                    continue
                return data
            except httpx.HTTPStatusError as exc:
                if _is_rotatable_failure(exc.response):
                    observer.mark_failed(
                        key,
                        f"HTTP {exc.response.status_code}: {_body_text(exc.response)}",
                    )
                    last_exc = exc
                    continue
                raise

    if last_exc:
        raise last_exc
    raise RuntimeError("Tavily request failed before a response was received.")
