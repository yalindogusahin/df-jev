"""A small client for the TypeSafe System One wire protocol."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit

import httpx


class JevError(RuntimeError):
    """A request failed or the endpoint returned an invalid decision."""


def _number(value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected a numeric value")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Numeric value outside the expected range")
    return float(value)


def _validate_answer(answer: Any, question: dict) -> None:
    """Validate one typed answer against its question."""
    kind = question["type"]
    if answer["type"] != kind:
        raise ValueError("Wrong answer type")
    if kind == "noul":
        _number(answer["noul"], 0, 1)
        return
    _number(answer["confidence"], 0, 1)
    expected = (
        set(question["criteria"])
        if kind == "choice"
        else {str(i) for i in range(len(question["criteria"]))}
    )
    probabilities = answer["probabilities"]
    if set(probabilities) != expected:
        raise ValueError("Probability options do not match the question")
    if abs(sum(_number(p, 0, 1) for p in probabilities.values()) - 1) > 0.001:
        raise ValueError("Probabilities do not sum to one")
    if kind == "choice" and answer["choice"] not in expected:
        raise ValueError("Unknown category")
    if kind == "score":
        _number(answer["score"], 0, len(expected) - 1)


def _validate_many(response: Any, questions: dict) -> None:
    """Validate a response containing answers for several named questions."""
    if not isinstance(response.get("model"), str) or not response["model"]:
        raise ValueError("Missing model identifier")
    answers = response["answers"]
    if set(answers) != set(questions):
        raise ValueError("Answer names do not match the questions")
    for name, question in questions.items():
        _validate_answer(answers[name], question)


class JevClient:
    """Sequential HTTP client with bounded, session-local caching.

    Reuse a client across operations to reuse cached decisions. No rows or keys
    are persisted to disk. Model aliases can change; pin a model or clear_cache().
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 20,
        retries: int = 2,
        cache_size: int = 10_000,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = (
            base_url or os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1")
        ).rstrip("/")
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Base URL must be an http(s) URL without a query or fragment")
        if parsed.username or parsed.password:
            raise ValueError("Use api_key rather than credentials in the URL")
        if retries < 0 or cache_size < 0 or timeout <= 0:
            raise ValueError("Invalid timeout, retry count, or cache size")
        self.model = model or os.getenv("TYPESAFE_MODEL", "jev-latest")
        self.retries = retries
        self.cache_size = cache_size
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key or os.getenv('TYPESAFE_API_KEY', 'dummy')}"
            },
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> JevClient:  # noqa: PYI034 -- Python 3.10 has no typing.Self.
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def decide(self, state: dict, question: dict) -> tuple[dict, bool]:
        """Ask one question. Returns (response, cached); response has answers.result."""
        return self.decide_many(state, {"result": question})

    def decide_many(self, state: dict, questions: dict) -> tuple[dict, bool]:
        """Ask several named questions in one request. Returns (response, cached).

        The response has ``answers`` keyed by the question names. Safe to call from
        multiple threads; the cache is guarded by a lock and the HTTP client is
        thread-safe. Identical requests reuse the same in-memory decision.
        """
        payload = {"model": self.model, "state": state, "questions": questions}
        key = sha256(
            (self.base_url + json.dumps(payload, sort_keys=True, allow_nan=False)).encode()
        ).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return deepcopy(self._cache[key]), True
        suffix = "/systemone" if self.base_url.endswith("/v1") else "/v1/systemone"
        for attempt in range(self.retries + 1):
            retry_after = None
            try:
                response = self._http.post(self.base_url + suffix, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status != 429 and status < 500:
                    raise JevError(
                        f"Endpoint returned HTTP {status}; check the model and credentials"
                    ) from exc
                message = f"Endpoint returned HTTP {status} after retries"
                retry_after = exc.response.headers.get("retry-after")
            except httpx.RequestError:
                message = "Could not reach the endpoint or the request timed out"
            else:
                try:
                    result = response.json()
                    _validate_many(result, questions)
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    raise JevError("Endpoint returned an invalid decision") from exc
                if self.cache_size:
                    with self._lock:
                        self._cache[key] = deepcopy(result)
                        if len(self._cache) > self.cache_size:
                            self._cache.popitem(last=False)
                return result, False
            if attempt < self.retries:
                try:
                    delay = min(max(float(retry_after), 0), 30)
                except (ValueError, TypeError):
                    delay = 0.5 * 2**attempt
                time.sleep(delay)
        raise JevError(message)
