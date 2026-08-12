"""A polite, cached client for the MediaWiki, Wikidata and Pageviews APIs.

Three things matter for a collection run that touches tens of thousands of
endpoints and may be interrupted halfway:

1. **Caching.** Every response is stored in a SQLite blob keyed by the request.
   Re-running a collection after a crash costs nothing and, more importantly,
   makes the analysis reproducible from a fixed snapshot of Wikipedia rather
   than from "whatever the site looked like when you ran it".
2. **Rate limiting.** A token-bucket limiter keeps us under the Wikimedia
   API etiquette guidelines regardless of how the callers are structured.
3. **Retries.** Transient 429/5xx responses are retried with exponential
   backoff and jitter; ``maxlag`` is honoured so we back off when the
   replication lag is high instead of making it worse.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import threading
import time
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from . import config

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


class TokenBucket:
    """Thread-safe token bucket. ``acquire`` blocks until a token is free."""

    def __init__(self, rate_per_second: float, burst: int | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self.rate = rate_per_second
        self.capacity = burst if burst is not None else max(1, int(rate_per_second))
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = (1.0 - self._tokens) / self.rate
            time.sleep(deficit)


# --------------------------------------------------------------------------
# Response cache
# --------------------------------------------------------------------------


class ResponseCache:
    """Compressed on-disk cache of JSON responses keyed by (endpoint, params)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._local = threading.local()
        with self._connect() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS responses (
                       key TEXT PRIMARY KEY,
                       fetched_at REAL NOT NULL,
                       payload BLOB NOT NULL
                   )"""
            )

    def _connect(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=30)
            con.execute("PRAGMA journal_mode=WAL")
            self._local.con = con
        return con

    @staticmethod
    def make_key(endpoint: str, params: dict[str, Any] | None) -> str:
        canonical = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
        return f"{endpoint}?{canonical}"

    def get(self, key: str) -> dict | None:
        row = self._connect().execute(
            "SELECT payload FROM responses WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(zlib.decompress(row[0]).decode("utf-8"))

    def put(self, key: str, value: dict) -> None:
        blob = zlib.compress(json.dumps(value).encode("utf-8"), level=6)
        con = self._connect()
        con.execute(
            "INSERT OR REPLACE INTO responses (key, fetched_at, payload) VALUES (?, ?, ?)",
            (key, time.time(), blob),
        )
        con.commit()

    def __len__(self) -> int:
        return self._connect().execute("SELECT COUNT(*) FROM responses").fetchone()[0]


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class WikiClient:
    """Thin wrapper over the endpoints this project needs."""

    def __init__(
        self,
        settings: config.Settings | None = None,
        cache_path: Path | None = None,
        session: requests.Session | None = None,
        use_cache: bool = True,
    ) -> None:
        self.settings = settings or config.DEFAULT_SETTINGS
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT})
        self.bucket = TokenBucket(self.settings.requests_per_second)
        self.cache = ResponseCache(cache_path or config.CACHE_DB) if use_cache else None
        self.stats = {"hits": 0, "misses": 0, "errors": 0}

    # -- low level ---------------------------------------------------------

    def _request(self, url: str, params: dict[str, Any] | None = None) -> dict:
        key = ResponseCache.make_key(url, params)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                self.stats["hits"] += 1
                return cached

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries):
            self.bucket.acquire()
            try:
                resp = self.session.get(
                    url, params=params, timeout=self.settings.timeout
                )
            except requests.RequestException as exc:  # network-level failure
                last_error = exc
                self._backoff(attempt)
                continue

            if resp.status_code in RETRYABLE_STATUS:
                last_error = RuntimeError(f"HTTP {resp.status_code} for {url}")
                # Honour Retry-After when the server supplies it.
                wait = resp.headers.get("Retry-After")
                if wait and wait.isdigit():
                    time.sleep(min(int(wait), 60))
                else:
                    self._backoff(attempt)
                continue

            if resp.status_code == 404:
                # Legitimately absent (e.g. no pageviews for this article).
                payload: dict = {"__missing__": True}
                if self.cache is not None:
                    self.cache.put(key, payload)
                self.stats["misses"] += 1
                return payload

            resp.raise_for_status()
            payload = resp.json()

            # MediaWiki signals replication lag in-band, with HTTP 200.
            if isinstance(payload, dict) and "error" in payload:
                code = payload["error"].get("code")
                if code == "maxlag":
                    last_error = RuntimeError("maxlag")
                    self._backoff(attempt, base=5.0)
                    continue

            if self.cache is not None:
                self.cache.put(key, payload)
            self.stats["misses"] += 1
            return payload

        self.stats["errors"] += 1
        raise RuntimeError(f"request failed after retries: {url}") from last_error

    @staticmethod
    def _backoff(attempt: int, base: float = 1.0) -> None:
        time.sleep(base * (2**attempt) * (0.5 + random.random()))

    def _query(self, params: dict[str, Any], api: str | None = None) -> dict:
        merged = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        return self._request(api or config.WIKI_API, merged)

    def _query_paged(
        self, params: dict[str, Any], api: str | None = None, max_pages: int = 200
    ) -> Iterator[dict]:
        """Follow the MediaWiki ``continue`` protocol."""
        cont: dict[str, Any] = {}
        for _ in range(max_pages):
            payload = self._query({**params, **cont}, api=api)
            if payload.get("__missing__"):
                return
            yield payload
            if "continue" not in payload:
                return
            cont = payload["continue"]

    # -- categories --------------------------------------------------------

    def category_members(self, category: str, limit: int = 500) -> list[dict]:
        """Return page members (namespace 0 only) of a category."""
        out: list[dict] = []
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmnamespace": "0",
            "cmlimit": "500",
            "cmtype": "page",
        }
        for payload in self._query_paged(params):
            members = payload.get("query", {}).get("categorymembers", [])
            out.extend(members)
            if len(out) >= limit:
                break
        return out[:limit]

    # -- page metadata -----------------------------------------------------

    def page_info(self, titles: list[str]) -> list[dict]:
        """Batch page metadata including the Wikidata item id."""
        out: list[dict] = []
        for chunk in _chunks(titles, 50):
            payload = self._query(
                {
                    "action": "query",
                    "titles": "|".join(chunk),
                    "prop": "info|pageprops",
                    "inprop": "created",
                    "ppprop": "wikibase_item|disambiguation",
                }
            )
            out.extend(payload.get("query", {}).get("pages", []))
        return out

    # -- revisions ---------------------------------------------------------

    def revisions(self, title: str) -> list[dict]:
        """Full revision metadata for one article, oldest first.

        Only metadata (timestamp / size / user / flags) is requested -- content
        is fetched separately and sparsely, because pulling every revision's
        wikitext for a large article is gigabytes of transfer for information
        we do not use.
        """
        params = {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvprop": "ids|timestamp|size|user|flags|comment|tags",
            "rvlimit": str(self.settings.revision_page_limit),
            "rvdir": "newer",
        }
        revs: list[dict] = []
        for payload in self._query_paged(params, max_pages=200):
            pages = payload.get("query", {}).get("pages", [])
            if not pages or "missing" in pages[0]:
                break
            revs.extend(pages[0].get("revisions", []))
            if len(revs) >= self.settings.max_revisions_per_article:
                break
        return revs

    def revision_content(self, revid: int) -> str | None:
        """Wikitext of a single revision."""
        payload = self._query(
            {
                "action": "query",
                "prop": "revisions",
                "revids": str(revid),
                "rvprop": "content",
                "rvslots": "main",
            }
        )
        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            return None
        revs = pages[0].get("revisions", [])
        if not revs:
            return None
        slots = revs[0].get("slots", {}).get("main", {})
        return slots.get("content")

    # -- wikidata ----------------------------------------------------------

    def wikidata_claims(self, qids: list[str]) -> dict[str, dict]:
        """Fetch claims for Wikidata items, batched 50 at a time."""
        out: dict[str, dict] = {}
        for chunk in _chunks(qids, 50):
            payload = self._request(
                config.WIKIDATA_API,
                {
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": "claims",
                    "format": "json",
                },
            )
            out.update(payload.get("entities", {}))
        return out

    # -- pageviews ---------------------------------------------------------

    def pageviews(self, title: str, start: str, end: str) -> list[dict]:
        """Monthly pageviews. ``start``/``end`` are YYYYMMDD strings."""
        url = config.PAGEVIEWS_API.format(
            title=quote(title.replace(" ", "_"), safe=""), start=start, end=end
        )
        payload = self._request(url)
        if payload.get("__missing__"):
            return []
        return payload.get("items", [])


def _chunks(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
