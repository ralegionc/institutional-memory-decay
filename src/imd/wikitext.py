"""Feature extraction from raw wikitext.

We deliberately parse wikitext with regexes rather than rendering HTML. The
markup we care about -- citation templates, dead-link tags, archive URLs --
exists *only* in the source; the rendered page hides exactly the maintenance
signals that make link rot measurable. Rendering would also require a network
round-trip per revision, which is the dominant cost of the whole pipeline.

The regexes are intentionally forgiving. Wikitext is not a regular language and
a small fraction of pathological pages will be miscounted; what matters is that
the *same* measurement is applied to every snapshot of every article, so the
bias is common-mode and the time trends remain interpretable.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse

# --- citations -------------------------------------------------------------
RE_REF_OPEN = re.compile(r"<ref\b(?![^>]*\bname\s*=\s*[^>]*/>)", re.IGNORECASE)
RE_REF_ANY = re.compile(r"<ref\b", re.IGNORECASE)
RE_REF_SELF_CLOSING = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)
# The negative lookahead matters: "{{citation needed}}" is a *complaint about a
# missing* citation, and counting it as a citation inverts the signal.
RE_CITE_TEMPLATE = re.compile(
    r"\{\{\s*(cite[ _][a-z]+|citation(?!\s*needed))\b", re.IGNORECASE
)
RE_SFN = re.compile(r"\{\{\s*(sfn|harvnb|harv)\b", re.IGNORECASE)

# --- link rot --------------------------------------------------------------
RE_DEAD_LINK = re.compile(
    r"\{\{\s*(dead[ _]?link|broken[ _]?link|link[ _]rot|dl)\b", re.IGNORECASE
)
RE_URL_STATUS_DEAD = re.compile(r"url[-_]?status\s*=\s*dead", re.IGNORECASE)
RE_ARCHIVE_URL = re.compile(r"archive[-_]?url\s*=\s*\S", re.IGNORECASE)
RE_WAYBACK = re.compile(r"web\.archive\.org|archive\.today|archive\.is", re.IGNORECASE)
RE_CITATION_NEEDED = re.compile(
    r"\{\{\s*(citation[ _]needed|cn|fact)\s*[|}]", re.IGNORECASE
)
RE_MAINTENANCE = re.compile(
    r"\{\{\s*(update|outdated|refimprove|unreferenced|original[ _]research|"
    r"disputed|npov|cleanup)\b",
    re.IGNORECASE,
)

# --- structure -------------------------------------------------------------
RE_URL = re.compile(r"https?://[^\s\]|<>{}\"]+", re.IGNORECASE)
RE_SECTION = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
RE_WIKILINK = re.compile(r"\[\[([^\]\[|#<>{}]+)(?:\|[^\]]*)?\]\]")
RE_FILE_LINK = re.compile(r"\[\[\s*(File|Image)\s*:", re.IGNORECASE)
RE_TEMPLATE = re.compile(r"\{\{")
RE_TABLE = re.compile(r"\{\|")
RE_CATEGORY = re.compile(r"\[\[\s*Category\s*:\s*([^\]|]+)", re.IGNORECASE)
RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

_NON_ARTICLE_PREFIXES = (
    "file:", "image:", "category:", "template:", "wikipedia:", "help:",
    "portal:", "wikt:", "media:", "special:", "talk:", "user:", "s:", "commons:",
)


@dataclass
class WikitextFeatures:
    """Structural measurements of a single revision's wikitext."""

    n_bytes: int = 0
    n_words: int = 0
    n_refs: int = 0
    n_reused_refs: int = 0
    n_cite_templates: int = 0
    n_urls: int = 0
    n_unique_domains: int = 0
    n_dead_link_tags: int = 0
    n_url_status_dead: int = 0
    n_archive_urls: int = 0
    n_wayback_urls: int = 0
    n_citation_needed: int = 0
    n_maintenance_tags: int = 0
    n_sections: int = 0
    n_images: int = 0
    n_templates: int = 0
    n_tables: int = 0
    n_wikilinks: int = 0
    n_unique_wikilinks: int = 0
    n_categories: int = 0
    wikilinks: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)

    @property
    def archive_rate(self) -> float:
        """Share of external citations that have been archived.

        Rising archive rate is the fingerprint of editors *repairing* rot; a
        high dead-link count with a low archive rate is rot left unrepaired.
        """
        return self.n_archive_urls / self.n_urls if self.n_urls else 0.0

    @property
    def rot_rate(self) -> float:
        return (
            (self.n_dead_link_tags + self.n_url_status_dead) / self.n_urls
            if self.n_urls
            else 0.0
        )

    @property
    def refs_per_kb(self) -> float:
        return 1000 * self.n_refs / self.n_bytes if self.n_bytes else 0.0

    def to_row(self) -> dict:
        """Flat dict for a DataFrame row (list fields excluded)."""
        row = {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"wikilinks", "domains"}
        }
        row.update(
            archive_rate=self.archive_rate,
            rot_rate=self.rot_rate,
            refs_per_kb=self.refs_per_kb,
        )
        return row


def normalise_title(raw: str) -> str:
    """Canonicalise a wikilink target to compare against article titles."""
    title = raw.strip().replace("_", " ")
    title = title.split("#", 1)[0].strip()
    if not title:
        return ""
    return title[0].upper() + title[1:]


def _is_article_link(target: str) -> bool:
    lowered = target.strip().lower()
    return not lowered.startswith(_NON_ARTICLE_PREFIXES)


def extract_features(text: str | None) -> WikitextFeatures:
    """Measure one revision of wikitext."""
    if not text:
        return WikitextFeatures()

    body = RE_COMMENT.sub("", text)

    urls = RE_URL.findall(body)
    domains: list[str] = []
    for url in urls:
        try:
            host = urlparse(url).netloc.lower()
        except ValueError:
            continue
        if host.startswith("www."):
            host = host[4:]
        if host:
            domains.append(host)

    links = [
        normalise_title(t)
        for t in RE_WIKILINK.findall(body)
        if _is_article_link(t)
    ]
    links = [t for t in links if t]

    total_refs = len(RE_REF_ANY.findall(body))
    reused = len(RE_REF_SELF_CLOSING.findall(body))

    return WikitextFeatures(
        n_bytes=len(text.encode("utf-8")),
        n_words=len(body.split()),
        # Distinct citations = all <ref> minus the pointer-only reuses.
        n_refs=max(total_refs - reused, 0),
        n_reused_refs=reused,
        n_cite_templates=len(RE_CITE_TEMPLATE.findall(body)) + len(RE_SFN.findall(body)),
        n_urls=len(urls),
        n_unique_domains=len(set(domains)),
        n_dead_link_tags=len(RE_DEAD_LINK.findall(body)),
        n_url_status_dead=len(RE_URL_STATUS_DEAD.findall(body)),
        n_archive_urls=len(RE_ARCHIVE_URL.findall(body)) or len(RE_WAYBACK.findall(body)),
        n_wayback_urls=len(RE_WAYBACK.findall(body)),
        n_citation_needed=len(RE_CITATION_NEEDED.findall(body)),
        n_maintenance_tags=len(RE_MAINTENANCE.findall(body)),
        n_sections=len(RE_SECTION.findall(body)),
        n_images=len(RE_FILE_LINK.findall(body)),
        n_templates=len(RE_TEMPLATE.findall(body)),
        n_tables=len(RE_TABLE.findall(body)),
        n_wikilinks=len(links),
        n_unique_wikilinks=len(set(links)),
        n_categories=len(RE_CATEGORY.findall(body)),
        wikilinks=sorted(set(links)),
        domains=sorted(Counter(domains)),
    )
