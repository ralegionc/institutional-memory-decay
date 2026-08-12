"""Central configuration: paths, API endpoints, and tunable constants.

Everything downstream reads from here so that a run can be redirected at a
different data directory (e.g. a scratch dir in tests) by setting ``IMD_DATA``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = Path(os.environ.get("IMD_DATA", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("IMD_OUTPUT", REPO_ROOT / "outputs"))
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DB = DATA_DIR / "http_cache.sqlite"

# Canonical artifact filenames. Keeping them in one place means the CLI, the
# tests and the notebooks never disagree about where a stage writes its output.
ARTIFACTS = {
    "corpus": "corpus.parquet",
    "revisions": "revisions.parquet",
    "snapshots": "snapshots.parquet",
    "snapshot_links": "snapshot_links.parquet",
    "pageviews": "pageviews.parquet",
    "panel": "panel.parquet",
    "fits": "decay_fits.parquet",
    "network": "network_metrics.parquet",
    "edge_survival": "edge_survival.parquet",
    "category_stats": "category_stats.parquet",
    "survival": "dormancy_survival.parquet",
    "resistance": "resistance_index.parquet",
}


def data_path(key: str) -> Path:
    """Resolve a canonical artifact name to a path inside the data dir."""
    return DATA_DIR / ARTIFACTS[key]


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# API endpoints
# --------------------------------------------------------------------------

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
PAGEVIEWS_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    "/en.wikipedia/all-access/user/{title}/monthly/{start}/{end}"
)

# The Wikimedia API policy asks for a descriptive User-Agent with contact info.
# Override via IMD_CONTACT before running a real collection.
USER_AGENT = (
    "InstitutionalMemoryDecay/1.0 (research project; "
    f"{os.environ.get('IMD_CONTACT', 'https://github.com/  set IMD_CONTACT')})"
)

# Pageviews only exist from 2015-07 onward; earlier months are structurally
# missing rather than zero, and must not be treated as "no attention".
PAGEVIEWS_EPOCH = "2015-07-01"


# --------------------------------------------------------------------------
# Collection / analysis parameters
# --------------------------------------------------------------------------


@dataclass
class Settings:
    """Tunable knobs for a collection + analysis run."""

    # --- corpus ---
    year_start: int = 2004
    year_end: int = 2022
    per_category_year: int = 4  # articles sampled per (category, year)
    min_revisions: int = 40  # drop stubs that never accumulated attention
    min_months_observed: int = 36  # need 3y of post-event history to fit decay

    # --- API politeness ---
    requests_per_second: float = 8.0
    max_retries: int = 5
    timeout: int = 60
    revision_page_limit: int = 500  # API max for anonymous users
    max_revisions_per_article: int = 20_000

    # --- snapshots ---
    # Years after the event at which we materialise the full wikitext. Sparse
    # by design: content fetches are ~100x heavier than revision metadata.
    snapshot_offsets_years: tuple[float, ...] = (0.0, 0.25, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0)

    # --- decay fitting ---
    fit_max_months: int = 240
    peak_window_months: int = 3  # months averaged to define the attention peak
    dormancy_threshold: float = 0.05  # fraction of peak defining "dormant"
    dormancy_window_months: int = 12  # trailing window used for the test

    # --- statistics ---
    bootstrap_iterations: int = 2000
    random_seed: int = 20260812

    def __post_init__(self) -> None:
        if self.year_end < self.year_start:
            raise ValueError("year_end must be >= year_start")
        if not 0 < self.dormancy_threshold < 1:
            raise ValueError("dormancy_threshold must be in (0, 1)")


DEFAULT_SETTINGS = Settings()


# --------------------------------------------------------------------------
# Event taxonomy
# --------------------------------------------------------------------------
# Each event class maps to a list of Wikipedia category-name templates. The
# {year} placeholder is filled in per collection year. Categories were chosen
# so that members are overwhelmingly *events* (things with a date) rather than
# people, places or concepts -- that distinction is what makes an anchored
# "months since event" axis meaningful.

EVENT_CATEGORIES: dict[str, list[str]] = {
    "natural_disaster": [
        "Category:{year} disasters",
        "Category:Earthquakes in {year}",
        "Category:{year} floods",
        "Category:{year} wildfires",
    ],
    "transport_accident": [
        "Category:Aviation accidents and incidents in {year}",
        "Category:Railway accidents in {year}",
        "Category:Maritime incidents in {year}",
    ],
    "armed_conflict": [
        "Category:Conflicts in {year}",
        "Category:Battles involving the United States",
        "Category:{year} in military history",
    ],
    "terrorism": [
        "Category:Terrorist incidents in {year}",
        "Category:Mass shootings in {year}",
    ],
    "election": [
        "Category:{year} elections",
        "Category:Referendums in {year}",
    ],
    "political_scandal": [
        "Category:Political scandals",
        "Category:{year} in politics",
        "Category:Impeachment",
    ],
    "science_tech": [
        "Category:{year} in science",
        "Category:{year} in spaceflight",
        "Category:Products introduced in {year}",
    ],
    "epidemic": [
        "Category:Disease outbreaks",
        "Category:{year} in health",
        "Category:Epidemics",
    ],
    "sport": [
        "Category:{year} in sports",
        "Category:Association football competitions",
    ],
    "culture_award": [
        "Category:{year} awards",
        "Category:{year} in music",
        "Category:{year} films",
    ],
    "economy": [
        "Category:{year} in economics",
        "Category:Financial crises",
        "Category:Corporate scandals",
    ],
}

# Titles matching these prefixes/patterns are structurally not events.
TITLE_EXCLUSION_PATTERNS: tuple[str, ...] = (
    r"^List of ",
    r"^Timeline of ",
    r"^Index of ",
    r"^Outline of ",
    r"\(disambiguation\)$",
    r"^Category:",
    r"^Template:",
    r"^Portal:",
    r"^Wikipedia:",
    r"^\d{4}$",  # bare year articles
)


@dataclass
class RunManifest:
    """Provenance record written alongside every collection run."""

    started_at: str
    settings: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    notes: str = ""
