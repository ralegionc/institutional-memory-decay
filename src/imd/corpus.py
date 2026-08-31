"""Build the event corpus: which Wikipedia articles are we studying, and when
did the thing they describe actually happen?

The second question is the load-bearing one. Every decay curve in this project
is measured on an axis of "months since the event", so a wrong t0 does not add
noise -- it smears the peak and biases the fitted half-life upward. We resolve
t0 in priority order:

1. **Wikidata point-in-time (P585)** or start-time (P580) on the article's
   linked item. This is a human-curated date for the event itself.
2. **First revision timestamp**, used when Wikidata has no date.

Source 2 is a good fallback for contemporary events (an article about a plane
crash is typically created within hours) but is wrong for historical ones (the
article on a 1912 shipwreck was created in 2001). We therefore record which
source was used and, in the analysis, we can restrict to Wikidata-dated events
as a robustness check.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict

import pandas as pd

from . import config
from .wikiapi import WikiClient

log = logging.getLogger(__name__)

_EXCLUDE_RE = re.compile("|".join(config.TITLE_EXCLUSION_PATTERNS))

# Wikidata properties carrying an event date, in order of preference.
DATE_PROPERTIES = ("P585", "P580", "P571")  # point in time, start time, inception

# Wikidata precision codes: 11 = day, 10 = month, 9 = year. A year-precision
# claim parses to January 1st, which is not a date so much as a placeholder --
# on a "months since event" axis it smears t0 by up to eleven months, and it is
# systematically *early*, which inflates every half-life measured from it. A
# contemporaneous article's first revision is the better estimate in that case,
# so anything coarser than month precision is rejected here and falls through
# to the revision-timestamp fallback.
MIN_DATE_PRECISION = 10


def _is_plausible_event_title(title: str) -> bool:
    return not _EXCLUDE_RE.search(title)


def gather_candidates(
    client: WikiClient, settings: config.Settings | None = None
) -> pd.DataFrame:
    """Walk the category taxonomy and collect candidate event articles."""
    settings = settings or config.DEFAULT_SETTINGS
    rows: list[dict] = []
    years = range(settings.year_start, settings.year_end + 1)

    for event_class, templates in config.EVENT_CATEGORIES.items():
        for year in years:
            for template in templates:
                category = template.format(year=year)
                try:
                    members = client.category_members(
                        category, limit=settings.per_category_year * 6
                    )
                except Exception as exc:  # a missing category is normal
                    log.debug("category %s failed: %s", category, exc)
                    continue
                kept = 0
                for member in members:
                    title = member.get("title", "")
                    if not _is_plausible_event_title(title):
                        continue
                    rows.append(
                        {
                            "title": title,
                            "pageid": member.get("pageid"),
                            "event_class": event_class,
                            "source_category": category,
                            "category_year": year,
                        }
                    )
                    kept += 1
                    if kept >= settings.per_category_year:
                        break

    if not rows:
        return pd.DataFrame(
            columns=["title", "pageid", "event_class", "source_category", "category_year"]
        )

    df = pd.DataFrame(rows)
    # An article can appear under several categories; keep its first (most
    # specific) assignment and record how many classes claimed it.
    multiplicity = df.groupby("title")["event_class"].nunique().rename("n_classes")
    df = df.drop_duplicates(subset="title", keep="first").merge(
        multiplicity, on="title", how="left"
    )
    log.info("gathered %d candidate articles", len(df))
    return df.reset_index(drop=True)


def _parse_wikidata_time(claim: dict) -> tuple[pd.Timestamp | None, int | None]:
    """Extract a timestamp and its precision from a Wikidata claim."""
    try:
        value = claim["mainsnak"]["datavalue"]["value"]
    except (KeyError, TypeError):
        return None, None
    raw = value.get("time", "")
    precision = value.get("precision")
    m = re.match(r"^[+-](\d{4})-(\d{2})-(\d{2})", raw)
    if not m:
        return None, precision
    year, month, day = (int(g) for g in m.groups())
    if year < 1900 or year > 2100:
        return None, precision
    # Wikidata uses 00 for unknown month/day at coarse precision.
    month = month or 1
    day = day or 1
    try:
        return pd.Timestamp(year=year, month=month, day=day, tz="UTC"), precision
    except ValueError:
        return None, precision


def resolve_event_dates(
    client: WikiClient, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Attach ``event_date`` and ``date_source`` columns to the candidate set."""
    if candidates.empty:
        return candidates.assign(event_date=pd.NaT, date_source=None, date_precision=None)

    titles = candidates["title"].tolist()
    info = {p["title"]: p for p in client.page_info(titles) if "title" in p}

    qids = {
        title: page.get("pageprops", {}).get("wikibase_item")
        for title, page in info.items()
    }
    entities = client.wikidata_claims([q for q in qids.values() if q])
    # Creation timestamps: the fallback t0 for articles Wikidata has no date
    # for, and the input to the retrospective-article filter for every article.
    created_times = client.first_revision_times(titles)

    records = []
    for title in titles:
        page = info.get(title, {})
        qid = qids.get(title)
        event_date: pd.Timestamp | None = None
        precision: int | None = None
        source = None

        if qid and qid in entities:
            claims = entities[qid].get("claims", {})
            for prop in DATE_PROPERTIES:
                for claim in claims.get(prop, []):
                    parsed, parsed_precision = _parse_wikidata_time(claim)
                    if parsed is None:
                        continue
                    if parsed_precision is not None and parsed_precision < MIN_DATE_PRECISION:
                        continue
                    event_date, precision = parsed, parsed_precision
                    source = f"wikidata:{prop}"
                    break
                if event_date is not None:
                    break

        created = created_times.get(title)
        created_ts = pd.to_datetime(created, utc=True, errors="coerce") if created else pd.NaT

        if event_date is None and pd.notna(created_ts):
            event_date = created_ts
            source = "first_revision"

        records.append(
            {
                "title": title,
                "wikidata_id": qid,
                "event_date": event_date,
                "date_source": source,
                "date_precision": precision,
                "created_at": created_ts,
                "is_disambiguation": "disambiguation" in page.get("pageprops", {}),
                "is_redirect": bool(page.get("redirect", False)),
                "current_length": page.get("length"),
            }
        )

    resolved = candidates.merge(pd.DataFrame(records), on="title", how="left")
    return resolved


def filter_corpus(
    resolved: pd.DataFrame, settings: config.Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the inclusion criteria and record why each article was dropped.

    Returns ``(kept, dropped)``. The dropped rows keep their ``drop_reason`` so
    a run can be audited for a filter that fired far more often than expected.
    """
    settings = settings or config.DEFAULT_SETTINGS
    df = resolved.copy()
    df["drop_reason"] = None

    def mark(mask: pd.Series, reason: str) -> None:
        target = mask & df["drop_reason"].isna()
        df.loc[target, "drop_reason"] = reason

    mark(df["is_disambiguation"].fillna(False), "disambiguation")
    # Category listings include redirects, whose "edit history" is a handful of
    # retargeting edits and says nothing about attention to the event.
    if "is_redirect" in df:
        mark(df["is_redirect"].fillna(False), "redirect")
    mark(df["event_date"].isna(), "no_event_date")

    # The article must exist for long enough after the event for a decay curve
    # to be observable at all.
    now = pd.Timestamp.now(tz="UTC")
    months_observed = (now - df["event_date"]).dt.days / 30.44
    df["months_observed"] = months_observed
    mark(months_observed < settings.min_months_observed, "too_recent")

    # Guard against a Wikidata date that predates the article by decades: that
    # is a historical-anniversary article, not a contemporaneous record, and its
    # edit history says nothing about how the event was forgotten.
    lag_years = (df["created_at"] - df["event_date"]).dt.days / 365.25
    df["creation_lag_years"] = lag_years
    mark(lag_years > 3, "retrospective_article")
    mark(lag_years < -0.5, "date_after_creation")

    kept = df[df["drop_reason"].isna()].drop(columns=["drop_reason"]).reset_index(drop=True)
    dropped = df[df["drop_reason"].notna()].reset_index(drop=True)
    log.info(
        "corpus filter: kept %d, dropped %d (%s)",
        len(kept),
        len(dropped),
        dropped["drop_reason"].value_counts().to_dict(),
    )
    # Only JSON-serialisable values may live in ``attrs``: pandas writes them
    # into the parquet metadata, so parking the dropped frame here makes the
    # corpus unwritable. The rows themselves go to their own artifact.
    kept.attrs["dropped_counts"] = dropped["drop_reason"].value_counts().to_dict()
    return kept, dropped


def build_corpus(
    client: WikiClient, settings: config.Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end corpus construction. Returns ``(corpus, dropped)``."""
    settings = settings or config.DEFAULT_SETTINGS
    candidates = gather_candidates(client, settings)
    resolved = resolve_event_dates(client, candidates)
    corpus, dropped = filter_corpus(resolved, settings)
    corpus.attrs["settings"] = asdict(settings)
    return corpus, dropped
