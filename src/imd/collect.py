"""Collection stages: revisions, sparse wikitext snapshots, pageviews.

Each stage is independently resumable -- it writes a parquet file and skips
articles already present in it -- because a full collection is a multi-hour
job against a rate-limited public API.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from . import config
from .wikiapi import WikiClient
from .wikitext import extract_features

log = logging.getLogger(__name__)

BOT_MARKERS = ("bot", "Bot")


def _looks_like_bot(user: str | None, tags: Iterable[str] | None) -> bool:
    """Heuristic bot detection.

    Bot edits inflate revision counts without representing human attention,
    which is the quantity this project is about. There is no authoritative
    public flag on the revision object, so we combine the ``bot`` change tag
    with the near-universal naming convention of ending the account in "bot".
    """
    if tags and any("bot" in str(t).lower() for t in tags):
        return True
    if not user:
        return False
    lowered = user.lower()
    return lowered.endswith("bot") or lowered.endswith("bot)")


# --------------------------------------------------------------------------
# Revisions
# --------------------------------------------------------------------------


def collect_revisions(
    client: WikiClient, corpus: pd.DataFrame, existing: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Full revision metadata for every article in the corpus."""
    done = set(existing["title"].unique()) if existing is not None and len(existing) else set()
    frames = [existing] if existing is not None and len(existing) else []

    todo = [t for t in corpus["title"] if t not in done]
    log.info("collecting revisions for %d articles (%d cached)", len(todo), len(done))

    for i, title in enumerate(todo, 1):
        try:
            revs = client.revisions(title)
        except Exception as exc:
            log.warning("revisions failed for %s: %s", title, exc)
            continue
        if not revs:
            continue
        rows = [
            {
                "title": title,
                "revid": r.get("revid"),
                "timestamp": r.get("timestamp"),
                "size": r.get("size", 0),
                "user": r.get("user"),
                "minor": bool(r.get("minor", False)),
                "anon": bool(r.get("anon", False)),
                "comment_len": len(r.get("comment") or ""),
                "is_bot": _looks_like_bot(r.get("user"), r.get("tags")),
                "is_revert": "mw-revert" in (r.get("tags") or [])
                or "revert" in (r.get("comment") or "").lower()[:40],
            }
            for r in revs
        ]
        frames.append(pd.DataFrame(rows))
        if i % 25 == 0:
            log.info("  ... %d/%d articles", i, len(todo))

    if not frames:
        return pd.DataFrame(
            columns=["title", "revid", "timestamp", "size", "user", "minor",
                     "anon", "comment_len", "is_bot", "is_revert"]
        )

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).drop_duplicates(subset=["title", "revid"])
    # Byte delta relative to the previous revision of the same article: the
    # signed magnitude of an edit is a better proxy for substantive work than
    # a raw edit count, which counts a typo fix the same as a rewrite.
    out = out.sort_values(["title", "timestamp"])
    out["delta"] = out.groupby("title")["size"].diff().fillna(out["size"])
    return out.reset_index(drop=True)


def filter_low_activity(
    corpus: pd.DataFrame, revisions: pd.DataFrame, settings: config.Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop articles that never accumulated enough human editing to fit.

    A decay curve needs something to decay from. Category listings are full of
    two-sentence stubs that took thirty edits in a decade; fitting a hazard
    model to those measures Poisson noise, not forgetting. The threshold is
    counted on *human* revisions, since a stub kept alive by bot maintenance
    has no attention to lose.

    Returns ``(kept, dropped)``; ``dropped`` carries the counts so the
    exclusion is auditable rather than invisible.
    """
    settings = settings or config.DEFAULT_SETTINGS
    if corpus.empty:
        return corpus, corpus.iloc[:0]

    if revisions.empty:
        counts = pd.Series(0, index=corpus["title"], dtype=int)
    else:
        counts = (
            revisions[~revisions["is_bot"]].groupby("title")["revid"].count()
        )

    out = corpus.copy()
    out["n_revisions"] = out["title"].map(counts).fillna(0).astype(int)
    eligible = out["n_revisions"] >= settings.min_revisions
    kept = out[eligible].reset_index(drop=True)
    dropped = out[~eligible].assign(drop_reason="too_few_revisions").reset_index(drop=True)
    log.info(
        "activity filter: kept %d, dropped %d below %d human revisions",
        len(kept), len(dropped), settings.min_revisions,
    )
    return kept, dropped


# --------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------


def pick_snapshot_revisions(
    revisions: pd.DataFrame,
    corpus: pd.DataFrame,
    settings: config.Settings | None = None,
    creation_tolerance_days: int = 90,
) -> pd.DataFrame:
    """For each (article, offset), find the last revision at or before t0+offset.

    "Last revision before the target date" -- not nearest -- so the snapshot is
    a state the article genuinely had at that moment, never a peek at the
    future.

    The one exception is the offset-0 snapshot. An article about an event is
    created *after* the event, sometimes hours later and sometimes weeks, so
    strictly there is no revision at or before t0 and the baseline snapshot
    would always be dropped. Where the article's first revision falls within
    ``creation_tolerance_days`` of the target we use it and record the lag, so
    the baseline exists and its staleness is visible in the data.
    """
    settings = settings or config.DEFAULT_SETTINGS
    dates = corpus.set_index("title")["event_date"]
    picks: list[dict] = []

    for title, group in revisions.groupby("title", sort=False):
        if title not in dates.index:
            continue
        t0 = dates.loc[title]
        if pd.isna(t0):
            continue
        group = group.sort_values("timestamp")
        stamps = group["timestamp"].to_numpy()
        for offset in settings.snapshot_offsets_years:
            target = t0 + pd.Timedelta(days=offset * 365.25)
            if target > pd.Timestamp.now(tz="UTC"):
                continue
            idx = stamps.searchsorted(target, side="right") - 1
            if idx < 0:
                # No revision yet at the target: allow the article's creation
                # revision to stand in, but only just after the target.
                first = group.iloc[0]
                if (first["timestamp"] - target).days > creation_tolerance_days:
                    continue
                idx = 0
            row = group.iloc[idx]
            picks.append(
                {
                    "title": title,
                    "offset_years": offset,
                    "revid": int(row["revid"]),
                    "snapshot_time": row["timestamp"],
                    "target_time": target,
                    "lag_days": (target - row["timestamp"]).days,
                }
            )

    return pd.DataFrame(picks)


def collect_snapshots(
    client: WikiClient, picks: pd.DataFrame, existing: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch wikitext for the chosen revisions and measure it.

    Returns ``(snapshot_features, snapshot_links)`` -- the second is the long
    edge list feeding the time-evolving article network.
    """
    done = set()
    if existing is not None and len(existing):
        done = set(existing["revid"].astype(int))

    feature_rows: list[dict] = []
    link_rows: list[dict] = []
    todo = picks[~picks["revid"].astype(int).isin(done)]
    log.info("fetching %d snapshot revisions (%d cached)", len(todo), len(done))

    for i, row in enumerate(todo.itertuples(index=False), 1):
        try:
            text = client.revision_content(int(row.revid))
        except Exception as exc:
            log.warning("content failed for revid %s: %s", row.revid, exc)
            continue
        feats = extract_features(text)
        feature_rows.append(
            {
                "title": row.title,
                "offset_years": row.offset_years,
                "revid": int(row.revid),
                "snapshot_time": row.snapshot_time,
                **feats.to_row(),
            }
        )
        for target in feats.wikilinks:
            link_rows.append(
                {"title": row.title, "offset_years": row.offset_years, "target": target}
            )
        if i % 50 == 0:
            log.info("  ... %d/%d snapshots", i, len(todo))

    features = pd.DataFrame(feature_rows)
    if existing is not None and len(existing):
        features = pd.concat([existing, features], ignore_index=True)
    return features, pd.DataFrame(link_rows)


# --------------------------------------------------------------------------
# Pageviews
# --------------------------------------------------------------------------


def collect_pageviews(client: WikiClient, corpus: pd.DataFrame) -> pd.DataFrame:
    """Monthly pageviews per article (only available from 2015-07)."""
    start = pd.Timestamp(config.PAGEVIEWS_EPOCH).strftime("%Y%m%d")
    end = pd.Timestamp.now().strftime("%Y%m%d")
    rows: list[dict] = []
    for title in corpus["title"]:
        try:
            items = client.pageviews(title, start, end)
        except Exception as exc:
            log.debug("pageviews failed for %s: %s", title, exc)
            continue
        for item in items:
            rows.append(
                {
                    "title": title,
                    "month": pd.to_datetime(item["timestamp"][:6], format="%Y%m", utc=True),
                    "views": item.get("views", 0),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Monthly panel
# --------------------------------------------------------------------------


def build_panel(
    corpus: pd.DataFrame,
    revisions: pd.DataFrame,
    pageviews: pd.DataFrame | None = None,
    settings: config.Settings | None = None,
) -> pd.DataFrame:
    """Assemble the balanced monthly panel that every later stage consumes.

    One row per (article, month-since-event), with zeros filled in for silent
    months -- an article nobody edited in month 40 must contribute a zero to
    the decay curve, not a gap.
    """
    settings = settings or config.DEFAULT_SETTINGS
    if revisions.empty:
        return pd.DataFrame()

    dates = corpus.set_index("title")["event_date"]
    rev = revisions[revisions["title"].isin(dates.index)].copy()
    rev["event_date"] = rev["title"].map(dates)
    rev = rev.dropna(subset=["event_date"])

    # Month index: 0 is the calendar month containing the event.
    delta_days = (rev["timestamp"] - rev["event_date"]).dt.total_seconds() / 86400.0
    rev["month_index"] = (delta_days / 30.44).apply(_floor_int)
    rev = rev[(rev["month_index"] >= 0) & (rev["month_index"] < settings.fit_max_months)]

    human = rev[~rev["is_bot"]]

    grouped = human.groupby(["title", "month_index"])
    agg = grouped.agg(
        edits=("revid", "count"),
        editors=("user", "nunique"),
        bytes_added=("delta", lambda s: float(s[s > 0].sum())),
        bytes_removed=("delta", lambda s: float(-s[s < 0].sum())),
        mean_abs_delta=("delta", lambda s: float(s.abs().mean())),
        anon_share=("anon", "mean"),
        revert_share=("is_revert", "mean"),
        end_size=("size", "last"),
    ).reset_index()

    bot_edits = (
        rev[rev["is_bot"]]
        .groupby(["title", "month_index"])["revid"]
        .count()
        .rename("bot_edits")
        .reset_index()
    )

    # Balance the panel out to each article's own observation horizon.
    frames = []
    now = pd.Timestamp.now(tz="UTC")
    for title, t0 in dates.items():
        if pd.isna(t0):
            continue
        horizon = int(min(((now - t0).days / 30.44), settings.fit_max_months))
        if horizon < 1:
            continue
        frames.append(
            pd.DataFrame({"title": title, "month_index": range(horizon)})
        )
    if not frames:
        return pd.DataFrame()
    skeleton = pd.concat(frames, ignore_index=True)

    panel = (
        skeleton.merge(agg, on=["title", "month_index"], how="left")
        .merge(bot_edits, on=["title", "month_index"], how="left")
        .fillna(
            {
                "edits": 0, "editors": 0, "bytes_added": 0.0, "bytes_removed": 0.0,
                "mean_abs_delta": 0.0, "anon_share": 0.0, "revert_share": 0.0,
                "bot_edits": 0,
            }
        )
    )
    panel["end_size"] = panel.groupby("title")["end_size"].ffill().fillna(0)

    if pageviews is not None and len(pageviews):
        pv = pageviews.copy()
        pv["event_date"] = pv["title"].map(dates)
        pv = pv.dropna(subset=["event_date"])
        pv["month_index"] = (
            (pv["month"] - pv["event_date"]).dt.total_seconds() / 86400.0 / 30.44
        ).apply(_floor_int)
        pv = pv[pv["month_index"] >= 0]
        panel = panel.merge(
            pv[["title", "month_index", "views"]], on=["title", "month_index"], how="left"
        )

    meta_cols = [c for c in ("event_class", "event_date", "date_source") if c in corpus]
    panel = panel.merge(corpus[["title", *meta_cols]], on="title", how="left")
    return panel.sort_values(["title", "month_index"]).reset_index(drop=True)


def _floor_int(x: float) -> int:
    import math

    return int(math.floor(x)) if pd.notna(x) else -1
