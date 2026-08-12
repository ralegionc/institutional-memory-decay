"""The time-evolving citation network between event articles, and link rot.

Two distinct notions of "citation" are measured here, and keeping them apart is
the point of the module:

**Internal citation (salience).** Wikipedia articles link to each other. When
article X links to event Y, an editor decided Y was worth invoking to explain
something else. The in-degree of Y within the event corpus is therefore a
measure of Y's *usefulness as a reference point* -- how often the encyclopedia
reaches for it when describing other things. Because we rebuild this graph
from each snapshot's wikitext, we can watch an event gain and lose that status
over time. Losing in-degree is a sharper form of forgetting than losing edits:
it means the event stopped being a thing other events are explained in terms of.

**External citation (evidence).** The URLs an article cites decay physically.
We track the density of dead-link tags and the archive-coverage rate over the
same snapshots. This is the substrate rotting underneath the article, and it
proceeds on its own schedule regardless of how much attention the article gets.

Edge survival ties them together: of the internal links present at the first
snapshot, what fraction still exist *k* years later? That curve is the
network-level analogue of the attention forgetting curve.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------


def build_snapshot_graphs(
    links: pd.DataFrame, corpus_titles: set[str] | None = None
) -> dict[float, nx.DiGraph]:
    """One directed graph per snapshot offset, restricted to corpus members.

    Restricting to the corpus is what makes in-degree interpretable: an
    event's in-degree from the whole of Wikipedia is dominated by year and
    country index pages, whereas in-degree from other *events* measures
    whether the encyclopedia treats it as explanatory context.
    """
    graphs: dict[float, nx.DiGraph] = {}
    if links.empty:
        return graphs

    titles = corpus_titles if corpus_titles is not None else set(links["title"])

    for offset, group in links.groupby("offset_years"):
        g = nx.DiGraph()
        g.add_nodes_from(titles)
        edges = [
            (s, t)
            for s, t in zip(group["title"], group["target"])
            if t in titles and s != t
        ]
        g.add_edges_from(edges)
        graphs[float(offset)] = g
    return graphs


def network_metrics(graphs: dict[float, nx.DiGraph]) -> pd.DataFrame:
    """Per-article, per-snapshot structural position."""
    rows: list[dict] = []
    for offset, g in sorted(graphs.items()):
        n_nodes = g.number_of_nodes()
        if n_nodes == 0:
            continue
        try:
            pagerank = nx.pagerank(g, alpha=0.85, max_iter=200)
        except Exception:
            pagerank = dict.fromkeys(g.nodes, 1.0 / n_nodes)
        undirected = g.to_undirected()
        core = nx.core_number(undirected) if undirected.number_of_edges() else {}

        for node in g.nodes:
            rows.append(
                {
                    "title": node,
                    "offset_years": offset,
                    "in_degree": g.in_degree(node),
                    "out_degree": g.out_degree(node),
                    "pagerank": pagerank.get(node, np.nan),
                    "k_core": core.get(node, 0),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Edge survival
# --------------------------------------------------------------------------


def edge_survival(graphs: dict[float, nx.DiGraph]) -> pd.DataFrame:
    """Fraction of the earliest snapshot's edges still present at each later one.

    Also reports edge *formation*: links that did not exist at the baseline and
    appear later. Retrospective linking -- an event acquiring inbound links
    years after the fact -- is the signature of an event being reinterpreted as
    a precedent, and it runs against the general decay trend.
    """
    if not graphs:
        return pd.DataFrame()

    offsets = sorted(graphs)
    baseline = graphs[offsets[0]]
    base_edges = set(baseline.edges)

    rows = []
    for offset in offsets:
        g = graphs[offset]
        edges = set(g.edges)
        surviving = base_edges & edges
        rows.append(
            {
                "offset_years": offset,
                "n_edges": len(edges),
                "n_baseline_edges": len(base_edges),
                "n_surviving": len(surviving),
                "survival_rate": len(surviving) / len(base_edges) if base_edges else np.nan,
                "n_new": len(edges - base_edges),
                "growth_ratio": len(edges) / len(base_edges) if base_edges else np.nan,
                "density": nx.density(g),
            }
        )
    return pd.DataFrame(rows)


def per_article_link_trajectory(metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-snapshot metrics into one row per article.

    ``indeg_slope`` is the OLS slope of in-degree against snapshot age, i.e.
    whether the encyclopedia is pointing at this event more or less over time.
    """
    if metrics.empty:
        return pd.DataFrame()

    rows = []
    for title, group in metrics.groupby("title"):
        group = group.sort_values("offset_years")
        x = group["offset_years"].to_numpy(dtype=float)
        y = group["in_degree"].to_numpy(dtype=float)
        slope = np.polyfit(x, y, 1)[0] if len(x) >= 2 and np.ptp(x) > 0 else np.nan
        first, last = y[0], y[-1]
        rows.append(
            {
                "title": title,
                "indeg_first": first,
                "indeg_last": last,
                "indeg_max": float(y.max()),
                "indeg_slope": slope,
                "indeg_retention": last / first if first > 0 else np.nan,
                "pagerank_last": float(group["pagerank"].iloc[-1]),
                "k_core_last": float(group["k_core"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Link rot
# --------------------------------------------------------------------------


def rot_trajectories(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Aggregate external-citation health across snapshot ages."""
    if snapshots.empty:
        return pd.DataFrame()

    df = snapshots.copy()
    df["dead_total"] = df["n_dead_link_tags"] + df["n_url_status_dead"]
    df["dead_per_100_urls"] = np.where(
        df["n_urls"] > 0, 100 * df["dead_total"] / df["n_urls"], np.nan
    )
    df["archive_pct"] = np.where(
        df["n_urls"] > 0, 100 * df["n_archive_urls"] / df["n_urls"], np.nan
    )

    grouped = df.groupby("offset_years")
    out = grouped.agg(
        n_articles=("title", "nunique"),
        median_urls=("n_urls", "median"),
        median_refs=("n_refs", "median"),
        median_bytes=("n_bytes", "median"),
        mean_dead_per_100=("dead_per_100_urls", "mean"),
        median_dead_per_100=("dead_per_100_urls", "median"),
        mean_archive_pct=("archive_pct", "mean"),
        mean_citation_needed=("n_citation_needed", "mean"),
        mean_maintenance=("n_maintenance_tags", "mean"),
    ).reset_index()
    return out


def rot_by_class(snapshots: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Rot trajectories split by event class."""
    if snapshots.empty or corpus.empty:
        return pd.DataFrame()
    merged = snapshots.merge(corpus[["title", "event_class"]], on="title", how="left")
    merged["dead_total"] = merged["n_dead_link_tags"] + merged["n_url_status_dead"]
    merged["dead_per_100_urls"] = np.where(
        merged["n_urls"] > 0, 100 * merged["dead_total"] / merged["n_urls"], np.nan
    )
    merged["archive_pct"] = np.where(
        merged["n_urls"] > 0, 100 * merged["n_archive_urls"] / merged["n_urls"], np.nan
    )
    return (
        merged.groupby(["event_class", "offset_years"])
        .agg(
            n=("title", "nunique"),
            median_dead_per_100=("dead_per_100_urls", "median"),
            median_archive_pct=("archive_pct", "median"),
            median_refs=("n_refs", "median"),
            median_bytes=("n_bytes", "median"),
        )
        .reset_index()
    )


def content_shrinkage(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Do articles actually shrink, or just stop growing?

    The popular claim is that neglected articles decay in *size*. Measuring it
    requires comparing each article to its own maximum rather than to the
    corpus, because absolute byte counts differ by two orders of magnitude
    between a plane crash and a war.
    """
    if snapshots.empty:
        return pd.DataFrame()
    rows = []
    for title, group in snapshots.groupby("title"):
        group = group.sort_values("offset_years")
        size = group["n_bytes"].to_numpy(dtype=float)
        refs = group["n_refs"].to_numpy(dtype=float)
        peak_i = int(np.argmax(size))
        rows.append(
            {
                "title": title,
                "size_peak": float(size.max()),
                "size_final": float(size[-1]),
                "size_peak_offset": float(group["offset_years"].iloc[peak_i]),
                "size_retention": float(size[-1] / size.max()) if size.max() > 0 else np.nan,
                "shrank": bool(size[-1] < 0.95 * size.max()),
                "refs_peak": float(refs.max()),
                "refs_final": float(refs[-1]),
                "refs_retention": float(refs[-1] / refs.max()) if refs.max() > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)
