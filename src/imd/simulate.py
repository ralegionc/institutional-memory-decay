"""A generative model of event-article histories, with known ground truth.

This exists for two reasons, and the second is the important one.

The obvious reason is offline development: the full pipeline can be exercised
without touching the Wikipedia API.

The real reason is **estimator validation**. Every headline number in this
project is a fitted parameter -- a half-life, a decay exponent, a floor height.
Those are only worth reporting if the fitting code can recover a parameter it
was given. The simulator draws ground-truth decay parameters per article,
generates edit histories consistent with them, and ``validate_recovery`` then
checks that the estimators get them back. If the recovery correlation is poor,
the finding is an artefact of the fitting code, not a fact about Wikipedia --
and we would rather find that out here than in the write-up.

The generative process per article:

- a burst: attention rises over a short ramp to a peak,
- a decay: power law with exponent *b* onto a floor *c*,
- Poisson noise on the monthly rate,
- an occasional **resurgence** (anniversary coverage, a later event that
  recontextualises this one) modelled as a log-normal bump. Resurgences are
  what make real forgetting curves non-monotonic, and any estimator that
  cannot tolerate them is not fit for the real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Class-level priors: (peak_rate_log_mean, decay_exponent_mean, floor_share_mean,
# resurgence_probability). Chosen to span a realistic range of behaviours so
# that the cross-class machinery has something to detect; they are NOT claims
# about real Wikipedia.
CLASS_PRIORS: dict[str, tuple[float, float, float, float]] = {
    "natural_disaster": (3.4, 1.25, 0.020, 0.35),
    "transport_accident": (3.0, 1.45, 0.012, 0.20),
    "armed_conflict": (3.8, 0.75, 0.060, 0.55),
    "terrorism": (3.6, 0.95, 0.045, 0.60),
    "election": (3.9, 1.10, 0.035, 0.30),
    "political_scandal": (3.2, 0.85, 0.050, 0.50),
    "science_tech": (2.9, 1.05, 0.030, 0.25),
    "epidemic": (3.5, 0.80, 0.055, 0.45),
    "sport": (3.1, 1.60, 0.010, 0.15),
    "culture_award": (2.8, 1.70, 0.008, 0.12),
    "economy": (3.3, 0.90, 0.040, 0.40),
}


@dataclass
class GroundTruth:
    title: str
    event_class: str
    amplitude: float
    exponent: float
    floor: float
    ramp_months: float
    n_resurgences: int


def _monthly_rate(
    months: np.ndarray, amplitude: float, exponent: float, floor: float, ramp: float
) -> np.ndarray:
    """Rise-then-decay-onto-floor mean rate."""
    decay = amplitude * np.power(months + 1.0, -exponent) + floor
    # Smooth ramp into the peak so month 0 is not automatically the maximum;
    # real articles often peak in month 1 as coverage accumulates.
    rise = 1 - np.exp(-(months + 0.5) / max(ramp, 1e-3))
    return decay * rise


def simulate_corpus(
    n_articles: int = 240,
    seed: int = 20260812,
    start_year: int = 2006,
    end_year: int = 2021,
    max_months: int = 220,
) -> dict[str, pd.DataFrame]:
    """Generate a complete synthetic dataset matching the real schema.

    Returns a dict with ``corpus``, ``revisions``, ``snapshots``,
    ``snapshot_links`` and ``ground_truth`` frames.
    """
    rng = np.random.default_rng(seed)
    classes = list(CLASS_PRIORS)

    corpus_rows: list[dict] = []
    revision_frames: list[pd.DataFrame] = []
    truth_rows: list[dict] = []
    per_article_series: dict[str, np.ndarray] = {}

    now = pd.Timestamp("2026-08-01", tz="UTC")

    for i in range(n_articles):
        event_class = classes[i % len(classes)]
        peak_log, exp_mean, floor_share, resurge_p = CLASS_PRIORS[event_class]

        year = int(rng.integers(start_year, end_year + 1))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 27))
        event_date = pd.Timestamp(year=year, month=month, day=day, tz="UTC")

        horizon = int(min((now - event_date).days / 30.44, max_months))
        if horizon < 40:
            continue

        amplitude = float(np.exp(rng.normal(peak_log, 0.55)))
        exponent = float(np.clip(rng.normal(exp_mean, 0.22), 0.25, 3.0))
        floor = float(max(amplitude * floor_share * np.exp(rng.normal(0, 0.5)), 0.02))
        ramp = float(np.clip(rng.gamma(2.0, 0.6), 0.15, 6.0))

        months = np.arange(horizon, dtype=float)
        rate = _monthly_rate(months, amplitude, exponent, floor, ramp)

        n_resurge = int(rng.binomial(3, resurge_p))
        for _ in range(n_resurge):
            centre = float(rng.uniform(12, max(horizon - 6, 13)))
            width = float(rng.uniform(1.0, 4.0))
            height = float(rate[int(centre)] * rng.uniform(1.5, 8.0) + 1.0)
            rate = rate + height * np.exp(-0.5 * ((months - centre) / width) ** 2)

        counts = rng.poisson(np.clip(rate, 0, None))
        if counts.sum() < 40:
            continue

        title = f"{event_class.replace('_', ' ').title()} of {year} #{i:03d}"
        per_article_series[title] = counts
        true_hl_continuous, true_hl_binned = _true_half_life(
            amplitude, exponent, floor, ramp
        )

        corpus_rows.append(
            {
                "title": title,
                "pageid": 1_000_000 + i,
                "event_class": event_class,
                "source_category": f"Category:{year} synthetic",
                "category_year": year,
                "n_classes": 1,
                "wikidata_id": f"Q{9_000_000 + i}",
                "event_date": event_date,
                "date_source": "wikidata:P585",
                "date_precision": 11,
                "created_at": event_date + pd.Timedelta(hours=int(rng.integers(1, 72))),
                "is_disambiguation": False,
                "months_observed": horizon,
                "creation_lag_years": 0.0,
                "current_length": 0,
            }
        )
        truth_rows.append(
            {
                "title": title,
                "event_class": event_class,
                "true_amplitude": amplitude,
                "true_exponent": exponent,
                "true_floor": floor,
                "true_floor_share": floor / (amplitude + floor),
                "true_ramp": ramp,
                "n_resurgences": n_resurge,
                "true_half_life": true_hl_continuous,
                "true_half_life_binned": true_hl_binned,
            }
        )
        revision_frames.append(
            _synthesise_revisions(title, event_date, counts, rng)
        )

    corpus = pd.DataFrame(corpus_rows)
    revisions = (
        pd.concat(revision_frames, ignore_index=True)
        if revision_frames
        else pd.DataFrame()
    )
    snapshots, links = _synthesise_snapshots(corpus, revisions, per_article_series, rng)

    return {
        "corpus": corpus,
        "revisions": revisions,
        "snapshots": snapshots,
        "snapshot_links": links,
        "ground_truth": pd.DataFrame(truth_rows),
    }


def _true_half_life(
    amplitude: float, exponent: float, floor: float, ramp: float, horizon: int = 600
) -> tuple[float, float]:
    """Ground truth for both half-life estimators.

    Returns ``(continuous, binned)``. The two exist because the two estimators
    answer subtly different questions and comparing each against the wrong one
    manufactures a bias that is definitional rather than statistical:

    - ``continuous`` is the half-life of the noiseless rate curve on a fine
      grid -- the target for the *fitted* half-life.
    - ``binned`` applies the very same smoothing-and-interpolation procedure
      the empirical estimator uses to the noiseless monthly rate -- the target
      for the *empirical* half-life. Any gap that remains is genuine estimator
      error from Poisson noise and resurgences, which is what we want to see.
    """
    from .decay import empirical_half_life

    fine = np.arange(0, horizon, 0.02, dtype=float)
    rate = _monthly_rate(fine, amplitude, exponent, floor, ramp)
    peak_idx = int(np.argmax(rate))
    below = np.flatnonzero(rate[peak_idx:] <= 0.5 * rate[peak_idx])
    continuous = float(fine[peak_idx + below[0]] - fine[peak_idx]) if below.size else np.inf

    months = np.arange(horizon, dtype=float)
    binned = empirical_half_life(_monthly_rate(months, amplitude, exponent, floor, ramp))
    return continuous, binned


def _synthesise_revisions(
    title: str, event_date: pd.Timestamp, counts: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Expand monthly counts into individual revision rows."""
    n_editors_pool = max(int(counts.sum() ** 0.6), 5)
    rows = []
    size = float(rng.integers(2000, 8000))
    for month_index, n in enumerate(counts):
        n = int(n)
        # Bot activity is roughly constant in time, unlike human attention --
        # which is precisely why the pipeline filters it out before fitting.
        n_bot = int(rng.poisson(0.35))
        for _ in range(n + n_bot):
            is_bot = _ >= n
            offset_days = month_index * 30.44 + rng.uniform(0, 30.44)
            delta = float(rng.normal(60, 400)) if not is_bot else float(rng.normal(0, 12))
            # Articles grow fast early and plateau; net growth shrinks with age.
            delta *= np.exp(-month_index / 60.0) + 0.15
            size = max(size + delta, 500.0)
            rows.append(
                {
                    "title": title,
                    "timestamp": event_date + pd.Timedelta(days=offset_days),
                    "size": size,
                    "user": f"bot_{rng.integers(1, 6)}bot"
                    if is_bot
                    else f"user_{rng.integers(1, n_editors_pool)}",
                    "minor": bool(rng.random() < 0.3),
                    "anon": bool(rng.random() < 0.22),
                    "comment_len": int(rng.integers(0, 90)),
                    "is_bot": is_bot,
                    "is_revert": bool(rng.random() < 0.06),
                    "delta": delta,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["revid"] = np.arange(len(df)) + abs(hash(title)) % 10_000_000
    return df


SNAPSHOT_OFFSETS = (0.0, 0.25, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0)


def _synthesise_snapshots(
    corpus: pd.DataFrame,
    revisions: pd.DataFrame,
    series: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Snapshot features and the inter-article link edge list.

    Link generation is deliberately *not* uniform: an article's baseline
    in-degree is tied to its total attention (big events get referenced more),
    and edges are lost over time at a rate tied to how fast the source article
    is decaying. That gives the network module a real signal to find, so a
    null result from it on synthetic data would indicate a bug.
    """
    if corpus.empty:
        return pd.DataFrame(), pd.DataFrame()

    titles = corpus["title"].tolist()
    attention = np.array([series[t].sum() for t in titles], dtype=float)
    weight = attention / attention.sum()

    feature_rows: list[dict] = []
    link_rows: list[dict] = []
    now = pd.Timestamp("2026-08-01", tz="UTC")

    # Each article starts with a stable set of outbound links, weighted toward
    # high-attention targets (big events get referenced more). These then
    # survive or die across snapshots rather than being redrawn.
    article_links: dict[str, list[str]] = {}
    for pos, title in enumerate(titles):
        n0 = int(rng.poisson(3.0 + 4.0 * weight[pos] * len(titles)))
        picks = rng.choice(len(titles), size=min(max(n0, 1), 12), replace=False, p=weight)
        article_links[title] = [titles[j] for j in picks if titles[j] != title]

    for idx, row in corpus.iterrows():
        title = row["title"]
        counts = series[title]
        event_date = row["event_date"]

        for offset in SNAPSHOT_OFFSETS:
            target = event_date + pd.Timedelta(days=offset * 365.25)
            if target > now:
                continue
            months = int(offset * 12)
            cum_edits = float(counts[: max(months, 1)].sum())

            # Size and citation count saturate with cumulative editing.
            n_bytes = int(3000 + 950 * cum_edits**0.62 * rng.uniform(0.8, 1.2))
            n_refs = int(max(1, 0.09 * cum_edits**0.75 * rng.uniform(0.7, 1.3)))
            n_urls = int(n_refs * rng.uniform(0.75, 1.05))

            # Link rot accumulates with the *age* of the citations, roughly
            # following the ~10%/year attrition reported for web references,
            # partially offset by archiving campaigns that ramp up after 2013.
            rot_fraction = 1 - np.exp(-0.11 * offset)
            archive_fraction = min(0.85, 0.10 + 0.075 * offset) * rng.uniform(0.7, 1.2)
            n_dead = int(rng.binomial(max(n_urls, 0), min(rot_fraction * 0.45, 0.9)))
            n_archived = int(rng.binomial(max(n_urls, 0), min(archive_fraction, 0.95)))

            feature_rows.append(
                {
                    "title": title,
                    "offset_years": offset,
                    "revid": int(1e9 + idx * 100 + offset * 10),
                    "snapshot_time": target,
                    "n_bytes": n_bytes,
                    "n_words": int(n_bytes / 6.2),
                    "n_refs": n_refs,
                    "n_reused_refs": int(n_refs * rng.uniform(0.05, 0.3)),
                    "n_cite_templates": int(n_refs * rng.uniform(0.7, 1.0)),
                    "n_urls": n_urls,
                    "n_unique_domains": int(n_urls * rng.uniform(0.5, 0.9)),
                    "n_dead_link_tags": n_dead,
                    "n_url_status_dead": int(n_dead * rng.uniform(0, 0.5)),
                    "n_archive_urls": n_archived,
                    "n_wayback_urls": n_archived,
                    "n_citation_needed": int(rng.poisson(1.2 + 0.25 * offset)),
                    "n_maintenance_tags": int(rng.poisson(0.6 + 0.1 * offset)),
                    "n_sections": int(4 + n_bytes / 4000),
                    "n_images": int(rng.poisson(2 + n_bytes / 12000)),
                    "n_templates": int(rng.poisson(8 + n_refs * 0.8)),
                    "n_tables": int(rng.poisson(0.7)),
                    "n_wikilinks": int(30 + n_bytes / 300),
                    "n_unique_wikilinks": int(25 + n_bytes / 350),
                    "n_categories": int(rng.poisson(6)),
                    "archive_rate": n_archived / n_urls if n_urls else 0.0,
                    "rot_rate": n_dead / n_urls if n_urls else 0.0,
                    "refs_per_kb": 1000 * n_refs / n_bytes if n_bytes else 0.0,
                }
            )

            # --- inter-article links -----------------------------------
            # Links must be *persistent objects*, not resampled each snapshot.
            # Drawing an independent set per snapshot would give an edge
            # survival rate of essentially zero by construction, and the
            # network module would appear to detect catastrophic link loss in
            # data that contains no such signal.
            recent_rate = float(counts[max(months - 12, 0) : max(months, 1)].mean())
            # A decayed source article stops maintaining its outbound links.
            hazard = 0.10 + 0.22 / (1.0 + recent_rate)
            survivors = [
                tgt for tgt in article_links[title]
                if rng.random() > 1 - np.exp(-hazard * max(offset - 0.25, 0))
            ]
            article_links[title] = survivors

            # New links accrue as the article is written, then taper off.
            n_new = int(rng.poisson(max(0.4, 2.2 * np.log1p(recent_rate))))
            if n_new:
                picks = rng.choice(len(titles), size=min(n_new, 8), p=weight)
                for t_idx in picks:
                    if titles[t_idx] != title and titles[t_idx] not in survivors:
                        survivors.append(titles[t_idx])

            for tgt in survivors:
                link_rows.append(
                    {"title": title, "offset_years": offset, "target": tgt}
                )

    return pd.DataFrame(feature_rows), pd.DataFrame(link_rows)


# --------------------------------------------------------------------------
# Estimator validation
# --------------------------------------------------------------------------


def validate_recovery(fits: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    """Compare fitted parameters against the values used to generate the data."""
    from scipy import stats

    merged = fits.merge(ground_truth, on="title", how="inner")
    if merged.empty:
        return {"error": "no overlap"}

    def _corr(a: str, b: str) -> dict:
        x = merged[a].replace([np.inf, -np.inf], np.nan)
        y = merged[b].replace([np.inf, -np.inf], np.nan)
        mask = x.notna() & y.notna()
        if mask.sum() < 5:
            return {"n": int(mask.sum()), "pearson": np.nan, "spearman": np.nan}
        r = stats.pearsonr(x[mask], y[mask])
        rho = stats.spearmanr(x[mask], y[mask])
        bias = float(np.median(x[mask] - y[mask]))
        return {
            "n": int(mask.sum()),
            "pearson": float(r.statistic),
            "spearman": float(rho.statistic),
            "median_bias": bias,
            "median_abs_error": float(np.median(np.abs(x[mask] - y[mask]))),
        }

    out = {
        "n_articles": len(merged),
        "exponent": _corr("power_exponent", "true_exponent"),
        "floor_share": _corr("floor_share_of_peak", "true_floor_share"),
        # Each estimator is scored against the ground truth defined the same
        # way it is (see _true_half_life).
        "half_life_empirical": _corr("empirical_half_life", "true_half_life_binned"),
        "half_life_fitted": _corr("half_life_fitted", "true_half_life"),
    }
    # The data are generated by a power-law-with-floor process, so a correctly
    # specified selector should pick that form most of the time.
    if "best_model" in merged:
        out["best_model_share"] = (
            merged["best_model"].value_counts(normalize=True).to_dict()
        )
    return out
