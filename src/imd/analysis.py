"""Cross-category comparison, the resistance index, and what predicts persistence.

This module answers the two questions the project exists to ask:

1. **Which kinds of events resist decay, and which fade fastest?** Answered by
   comparing decay statistics across event classes with bootstrap confidence
   intervals, an omnibus Kruskal-Wallis test, and BH-corrected pairwise tests.
   Medians and rank tests throughout, because half-life distributions are
   heavily right-skewed and a handful of never-forgotten events would otherwise
   drag every group mean around.

2. **What makes an event resist decay?** Answered by regressing persistence on
   properties measurable *at the time of the event* -- how big the initial
   burst was, how many distinct editors showed up, how well cited it was early,
   how central it was in the link graph. The point is not prediction accuracy;
   it is separating "events we remember because they were enormous" from
   "events we remember because they became explanatory reference points".
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from scipy import stats

from . import survival

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def bootstrap_ci(
    values, statistic=np.median, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI, NaN-safe. Returns (point, lower, upper)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    point = float(statistic(v))
    if v.size < 3:
        return (point, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True)
    stats_ = np.apply_along_axis(statistic, 1, draws)
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))


def benjamini_hochberg(pvalues) -> np.ndarray:
    """BH step-up FDR correction."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def cliffs_delta(a, b) -> float:
    """Non-parametric effect size in [-1, 1]; 0 means complete overlap.

    Reported alongside p-values because with a few hundred articles almost any
    difference reaches significance, and the question of interest is whether
    the difference is *large*.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    # Rank-based computation instead of the O(nm) pairwise comparison.
    combined = np.concatenate([a, b])
    ranks = stats.rankdata(combined)
    r_a = ranks[: a.size].sum()
    u = r_a - a.size * (a.size + 1) / 2
    return float(2 * u / (a.size * b.size) - 1)


# --------------------------------------------------------------------------
# Category comparison
# --------------------------------------------------------------------------

DECAY_METRICS = (
    "empirical_half_life",
    "half_life_fitted",
    "share_year1",
    "gini_time",
    "tail_rate_5y",
    "floor_share_of_peak",
    "power_exponent",
)


def category_summary(
    fits: pd.DataFrame,
    metrics: tuple[str, ...] = DECAY_METRICS,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Median + bootstrap CI of each decay metric within each event class."""
    if fits.empty or "event_class" not in fits:
        return pd.DataFrame()

    rows = []
    for metric in metrics:
        if metric not in fits.columns:
            continue
        for cls, group in fits.groupby("event_class"):
            values = group[metric].replace([np.inf, -np.inf], np.nan)
            point, lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "metric": metric,
                    "event_class": cls,
                    "n": int(values.notna().sum()),
                    "median": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "mean": float(values.mean(skipna=True)),
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                    # Share censored at infinity: articles whose fitted curve
                    # never halves. Reported explicitly instead of dropped.
                    "share_never_halved": float(
                        np.mean(np.isinf(group[metric].to_numpy(dtype=float)))
                    )
                    if metric.endswith("half_life")
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def omnibus_and_pairwise(
    fits: pd.DataFrame, metric: str = "empirical_half_life", min_group: int = 5
) -> tuple[dict, pd.DataFrame]:
    """Kruskal-Wallis across classes, then BH-corrected pairwise Mann-Whitney."""
    if fits.empty or metric not in fits.columns:
        return {}, pd.DataFrame()

    groups = {
        cls: g[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        for cls, g in fits.groupby("event_class")
    }
    groups = {k: v for k, v in groups.items() if len(v) >= min_group}
    if len(groups) < 2:
        return {}, pd.DataFrame()

    stat, p = stats.kruskal(*groups.values())
    # Epsilon-squared: the rank analogue of eta-squared.
    n_total = sum(len(v) for v in groups.values())
    eps_sq = (stat - len(groups) + 1) / (n_total - len(groups)) if n_total > len(groups) else np.nan
    omnibus = {
        "metric": metric,
        "test": "kruskal_wallis",
        "statistic": float(stat),
        "p_value": float(p),
        "n_groups": len(groups),
        "n_total": n_total,
        "epsilon_squared": float(eps_sq),
    }

    rows = []
    for a, b in itertools.combinations(sorted(groups), 2):
        u, pu = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        rows.append(
            {
                "metric": metric,
                "class_a": a,
                "class_b": b,
                "n_a": len(groups[a]),
                "n_b": len(groups[b]),
                "median_a": float(np.median(groups[a])),
                "median_b": float(np.median(groups[b])),
                "u": float(u),
                "p_raw": float(pu),
                "cliffs_delta": cliffs_delta(groups[a], groups[b]),
            }
        )
    pairwise = pd.DataFrame(rows)
    if not pairwise.empty:
        pairwise["p_adj"] = benjamini_hochberg(pairwise["p_raw"])
        pairwise["significant"] = pairwise["p_adj"] < 0.05
        pairwise = pairwise.sort_values("p_adj").reset_index(drop=True)
    return omnibus, pairwise


# --------------------------------------------------------------------------
# Survival by class
# --------------------------------------------------------------------------


def survival_by_class(
    events: pd.DataFrame, min_group: int = 5
) -> tuple[pd.DataFrame, dict]:
    """Kaplan-Meier dormancy curves per event class + log-rank comparison."""
    if events.empty or "event_class" not in events:
        return pd.DataFrame(), {}

    curves = []
    groups = {}
    for cls, g in events.groupby("event_class"):
        if len(g) < min_group:
            continue
        d = g["duration_months"].to_numpy(dtype=float)
        o = g["observed"].to_numpy(dtype=bool)
        km = survival.kaplan_meier(d, o)
        frame = km.to_frame()
        frame["event_class"] = cls
        frame["n"] = len(g)
        frame["median_survival"] = km.median_survival()
        curves.append(frame)
        groups[cls] = (d, o)

    if not curves:
        return pd.DataFrame(), {}
    return pd.concat(curves, ignore_index=True), survival.logrank_test(groups)


# --------------------------------------------------------------------------
# What predicts persistence
# --------------------------------------------------------------------------

PREDICTORS = (
    "peak_rate",
    "log_peak_rate",
    "editors_year1",
    "refs_at_t0",
    "bytes_at_t0",
    "indeg_first",
    "anon_share_year1",
    "n_months",
)


def assemble_model_frame(
    fits: pd.DataFrame,
    panel: pd.DataFrame,
    snapshots: pd.DataFrame | None = None,
    link_traj: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join decay outcomes to features measurable in the event's first year."""
    if fits.empty:
        return pd.DataFrame()

    year1 = (
        panel[panel["month_index"] < 12]
        .groupby("title")
        .agg(
            editors_year1=("editors", "sum"),
            edits_year1=("edits", "sum"),
            anon_share_year1=("anon_share", "mean"),
        )
        .reset_index()
    )

    df = fits.merge(year1, on="title", how="left")

    if snapshots is not None and len(snapshots):
        t0 = (
            snapshots[snapshots["offset_years"] <= 0.3]
            .sort_values("offset_years")
            .groupby("title")
            .first()
            .reset_index()[["title", "n_refs", "n_bytes", "n_urls"]]
            .rename(
                columns={
                    "n_refs": "refs_at_t0",
                    "n_bytes": "bytes_at_t0",
                    "n_urls": "urls_at_t0",
                }
            )
        )
        df = df.merge(t0, on="title", how="left")

    if link_traj is not None and len(link_traj):
        df = df.merge(link_traj, on="title", how="left")

    df["log_peak_rate"] = np.log1p(df["peak_rate"])
    return df


def persistence_model(
    frame: pd.DataFrame,
    target: str = "empirical_half_life",
    predictors: tuple[str, ...] = PREDICTORS,
    seed: int = 0,
) -> dict:
    """Gradient-boosted model of persistence + permutation importance.

    Cross-validated, and reported against a median-only baseline so that a
    weak R^2 is readable as "these features barely explain persistence" rather
    than mistaken for a modelling failure. Both are interesting findings.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold, cross_val_predict

    cols = [c for c in predictors if c in frame.columns]
    if not cols or target not in frame.columns:
        return {}

    work = frame[[target, *cols, "event_class"]].replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[target])
    if len(work) < 30:
        return {"error": "insufficient_data", "n": len(work)}

    y = np.log1p(work[target].to_numpy(dtype=float))
    # Event class enters as dummies, not as a label encoding: the classes have
    # no natural order and imposing one would let the tree split on an
    # arbitrary alphabetical boundary.
    dummies = pd.get_dummies(work["event_class"], prefix="class", dtype=float)
    design = pd.concat([work[cols].reset_index(drop=True),
                        dummies.reset_index(drop=True)], axis=1)
    cols = list(design.columns)
    x = design.to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(
        max_depth=3, max_iter=300, learning_rate=0.06, random_state=seed
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, x, y, cv=cv)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.median(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    model.fit(x, y)
    imp = permutation_importance(model, x, y, n_repeats=20, random_state=seed)
    importance = (
        pd.DataFrame(
            {
                "feature": cols,
                "importance": imp.importances_mean,
                "std": imp.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Does event class add anything beyond the numeric features? Compare the
    # residual spread across classes.
    resid = y - pred
    class_resid = (
        pd.DataFrame({"event_class": work["event_class"].to_numpy(), "resid": resid})
        .groupby("event_class")["resid"]
        .agg(["mean", "count"])
        .reset_index()
    )

    return {
        "n": len(work),
        "target": target,
        "cv_r2": float(r2),
        "importance": importance,
        "class_residuals": class_resid,
        "features": cols,
    }


# --------------------------------------------------------------------------
# Resistance index
# --------------------------------------------------------------------------

RESISTANCE_COMPONENTS = {
    # component: (column, higher_is_more_resistant)
    "half_life": ("empirical_half_life", True),
    "tail_activity": ("tail_rate_5y", True),
    "burst_concentration": ("share_year1", False),
    "temporal_evenness": ("gini_time", False),
    "network_retention": ("indeg_retention", True),
}
# Deliberately NOT a component: ``floor_share_of_peak``. The maintenance floor
# is a real and interesting parameter in aggregate -- a power law with a floor
# is the winning functional form for a plurality of articles -- but validation
# against the simulator shows the *per-article* floor estimate recovers its
# true value at only rho ~= 0.2, because the floor is a few percent of peak and
# monthly counts that small are dominated by Poisson noise. Including it would
# add noise to the index while looking like extra evidence.


def resistance_index(
    frame: pd.DataFrame, components: dict | None = None
) -> pd.DataFrame:
    """Composite decay-resistance score.

    Each component is converted to a within-corpus percentile rank (robust to
    the heavy tails and to the different units involved), sign-corrected so
    that higher always means "more resistant", then averaged over whichever
    components are available for that article. Percentile ranks rather than
    z-scores because several components are bounded or infinite-valued.
    """
    components = components or RESISTANCE_COMPONENTS
    if frame.empty:
        return pd.DataFrame()

    out = frame[["title"]].copy()
    if "event_class" in frame:
        out["event_class"] = frame["event_class"]

    used = []
    for name, (col, higher_better) in components.items():
        if col not in frame.columns:
            continue
        values = frame[col].replace([np.inf], np.nan).astype(float)
        if values.notna().sum() < 5:
            continue
        # Infinity (never halved) is maximal resistance, not missing data.
        pct = values.rank(pct=True, na_option="keep")
        if np.isinf(frame[col].to_numpy(dtype=float)).any():
            pct = pct.where(~np.isinf(frame[col].to_numpy(dtype=float)), 1.0)
        out[f"pct_{name}"] = pct if higher_better else 1 - pct
        used.append(f"pct_{name}")

    if not used:
        return out
    out["resistance_index"] = out[used].mean(axis=1, skipna=True)
    out["n_components"] = out[used].notna().sum(axis=1)
    return out.sort_values("resistance_index", ascending=False).reset_index(drop=True)
