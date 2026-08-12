"""Figures.

Design rules applied throughout (and worth stating, because they are the
difference between a chart that is read and one that is merely produced):

- **One y-axis per panel, always.** Where two measures of different scale need
  comparing (dead links vs. archive coverage), they get stacked panels sharing
  an x-axis, never twin axes.
- **Colour is assigned by role, in fixed slot order, never cycled.** At most
  three categorical hues appear in any panel; when there are eleven event
  classes, the chart highlights the extremes and greys the rest rather than
  inventing eleven colours nobody can tell apart.
- **Marks are thin, grids are recessive, and text never wears the series
  colour** -- a coloured mark next to a label carries identity instead.
- Every categorical hue used here passes the adjacent-pair and all-pairs CVD
  separation gates on the light surface. Aqua sits below 3:1 contrast, so it is
  always shipped with a visible direct label.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# --- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e5e4e0"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua
SEQ = ("#cfe0f5", "#9dc0ea", "#6a9fdf", "#2a78d6", "#1a4c8a")  # single-hue ramp

FIGSIZE = (8.2, 4.8)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.dpi": 130,
        }
    )


def _titles(ax, title: str, subtitle: str = "") -> None:
    """Draw title and subtitle as explicit axes-fraction text.

    ``set_title`` plus a hand-placed subtitle collide as soon as the figure is
    resized, because one is positioned in points and the other in axes
    fractions. Placing both in the same coordinate system keeps the gap fixed.
    """
    ax.set_title("")
    # Offsets in points, not axes fractions: a 12-inch-tall panel and a
    # 3-inch-tall one must show the same gap under the title.
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 30 if subtitle else 10), textcoords="offset points",
                fontsize=12, fontweight="600", color=INK, va="bottom", ha="left")
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 12), textcoords="offset points",
                    fontsize=9.5, color=INK_2, va="bottom", ha="left")


def _finish(fig, ax, title: str, subtitle: str = "", source: str = "") -> None:
    _titles(ax, title, subtitle)
    if source:
        fig.text(0.01, -0.02, source, fontsize=8, color=INK_MUTED, ha="left")
    ax.set_axisbelow(True)
    fig.tight_layout()


def _edge_label(ax, x, y, text, color, pad_frac: float = 0.02):
    """Direct-label a series at its right end and reserve room for the text.

    Direct labels beat a legend box, but only if they fit. Matplotlib will not
    widen the axes to accommodate text, so we do it explicitly rather than
    letting the label run off the canvas.
    """
    lo, hi = ax.get_xlim()
    span = hi - lo
    ax.text(x + pad_frac * span, y, text, color=color, fontsize=9,
            va="center", ha="left", clip_on=False)
    needed = x + pad_frac * span + 0.30 * span
    if needed > hi:
        ax.set_xlim(lo, needed)


def _save(fig, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 1. The forgetting curve
# --------------------------------------------------------------------------


def plot_mean_curve(panel: pd.DataFrame, outdir: Path, max_months: int = 180) -> Path:
    """Corpus-average attention against months since the event, log-log.

    Each article is normalised by its own peak before averaging, so a single
    enormous event cannot define the corpus curve. Log-log axes because a
    straight line there *is* the test for power-law decay -- the shape of the
    data on these axes is the finding, not decoration.
    """
    apply_style()
    norm = []
    for _, g in panel.groupby("title"):
        g = g.sort_values("month_index")
        y = g["edits"].to_numpy(dtype=float)
        smooth = pd.Series(y).rolling(3, center=True, min_periods=1).mean()
        peak = smooth.max()
        if peak <= 0 or len(y) < 36:
            continue
        # Quantiles are taken over the *smoothed* series. Raw monthly counts
        # are zero-inflated in the tail, so the raw 25th percentile is exactly
        # zero for most late months and the band would collapse to a row of
        # spikes on a log axis -- an artefact of discreteness, not a finding.
        norm.append((smooth[:max_months] / peak).reindex(range(max_months)))
    if not norm:
        raise ValueError("no usable series")
    mat = pd.concat(norm, axis=1)
    median = mat.median(axis=1)
    q25, q75 = mat.quantile(0.25, axis=1), mat.quantile(0.75, axis=1)

    # A log axis cannot render a band that touches zero, and the quartile of a
    # corpus where a quarter of articles are silent in a given month is exactly
    # zero. Clip the band to the smallest non-zero rate we can actually show,
    # rather than letting it spike to the bottom of the canvas.
    floor = max(float(median[median > 0].min()) / 4, 1e-4)
    q25 = q25.clip(lower=floor)

    t = np.arange(max_months) + 1
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(t, q25, q75, color=SERIES[0], alpha=0.16, linewidth=0)
    ax.plot(t, median, color=SERIES[0], linewidth=2.0, solid_capstyle="round")

    mask = (median > 0) & np.isfinite(median)
    if mask.sum() > 10:
        slope, intercept = np.polyfit(np.log(t[mask]), np.log(median[mask]), 1)
        ax.plot(
            t, np.exp(intercept) * t**slope, color=INK_2, linewidth=1.2,
            linestyle=(0, (4, 3)),
        )
        ax.annotate(
            f"power law, slope {slope:.2f}",
            xy=(t[-1], np.exp(intercept) * t[-1] ** slope),
            xytext=(-4, -16), textcoords="offset points",
            color=INK_2, fontsize=9, ha="right",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor * 0.8)
    ax.set_xlabel("Months since event (log scale)")
    ax.set_ylabel("Edit rate, share of the article's own peak")
    ax.annotate("median article", xy=(t[6], median.iloc[6]),
                xytext=(6, 10), textcoords="offset points",
                color=INK, fontsize=9.5)
    _finish(
        fig, ax,
        "Collective attention decays as a power law, not exponentially",
        "Median across all event articles, each normalised to its own peak. Shaded band: interquartile range.",
        f"n = {mat.shape[1]} articles",
    )
    return _save(fig, outdir, "01_mean_forgetting_curve")


# --------------------------------------------------------------------------
# 2. Model selection
# --------------------------------------------------------------------------


def plot_model_weights(fits: pd.DataFrame, outdir: Path) -> Path:
    """Average Akaike weight per functional form: magnitude, so one hue ramp."""
    apply_style()
    cols = [c for c in fits.columns if c.startswith("weight_")]
    means = fits[cols].mean().sort_values()
    labels = [c.replace("weight_", "").replace("_", " ") for c in means.index]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ramp = list(SEQ[-len(means):]) if len(means) <= len(SEQ) else [SEQ[-1]] * len(means)
    bars = ax.barh(labels, means.to_numpy(), color=ramp, height=0.62)
    for bar, value in zip(bars, means.to_numpy()):
        ax.text(
            bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}", va="center", fontsize=9.5, color=INK,
        )
    ax.set_xlim(0, min(1.0, means.max() * 1.28))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Mean Akaike weight (probability this form is best)")
    ax.grid(axis="y", visible=False)
    _finish(
        fig, ax,
        "Which decay law fits an event article's history?",
        "Averaged over articles; weights within an article sum to 100%.",
        f"n = {len(fits)} articles",
    )
    return _save(fig, outdir, "02_model_weights")


# --------------------------------------------------------------------------
# 3. Half-life by event class
# --------------------------------------------------------------------------


def plot_halflife_by_class(stats_df: pd.DataFrame, outdir: Path,
                           metric: str = "empirical_half_life") -> Path:
    """Dot-and-interval plot: one measure across classes, so one colour."""
    apply_style()
    d = stats_df[stats_df["metric"] == metric].dropna(subset=["median"])
    if d.empty:
        raise ValueError(f"no rows for metric {metric}")
    d = d.sort_values("median")
    labels = [c.replace("_", " ") for c in d["event_class"]]
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(d) + 2.0))
    ax.hlines(y, d["ci_low"], d["ci_high"], color=SERIES[0], alpha=0.35, linewidth=2.4)
    ax.scatter(d["median"], y, s=64, color=SERIES[0], zorder=3,
               edgecolor=SURFACE, linewidth=2)
    for yi, (median, n) in enumerate(zip(d["median"], d["n"])):
        ax.text(median, yi + 0.30, f"{median:.0f} mo  (n={n})",
                fontsize=8.6, color=INK_2, ha="center")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Months from peak until attention halves")
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.06)
    _finish(
        fig, ax,
        "Some kinds of events are forgotten far faster than others",
        "Median half-life with 95% bootstrap confidence interval.",
        "Bootstrap: 2,000 resamples of the article-level medians",
    )
    return _save(fig, outdir, "03_halflife_by_class")


# --------------------------------------------------------------------------
# 4. Dormancy survival
# --------------------------------------------------------------------------


def plot_survival(curves: pd.DataFrame, outdir: Path, highlight: int = 2) -> Path:
    """Kaplan-Meier curves: highlight the extremes, grey the middle.

    Eleven classes cannot each carry a distinguishable hue. Highlighting the
    slowest and fastest and rendering the rest as context keeps the comparison
    legible and keeps us inside the three-slot colour budget.
    """
    apply_style()
    if curves.empty:
        raise ValueError("no survival curves")

    medians = (
        curves.groupby("event_class")["median_survival"].first().sort_values()
    )
    finite = medians.replace([np.inf], np.nan).dropna()
    ordered = list(finite.index) + [c for c in medians.index if c not in finite.index]
    fastest = ordered[:highlight]
    slowest = ordered[-highlight:]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels: list[tuple[float, float, str, str]] = []
    for cls, g in curves.groupby("event_class"):
        g = g.sort_values("time")
        if cls in fastest:
            color, lw, alpha, z = SERIES[1], 2.0, 1.0, 3
        elif cls in slowest:
            color, lw, alpha, z = SERIES[0], 2.0, 1.0, 3
        else:
            color, lw, alpha, z = INK_MUTED, 1.0, 0.35, 2
        ax.step(g["time"], g["survival"], where="post",
                color=color, linewidth=lw, alpha=alpha, zorder=z)
        if cls in fastest or cls in slowest:
            labels.append(
                (float(g["time"].iloc[-1]), float(g["survival"].iloc[-1]),
                 cls.replace("_", " "), color)
            )

    ax.axhline(0.5, color=INK_MUTED, linewidth=0.9, linestyle=(0, (3, 3)))
    ax.text(2, 0.515, "half the articles dormant", fontsize=8.6, color=INK_MUTED)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Months since the article's attention peak")
    ax.set_ylabel("Share still actively edited")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Labels go ON the curve at a readable height rather than at its right
    # end: the fast-decaying classes all terminate near zero survival and
    # stacking their names there is illegible regardless of nudging.
    x_max = float(curves["time"].max())
    ax.set_xlim(left=-4, right=x_max * 1.02)
    for group, color in ((fastest, SERIES[1]), (slowest, SERIES[0])):
        for i, cls in enumerate(group):
            g = curves[curves["event_class"] == cls].sort_values("time")
            if g.empty:
                continue
            # Anchor each label at a different height on its own curve, so two
            # classes that decay at similar rates do not land on the same spot.
            threshold = 0.68 - 0.20 * i
            target = g[g["survival"] <= threshold]
            row = target.iloc[0] if len(target) else g.iloc[-1]
            ax.annotate(
                cls.replace("_", " "),
                xy=(float(row["time"]), float(row["survival"])),
                xytext=(9, 11 if color == SERIES[0] else -15),
                textcoords="offset points",
                color=color, fontsize=9, zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE,
                          edgecolor="none", alpha=0.9),
            )
    _finish(
        fig, ax,
        "Time until an event article goes dormant",
        "Kaplan-Meier estimate; dormancy = trailing-year edit rate below 5% of peak. Grey lines: remaining classes.",
        "Right-censored at each article's observation horizon",
    )
    return _save(fig, outdir, "04_dormancy_survival")


# --------------------------------------------------------------------------
# 5. Link rot
# --------------------------------------------------------------------------


def plot_link_rot(rot: pd.DataFrame, outdir: Path) -> Path:
    """Two measures, two stacked panels, one shared x-axis. Never twin axes."""
    apply_style()
    if rot.empty:
        raise ValueError("no rot trajectories")
    rot = rot.sort_values("offset_years")
    x = rot["offset_years"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 6.0), sharex=True, gridspec_kw={"hspace": 0.28}
    )
    ax1.plot(x, rot["median_dead_per_100"], color=SERIES[1], linewidth=2.0,
             marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.6)
    ax1.set_ylabel("Dead links per 100 citations")
    _edge_label(ax1, x.iloc[-1], rot["median_dead_per_100"].iloc[-1],
                "citations rot", SERIES[1])

    ax2.plot(x, rot["mean_archive_pct"], color=SERIES[2], linewidth=2.0,
             marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.6)
    ax2.set_ylabel("Citations with an archive copy (%)")
    ax2.set_xlabel("Years since event")
    # Aqua is below the 3:1 contrast floor, so it always ships with a label.
    _edge_label(ax2, x.iloc[-1], rot["mean_archive_pct"].iloc[-1],
                "editors repair them", SERIES[2])

    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
    ax1.set_xlim(ax2.get_xlim())
    _titles(ax1, "The evidence underneath an article decays on its own schedule",
            "Median dead-link density and mean archive coverage across snapshot ages.")
    # Explicit margins rather than tight_layout: the two panels already carry a
    # hand-set hspace, and tight_layout warns when asked to re-solve that.
    fig.subplots_adjust(top=0.86, bottom=0.09, left=0.10, right=0.97, hspace=0.28)
    return _save(fig, outdir, "05_link_rot")


# --------------------------------------------------------------------------
# 6. Network edge survival
# --------------------------------------------------------------------------


def plot_edge_survival(edges: pd.DataFrame, outdir: Path) -> Path:
    apply_style()
    if edges.empty:
        raise ValueError("no edge survival data")
    edges = edges.sort_values("offset_years")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(edges["offset_years"], edges["survival_rate"], color=SERIES[0],
            linewidth=2.0, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.8)
    ax.plot(edges["offset_years"], edges["growth_ratio"].clip(upper=3),
            color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.set_xlim(left=-0.4, right=float(edges["offset_years"].max()))
    _edge_label(ax, edges["offset_years"].iloc[-1], edges["survival_rate"].iloc[-1],
                "original links surviving", SERIES[0])
    _edge_label(ax, edges["offset_years"].iloc[-1],
                min(edges["growth_ratio"].iloc[-1], 3),
                "total links vs. baseline", INK_2)
    ax.set_xlabel("Years since event")
    ax.set_ylabel("Share of baseline links")
    ax.set_ylim(bottom=0)
    _finish(
        fig, ax,
        "Events stop being referenced before they stop being edited",
        "Links between event articles, measured against the earliest snapshot of each article.",
        f"Baseline edges: {int(edges['n_baseline_edges'].iloc[0]):,}",
    )
    return _save(fig, outdir, "06_edge_survival")


# --------------------------------------------------------------------------
# 7. Estimator validation
# --------------------------------------------------------------------------


def plot_recovery(fits: pd.DataFrame, truth: pd.DataFrame, outdir: Path) -> Path:
    """Fitted vs. true parameters on simulated data -- the estimator's receipt."""
    apply_style()
    merged = fits.merge(truth, on="title", how="inner")
    pairs = [
        ("power_exponent", "true_exponent", "Decay exponent b"),
        ("floor_share_of_peak", "true_floor_share", "Maintenance floor (share of peak)"),
        ("empirical_half_life", "true_half_life", "Half-life (months)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    for ax, (fit_col, true_col, label) in zip(axes, pairs):
        if fit_col not in merged or true_col not in merged:
            ax.axis("off")
            continue
        x = merged[true_col].replace([np.inf, -np.inf], np.nan)
        y = merged[fit_col].replace([np.inf, -np.inf], np.nan)
        mask = x.notna() & y.notna()
        ax.scatter(x[mask], y[mask], s=22, color=SERIES[0], alpha=0.55,
                   edgecolor=SURFACE, linewidth=0.6)
        if mask.sum() > 2:
            lim = [
                float(min(x[mask].min(), y[mask].min())),
                float(max(x[mask].max(), y[mask].max())),
            ]
            ax.plot(lim, lim, color=INK_2, linewidth=1.1, linestyle=(0, (4, 3)))
            r = np.corrcoef(x[mask], y[mask])[0, 1]
            ax.text(0.04, 0.93, f"r = {r:.2f}", transform=ax.transAxes,
                    fontsize=9.5, color=INK, va="top")
        ax.set_xlabel(f"True {label.lower()}")
        ax.set_ylabel(f"Recovered")
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_axisbelow(True)
    fig.tight_layout()
    # suptitle after tight_layout, then make room by hand: tight_layout does
    # not account for a figure-level title and warns if asked to.
    fig.subplots_adjust(top=0.80)
    fig.suptitle(
        "Estimator validation: recovering known parameters from simulated histories",
        x=0.006, y=0.97, ha="left", fontsize=12, fontweight="600", color=INK,
    )
    return _save(fig, outdir, "07_estimator_recovery")


# --------------------------------------------------------------------------
# 8. Resistance ranking
# --------------------------------------------------------------------------


def plot_resistance(resistance: pd.DataFrame, outdir: Path, top_n: int = 14) -> Path:
    apply_style()
    if resistance.empty:
        raise ValueError("no resistance index")
    d = resistance.dropna(subset=["resistance_index"])
    head = d.head(top_n).iloc[::-1]
    labels = [t if len(t) <= 46 else t[:44] + "..." for t in head["title"]]

    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(head) + 1.9))
    ax.barh(labels, head["resistance_index"], color=SERIES[0], height=0.62)
    for i, v in enumerate(head["resistance_index"]):
        ax.text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=8.6, color=INK)
    ax.set_xlim(0, min(1.0, head["resistance_index"].max() * 1.16))
    ax.set_xlabel("Decay-resistance index (0 = forgotten fastest, 1 = most persistent)")
    ax.grid(axis="y", visible=False)
    _finish(
        fig, ax,
        "The events the encyclopedia keeps working on",
        "Composite percentile rank across half-life, tail activity, burst concentration, evenness and link retention.",
        f"Top {len(head)} of {len(d)} articles",
    )
    return _save(fig, outdir, "08_resistance_index")
