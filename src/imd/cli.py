"""Command-line entry point.

Stages are separate subcommands and each writes a parquet artifact, so a run
can be resumed, inspected or partially re-executed. ``imd all`` chains them.

Typical real run::

    export IMD_CONTACT="you@example.com"
    imd corpus            # ~10 min
    imd collect           # hours; resumable, cached
    imd analyze
    imd figures
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import analysis, collect, config, decay, network, simulate, survival, viz
from .wikiapi import WikiClient

log = logging.getLogger("imd")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _read(key: str) -> pd.DataFrame:
    path = config.data_path(key)
    if not path.exists():
        raise SystemExit(
            f"missing artifact {path.name}. Run the earlier stage first "
            f"(or `imd simulate` for an offline dataset)."
        )
    return pd.read_parquet(path)


def _write(df: pd.DataFrame, key: str) -> None:
    config.ensure_dirs()
    path = config.data_path(key)
    # Object columns of mixed type break the parquet writer; coerce them.
    df.to_parquet(path, index=False)
    log.info("wrote %-22s %6d rows  ->  %s", key, len(df), path.name)


def _maybe_read(key: str) -> pd.DataFrame | None:
    path = config.data_path(key)
    return pd.read_parquet(path) if path.exists() else None


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def cmd_corpus(args) -> None:
    from . import corpus as corpus_mod

    settings = config.Settings(
        year_start=args.year_start,
        year_end=args.year_end,
        per_category_year=args.per_category,
    )
    client = WikiClient(settings)
    df, dropped = corpus_mod.build_corpus(client, settings)
    if df.empty:
        raise SystemExit("corpus is empty -- check network access and category names")
    _write(df, "corpus")
    if len(dropped):
        _write(dropped, "corpus_dropped")
        log.info("dropped: %s", df.attrs.get("dropped_counts"))
    print(df["event_class"].value_counts().to_string())


def cmd_collect(args) -> None:
    settings = config.DEFAULT_SETTINGS
    corpus = _read("corpus")
    client = WikiClient(settings)

    revisions = collect.collect_revisions(client, corpus, _maybe_read("revisions"))
    _write(revisions, "revisions")

    # Stub articles can only be recognised once their history is in hand, so
    # the activity threshold is applied here rather than at corpus time. The
    # excluded rows are written out so the exclusion stays auditable.
    corpus, inactive = collect.filter_low_activity(corpus, revisions, settings)
    if corpus.empty:
        raise SystemExit(
            f"every article fell below min_revisions={settings.min_revisions}"
        )
    if len(inactive):
        prior = _maybe_read("corpus_dropped")
        if prior is not None and len(prior):
            inactive = pd.concat([prior, inactive], ignore_index=True)
            inactive = inactive.drop_duplicates(subset="title", keep="last")
        _write(inactive, "corpus_dropped")
    _write(corpus, "corpus")

    picks = collect.pick_snapshot_revisions(revisions, corpus, settings)
    snapshots, links = collect.collect_snapshots(client, picks, _maybe_read("snapshots"))
    _write(snapshots, "snapshots")
    _write(links, "snapshot_links")

    if not args.skip_pageviews:
        pv = collect.collect_pageviews(client, corpus)
        _write(pv, "pageviews")

    panel = collect.build_panel(corpus, revisions, _maybe_read("pageviews"), settings)
    _write(panel, "panel")
    log.info("cache: %s", client.stats)


def cmd_simulate(args) -> None:
    """Generate an offline dataset with known ground truth."""
    config.ensure_dirs()
    data = simulate.simulate_corpus(n_articles=args.n, seed=args.seed)
    for key in ("corpus", "revisions", "snapshots", "snapshot_links"):
        _write(data[key], key)
    data["ground_truth"].to_parquet(config.DATA_DIR / "ground_truth.parquet", index=False)
    panel = collect.build_panel(data["corpus"], data["revisions"])
    _write(panel, "panel")
    print(f"simulated {len(data['corpus'])} articles, {len(data['revisions']):,} revisions")


def cmd_analyze(args) -> None:
    settings = config.DEFAULT_SETTINGS
    corpus = _read("corpus")
    panel = _read("panel")

    fits = decay.fit_panel(
        panel,
        peak_window=settings.peak_window_months,
        min_months=settings.min_months_observed,
        seed=settings.random_seed,
    )
    if fits.empty:
        raise SystemExit("no articles had enough history to fit")
    _write(fits, "fits")

    snapshots = _maybe_read("snapshots")
    links = _maybe_read("snapshot_links")

    link_traj = pd.DataFrame()
    if links is not None and len(links):
        graphs = network.build_snapshot_graphs(links, set(corpus["title"]))
        metrics = network.network_metrics(graphs)
        _write(metrics, "network")
        _write(network.edge_survival(graphs), "edge_survival")
        link_traj = network.per_article_link_trajectory(metrics)

    cat = analysis.category_summary(
        fits, n_boot=settings.bootstrap_iterations, seed=settings.random_seed
    )
    _write(cat, "category_stats")

    events = survival.dormancy_events(
        panel,
        threshold=settings.dormancy_threshold,
        window=settings.dormancy_window_months,
        peak_window=settings.peak_window_months,
    )
    curves, logrank = analysis.survival_by_class(events)
    if not curves.empty:
        _write(curves, "survival")

    frame = analysis.assemble_model_frame(fits, panel, snapshots, link_traj)
    resistance = analysis.resistance_index(frame)
    _write(resistance, "resistance")

    report: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "n_articles_fitted": len(fits),
                    "settings": asdict(settings)}

    for metric in ("empirical_half_life", "share_year1", "floor_share_of_peak"):
        omnibus, pairwise = analysis.omnibus_and_pairwise(fits, metric)
        if omnibus:
            report.setdefault("omnibus", {})[metric] = omnibus
        if not pairwise.empty:
            pairwise.to_csv(
                config.OUTPUT_DIR / f"pairwise_{metric}.csv", index=False
            )
            report.setdefault("n_significant_pairs", {})[metric] = int(
                pairwise["significant"].sum()
            )
    if logrank:
        report["logrank"] = {k: v for k, v in logrank.items()
                             if k in ("statistic", "p_value", "df")}

    report["best_model_share"] = fits["best_model"].value_counts(normalize=True).to_dict()
    report["median_half_life_months"] = float(
        fits["empirical_half_life"].median(skipna=True)
    )

    model = analysis.persistence_model(frame)
    if model and "importance" in model:
        model["importance"].to_csv(config.OUTPUT_DIR / "feature_importance.csv", index=False)
        report["persistence_model"] = {
            "n": model["n"], "cv_r2": model["cv_r2"],
            "top_features": model["importance"].head(5)["feature"].tolist(),
        }

    truth_path = config.DATA_DIR / "ground_truth.parquet"
    if truth_path.exists():
        report["estimator_recovery"] = simulate.validate_recovery(
            fits, pd.read_parquet(truth_path)
        )

    out = config.OUTPUT_DIR / "findings.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote %s", out)
    print(json.dumps({k: report[k] for k in
                      ("n_articles_fitted", "median_half_life_months", "best_model_share")
                      if k in report}, indent=2, default=str))


def cmd_figures(args) -> None:
    config.ensure_dirs()
    outdir = config.FIGURE_DIR
    made: list[str] = []
    failed: list[str] = []

    def attempt(name, fn, *a, **kw):
        try:
            path = fn(*a, **kw)
            made.append(path.name)
        except Exception as exc:
            failed.append(f"{name}: {exc}")

    panel = _maybe_read("panel")
    fits = _maybe_read("fits")

    if panel is not None:
        attempt("mean_curve", viz.plot_mean_curve, panel, outdir)
    if fits is not None:
        attempt("model_weights", viz.plot_model_weights, fits, outdir)
    cat = _maybe_read("category_stats")
    if cat is not None:
        attempt("halflife", viz.plot_halflife_by_class, cat, outdir)
    curves = _maybe_read("survival")
    if curves is not None:
        attempt("survival", viz.plot_survival, curves, outdir)
    snapshots = _maybe_read("snapshots")
    if snapshots is not None:
        attempt("link_rot", viz.plot_link_rot, network.rot_trajectories(snapshots), outdir)
    edges = _maybe_read("edge_survival")
    if edges is not None:
        attempt("edge_survival", viz.plot_edge_survival, edges, outdir)
    truth_path = config.DATA_DIR / "ground_truth.parquet"
    if fits is not None and truth_path.exists():
        attempt("recovery", viz.plot_recovery, fits, pd.read_parquet(truth_path), outdir)
    resistance = _maybe_read("resistance")
    if resistance is not None:
        attempt("resistance", viz.plot_resistance, resistance, outdir)

    print(f"wrote {len(made)} figures to {outdir}")
    for name in made:
        print(f"  {name}")
    for message in failed:
        print(f"  skipped {message}", file=sys.stderr)


def cmd_all(args) -> None:
    if args.simulated:
        cmd_simulate(args)
    else:
        cmd_corpus(args)
        cmd_collect(args)
    cmd_analyze(args)
    cmd_figures(args)


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="imd", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("corpus", help="build the event corpus from Wikipedia categories")
    c.add_argument("--year-start", type=int, default=config.DEFAULT_SETTINGS.year_start)
    c.add_argument("--year-end", type=int, default=config.DEFAULT_SETTINGS.year_end)
    c.add_argument("--per-category", type=int,
                   default=config.DEFAULT_SETTINGS.per_category_year)
    c.set_defaults(func=cmd_corpus)

    c = sub.add_parser("collect", help="fetch revisions, snapshots and pageviews")
    c.add_argument("--skip-pageviews", action="store_true")
    c.set_defaults(func=cmd_collect)

    c = sub.add_parser("simulate", help="generate an offline dataset with ground truth")
    c.add_argument("-n", type=int, default=240)
    c.add_argument("--seed", type=int, default=config.DEFAULT_SETTINGS.random_seed)
    c.set_defaults(func=cmd_simulate)

    c = sub.add_parser("analyze", help="fit decay models and run the statistics")
    c.set_defaults(func=cmd_analyze)

    c = sub.add_parser("figures", help="render every figure")
    c.set_defaults(func=cmd_figures)

    c = sub.add_parser("all", help="run the whole pipeline")
    c.add_argument("--simulated", action="store_true",
                   help="use the simulator instead of the live API")
    c.add_argument("-n", type=int, default=240)
    c.add_argument("--seed", type=int, default=config.DEFAULT_SETTINGS.random_seed)
    c.add_argument("--year-start", type=int, default=config.DEFAULT_SETTINGS.year_start)
    c.add_argument("--year-end", type=int, default=config.DEFAULT_SETTINGS.year_end)
    c.add_argument("--per-category", type=int,
                   default=config.DEFAULT_SETTINGS.per_category_year)
    c.add_argument("--skip-pageviews", action="store_true")
    c.set_defaults(func=cmd_all)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    config.ensure_dirs()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
