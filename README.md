# Institutional Memory Decay

**How fast does the world forget? Measuring the decay of collective memory through Wikipedia's edit histories and the citation network between event articles.**

Every event that matters enough to get a Wikipedia article gets a burst of
attention: hundreds of editors, thousands of revisions, a wall of citations to
breaking news. Then the attention drains away. This project measures the shape
of that draining — the **forgetting curve** for world events — and asks which
kinds of events resist it.

It combines three things that decay on different schedules:

| Layer | Signal | What it measures |
|---|---|---|
| **Attention** | monthly human edit rate | whether anyone is still working on it |
| **Substance** | article size, citation count | whether the record is still growing |
| **Structure** | inbound links from other event articles | whether the event is still used to explain other things |
| **Evidence** | dead-link tags, archive coverage | whether the sources still exist |

---

## What it found

`outputs/` holds the results of a **live run against English Wikipedia**: 690
event articles spanning 2001–2023, 239,197 revisions, 5,165 wikitext snapshots,
6,900 API requests, zero errors. The written analysis is in
[`report/`](report/institutional-memory-decay.md).

- **Forgetting is not exponential.** Across 688 fitted articles the pure
  exponential takes under 8% of the Akaike weight, against 46% for a power law
  decaying onto a maintenance floor. Collective forgetting has no characteristic
  timescale — though stretched exponential takes a third of the weight, so
  "heavy-tailed" is better supported than "power law" specifically.
- **Event type matters less than expected, and mostly through elections.**
  Election articles hold attention 2.0× longer than terrorism or transport
  accidents (median half-life 2.6 vs 1.3 months), but eight of eleven classes
  sit within 0.15 months of each other. Kruskal–Wallis H = 85.7,
  p = 3.8 × 10⁻¹⁴, but ε² = 0.115.
- **Evidence rots faster than it is repaired.** Mean dead-link density climbs
  monotonically from ~0 to 11 per 100 URLs over sixteen years while archive
  coverage rises only from 2.5% to 21%. The median stays at zero for eight
  years and then climbs steeply — rot is a threshold articles cross one at a
  time, not a drizzle across the corpus.
- **Structure outlives attention.** The median article has zero human edits by
  month 12, yet 75% of its inbound links from other event articles are still
  present three years out, and total inbound links are still growing.

Two limitations are load-bearing enough to state up front: the monthly panel is
**too coarse to resolve the early curve** (78% of articles peak in month 0, and
two-thirds have half-lives under 1.5 months, which is the estimator's floor),
and the internal-link layer rests on only **80 baseline edges**, because links
between articles that are both in a sample of Wikipedia are rare by
construction. Both are discussed in the report.

---

## Headline design decisions

Most of the interesting work in this project is in getting the measurement
right, not in getting a number out. Four choices do the heavy lifting:

**Poisson MLE, not least squares.** The observable is a count that spans four
orders of magnitude within one article's history. OLS on raw counts is
dominated entirely by the peak month; OLS on log counts is undefined for the
many months with zero edits. Poisson likelihood handles both ends and makes
AIC comparisons across functional forms legitimate.

**Four candidate decay laws, compared per article.** Exponential (constant
hazard of forgetting), power law (scale-free), stretched exponential, and
**power law onto a floor** — decay to a non-zero asymptote representing the
permanent trickle of maintenance an article settles into. Akaike weights are
reported, not just the argmin, so it is visible how often the winner wins by a
nose.

**Survival analysis for the "which events resist" question.** An article about
a 2019 event that is *still* being edited has not "survived five years and then
gone dormant" — we simply have not observed its dormancy yet. Treating that as
either a long life or as missing data biases the comparison. Kaplan–Meier with
right-censoring handles it correctly. Implemented from scratch in numpy alongside the
log-rank test, so the repository has no compiled dependencies and both
estimators are auditable end to end.

**Ground-truth validation before any claim.** Every headline number is a
fitted parameter, and a fitted parameter is only worth reporting if the fitting
code can recover a parameter it was given. `imd.simulate` generates edit
histories from known decay parameters; `validate_recovery` checks the
estimators get them back. This is not a formality — it caught two real bugs
(see below).

---

## What validation caught

Both of these were found by the simulator, not by reading the code, and both
materially changed the results.

**1. Resurgences were destroying the fits.** Anniversary coverage, a court
verdict, a later related event — these add a burst of editing on top of the
ongoing decay. Left in, a single large bump drags the fitted exponent toward
zero and makes a fast-fading event look persistent. Fitting now runs twice,
masking months whose Pearson residual exceeds +3 on the **high side only**
(a quiet month is signal, never an outlier). Exponent recovery went from
Spearman **0.27 → 0.76**.

**2. Peak detection was latching onto those resurgences.** An unconstrained
argmax put the "attention peak" of some articles 33 months after the event, so
"months since peak" started counting from the resurgence and discarded the
entire original decay. The peak search is now confined to the first year, where
an event's peak is by construction.

Recovery on 286 simulated articles after both fixes (this is the one place
synthetic data is used on purpose — a parameter can only be checked for
recovery against a value that is known. The full validation run is preserved in
`outputs/validation_synthetic/`):

| Parameter | Pearson r | Spearman ρ | Median bias |
|---|---|---|---|
| Decay exponent *b* | 0.76 | 0.76 | −0.30 |
| Half-life (empirical) | 0.74 | 0.76 | −0.67 mo |
| Half-life (fitted) | 0.35 | 0.80 | −0.84 mo |
| Maintenance floor share | 0.36 | **0.18** | +0.00 |

The last row is a **negative result that is reported rather than buried**: the
per-article floor estimate does not recover reliably, because the floor is a
few percent of peak and monthly counts that small are dominated by Poisson
noise. The floor is therefore used only in aggregate (as a model-selection
result), and is deliberately **excluded** from the composite resistance index —
including it would have added noise while looking like extra evidence.

---

## Quick start

```bash
pip install -e .

# Offline: generate a synthetic corpus with known ground truth and run
# the entire pipeline end to end. No network needed. ~1 minute.
imd all --simulated -n 300
```

### Running it on real Wikipedia data

```bash
export IMD_CONTACT="you@example.com"   # Wikimedia API etiquette wants a contact

imd corpus      # walk the category taxonomy, resolve event dates   (~8 min)
imd collect     # revisions, wikitext snapshots, pageviews          (~35 min)
imd analyze     # fit decay models, run the statistics              (~6 min)
imd figures     # render every figure
```

Wall-clock times are from the run in `outputs/`, at the default settings
(2004–2022, four articles per category-year). They scale with corpus size; the
dominating cost is one wikitext fetch per article per snapshot offset.

`collect` is the long stage. It is **resumable and cached**: every API response
is stored in a compressed SQLite blob keyed by the request, so an interrupted
run costs nothing to restart, and — more importantly — the analysis becomes
reproducible from a fixed snapshot of Wikipedia rather than from "whatever the
site looked like when you ran it". Requests are rate-limited by a token bucket
and retried with exponential backoff, honouring `Retry-After` and MediaWiki's
in-band `maxlag` signal.

---

## How the corpus is built

Articles are drawn from year-scoped Wikipedia categories across 11 event
classes (natural disasters, transport accidents, armed conflicts, terrorism,
elections, political scandals, science/tech, epidemics, sport, culture/awards,
economy).

The load-bearing step is **resolving *t*₀**, the date the event happened. Every
curve here is measured on an axis of "months since event", so a wrong *t*₀ does
not add noise — it smears the peak and biases the fitted half-life upward.
Resolution order:

1. **Wikidata P585** (point in time) / P580 (start time) on the article's item,
   **at month precision or finer**. A year-precision claim parses to January
   1st, which is a placeholder rather than a date — it displaces *t*₀ by up to
   eleven months and always in the same direction, which inflates every
   half-life measured from it. Those are refused in favour of source 2.
2. **First revision timestamp**, as a fallback.

Source 2 is fine for contemporary events (an article about a plane crash is
typically created within hours) but wrong for historical ones (the article on a
1912 shipwreck was created in 2001). The source is recorded per article, so the
analysis can be restricted to Wikidata-dated events as a robustness check, and
articles created more than three years after their event are dropped as
retrospective rather than contemporaneous records. On the live corpus the
fallback carries the majority: 395 of 690 articles are dated from their first
revision.

Redirects, disambiguation pages and chronicle pages (`2015 in music`, `Deaths
in 2015`) are excluded — the last of these sit inside the year-scoped
categories without being events, and their attention curve tracks the calendar
rather than any act of forgetting. Articles that never reached 40 human
revisions are dropped once their history is in hand, since a decay curve needs
something to decay from. Every exclusion is written to
`data/corpus_dropped.parquet` with its reason, so a filter that fires far more
often than expected is visible rather than silent.

## Why regexes on wikitext, not rendered HTML

The markup that matters — `{{cite news}}`, `{{dead link}}`, `url-status=dead`,
`archive-url=` — exists *only* in the source. The rendered page hides precisely
the maintenance signals that make link rot measurable. (One bug the tests
caught: `{{citation needed}}` was being counted as a citation, which inverts
the signal entirely.)

Link rot is measured through Wikipedia's own maintenance tags rather than by
requesting the URLs, which keeps the pipeline to one API and makes the
measurement reproducible years later.

---

## Repository layout

```
src/imd/
  config.py      paths, event taxonomy, all tunable parameters
  wikiapi.py     cached + rate-limited + retrying API client
  corpus.py      corpus construction and event-date resolution
  wikitext.py    citation / rot / structure extraction from wikitext
  collect.py     revisions, sparse snapshots, pageviews, monthly panel
  decay.py       the four decay models, Poisson MLE, AICc selection
  network.py     time-evolving link graph, edge survival, rot trajectories
  survival.py    Kaplan-Meier + log-rank, from scratch
  analysis.py    bootstrap CIs, BH-corrected tests, resistance index
  simulate.py    generative model with ground truth + recovery validation
  viz.py         figures
  cli.py         `imd` command
tests/           67 tests, no network required
report/          the written analysis
```

## Testing

```bash
pytest
```

67 tests, none of which touch the network. The API client is exercised against
a fake session (caching, retry, 404 handling, rate limiting); the estimators
are exercised against series with known parameters. Three tests are explicit
**quality gates** — if estimator recovery drops below the thresholds quoted
above, they fail, because at that point the numbers in the report are not
trustworthy.

A further group covers the **live-API contract**: which parameters MediaWiki
actually accepts, that the oldest revision is fetched one page at a time
(`rvdir`/`rvlimit` are rejected for multi-title queries), that a year-precision
Wikidata date is refused in favour of the revision timestamp, and that a corpus
frame survives a parquet round-trip. These exist because a synthetic dataset
cannot exercise any of it — every one of them was written against a bug the
simulated path ran straight past.

## Method notes and limitations

- **Edits are a proxy for attention, not attention itself.** Pageviews are
  collected where available, but the Wikimedia pageviews API only reaches back
  to July 2015, so it cannot anchor a decay curve for a 2008 event. Those
  months are structurally missing, not zero, and are treated as such.
- **Bot edits are excluded** by a name-and-tag heuristic. They are roughly
  constant in time and would flatten every curve.
- **The fitted exponent carries a ≈ −0.3 systematic bias** (visible in the
  validation table). It is close to constant across articles, so it does not
  affect rankings or between-class comparisons — which is what this project
  actually claims — but absolute exponents should be read with it in mind.
- **English Wikipedia only.** "Collective memory" here means the memory of one
  language edition's editor community, which is not the world's.
- **Category membership is a proxy for event type**, and Wikipedia's categories
  are inconsistent. An article can plausibly belong to several classes; the
  number of classes claiming it is recorded.

## License

MIT.
