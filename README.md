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

Recovery on 286 simulated articles after both fixes:

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

imd corpus      # walk the category taxonomy, resolve event dates   (~10 min)
imd collect     # revisions, wikitext snapshots, pageviews          (hours)
imd analyze     # fit decay models, run the statistics
imd figures     # render every figure
```

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

1. **Wikidata P585** (point in time) / P580 (start time) on the article's item.
2. **First revision timestamp**, as a fallback.

Source 2 is fine for contemporary events (an article about a plane crash is
typically created within hours) but wrong for historical ones (the article on a
1912 shipwreck was created in 2001). The source is recorded per article, so the
analysis can be restricted to Wikidata-dated events as a robustness check, and
articles created more than three years after their event are dropped as
retrospective rather than contemporaneous records.

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
tests/           60 tests, no network required
report/          the written analysis
```

## Testing

```bash
pytest
```

60 tests, none of which touch the network. The API client is exercised against
a fake session (caching, retry, 404 handling, rate limiting); the estimators
are exercised against series with known parameters. Three tests are explicit
**quality gates** — if estimator recovery drops below the thresholds quoted
above, they fail, because at that point the numbers in the report are not
trustworthy.

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
