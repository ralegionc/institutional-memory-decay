"""Tests for the collection layer, network analysis, survival and the API client.

These use fixtures and mocked HTTP rather than the network, so the suite runs
in CI without touching Wikipedia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imd import analysis, collect, config, network, simulate, survival, wikiapi


# --------------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_dataset():
    t0 = pd.Timestamp("2015-01-15", tz="UTC")
    corpus = pd.DataFrame(
        [{"title": "E1", "event_date": t0, "event_class": "test", "date_source": "wikidata:P585"}]
    )
    stamps = [t0 + pd.Timedelta(days=d) for d in (1, 2, 3, 40, 400, 402)]
    revisions = pd.DataFrame(
        {
            "title": "E1",
            "revid": range(6),
            "timestamp": stamps,
            "size": [100, 200, 300, 320, 330, 335],
            "user": ["a", "b", "a", "c", "d", "Xbot"],
            "minor": False,
            "anon": [True, False, False, False, False, False],
            "comment_len": 10,
            "is_bot": [False] * 5 + [True],
            "is_revert": False,
            "delta": [100, 100, 100, 20, 10, 5],
        }
    )
    return corpus, revisions


def test_panel_is_balanced_and_zero_filled(tiny_dataset):
    corpus, revisions = tiny_dataset
    panel = collect.build_panel(corpus, revisions)
    months = panel["month_index"].to_numpy()
    # No gaps: a month with no edits must be present as a zero.
    assert list(months) == list(range(months.min(), months.max() + 1))
    assert panel.loc[panel["month_index"] == 5, "edits"].iloc[0] == 0
    assert panel["edits"].sum() == 5  # the bot edit is excluded


def test_bot_edits_are_separated(tiny_dataset):
    corpus, revisions = tiny_dataset
    panel = collect.build_panel(corpus, revisions)
    assert panel["bot_edits"].sum() == 1
    assert panel["editors"].max() == 2  # month 0 saw a, b, a -> 2 distinct humans


def test_month_index_anchors_on_the_event(tiny_dataset):
    corpus, revisions = tiny_dataset
    panel = collect.build_panel(corpus, revisions)
    # Edits 1-3 days after the event belong to month 0.
    assert panel.loc[panel["month_index"] == 0, "edits"].iloc[0] == 3
    # The edit 400 days later lands in month 13.
    assert panel.loc[panel["month_index"] == 13, "edits"].iloc[0] == 1


def test_bot_detection_heuristic():
    assert collect._looks_like_bot("ClueBot", None)
    assert collect._looks_like_bot("Someone", ["mw-rollback", "bot"])
    assert not collect._looks_like_bot("Robert", None)
    assert not collect._looks_like_bot(None, None)


# --------------------------------------------------------------------------
# Snapshot selection
# --------------------------------------------------------------------------


def test_snapshot_never_looks_into_the_future(tiny_dataset):
    corpus, revisions = tiny_dataset
    settings = config.Settings(snapshot_offsets_years=(0.0, 1.0))
    picks = collect.pick_snapshot_revisions(revisions, corpus, settings)
    assert set(picks["offset_years"]) == {0.0, 1.0}
    later = picks[picks["offset_years"] > 0]
    for row in later.itertuples():
        assert row.snapshot_time <= row.target_time
    # The baseline is the creation revision, one day after the event.
    base = picks[picks["offset_years"] == 0.0].iloc[0]
    assert base["lag_days"] == -1


def test_baseline_snapshot_rejects_a_late_creation(tiny_dataset):
    """A retrospectively created article gets no offset-0 snapshot."""
    corpus, revisions = tiny_dataset
    revisions = revisions.copy()
    revisions["timestamp"] = revisions["timestamp"] + pd.Timedelta(days=200)
    settings = config.Settings(snapshot_offsets_years=(0.0, 1.0))
    picks = collect.pick_snapshot_revisions(revisions, corpus, settings)
    assert 0.0 not in set(picks["offset_years"])


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


@pytest.fixture
def link_frame():
    return pd.DataFrame(
        [
            {"title": "A", "offset_years": 0.0, "target": "B"},
            {"title": "A", "offset_years": 0.0, "target": "C"},
            {"title": "B", "offset_years": 0.0, "target": "C"},
            {"title": "A", "offset_years": 5.0, "target": "B"},  # A->C dropped
            {"title": "C", "offset_years": 5.0, "target": "A"},  # new edge
        ]
    )


def test_graphs_are_restricted_to_the_corpus(link_frame):
    graphs = network.build_snapshot_graphs(link_frame, {"A", "B", "C"})
    assert set(graphs) == {0.0, 5.0}
    assert graphs[0.0].number_of_edges() == 3
    # An outside target must not create a node.
    extra = pd.concat(
        [link_frame, pd.DataFrame([{"title": "A", "offset_years": 0.0, "target": "Z"}])]
    )
    g = network.build_snapshot_graphs(extra, {"A", "B", "C"})[0.0]
    assert "Z" not in g.nodes


def test_edge_survival_counts_losses_and_gains(link_frame):
    graphs = network.build_snapshot_graphs(link_frame, {"A", "B", "C"})
    surv = network.edge_survival(graphs).set_index("offset_years")
    assert surv.loc[0.0, "survival_rate"] == 1.0
    assert surv.loc[5.0, "n_surviving"] == 1  # only A->B persists
    assert surv.loc[5.0, "survival_rate"] == pytest.approx(1 / 3)
    assert surv.loc[5.0, "n_new"] == 1  # C->A appeared


def test_in_degree_trajectory(link_frame):
    graphs = network.build_snapshot_graphs(link_frame, {"A", "B", "C"})
    traj = network.per_article_link_trajectory(network.network_metrics(graphs))
    c = traj.set_index("title").loc["C"]
    assert c["indeg_first"] == 2 and c["indeg_last"] == 0
    assert c["indeg_slope"] < 0


def test_rot_trajectories_shape():
    snapshots = pd.DataFrame(
        {
            "title": ["A", "A"],
            "offset_years": [0.0, 5.0],
            "n_urls": [10, 10],
            "n_refs": [8, 9],
            "n_bytes": [1000, 1200],
            "n_dead_link_tags": [0, 3],
            "n_url_status_dead": [0, 1],
            "n_archive_urls": [1, 6],
            "n_citation_needed": [0, 2],
            "n_maintenance_tags": [0, 1],
        }
    )
    rot = network.rot_trajectories(snapshots).set_index("offset_years")
    assert rot.loc[0.0, "median_dead_per_100"] == 0
    assert rot.loc[5.0, "median_dead_per_100"] == 40
    assert rot.loc[5.0, "mean_archive_pct"] == 60


# --------------------------------------------------------------------------
# Survival analysis
# --------------------------------------------------------------------------


def test_kaplan_meier_matches_hand_computation():
    """Classic worked example: n=5, events at 2 and 5, censoring at 3."""
    d = np.array([2.0, 3.0, 5.0, 5.0, 7.0])
    o = np.array([True, False, True, True, False])
    km = survival.kaplan_meier(d, o)
    # S(2) = 4/5; then at t=5, 3 at risk, 2 events -> S = 0.8 * (1/3)
    assert km.survival[1] == pytest.approx(0.8)
    assert km.survival[2] == pytest.approx(0.8 / 3)


def test_kaplan_meier_bands_are_bounded():
    rng = np.random.default_rng(0)
    d = rng.exponential(20, 80)
    o = rng.random(80) < 0.7
    km = survival.kaplan_meier(d, o)
    assert np.all(km.lower >= 0) and np.all(km.upper <= 1.0000001)
    assert np.all(km.lower <= km.survival + 1e-9)


def test_censoring_changes_the_answer():
    """The whole reason for using survival analysis: ignoring censoring
    understates persistence."""
    d = np.array([10.0] * 5 + [50.0] * 5)
    all_events = survival.kaplan_meier(d, np.ones(10, dtype=bool))
    censored = survival.kaplan_meier(d, np.array([True] * 5 + [False] * 5))
    assert all_events.median_survival() <= censored.median_survival()


def test_logrank_detects_a_real_difference():
    rng = np.random.default_rng(1)
    fast = rng.exponential(8, 120)
    slow = rng.exponential(40, 120)
    groups = {
        "fast": (fast, np.ones(120, dtype=bool)),
        "slow": (slow, np.ones(120, dtype=bool)),
    }
    result = survival.logrank_test(groups)
    assert result["p_value"] < 1e-6


def test_logrank_null_is_not_significant():
    rng = np.random.default_rng(2)
    groups = {
        name: (rng.exponential(20, 150), np.ones(150, dtype=bool))
        for name in ("a", "b")
    }
    assert survival.logrank_test(groups)["p_value"] > 0.01


def test_dormancy_requires_a_persistent_drop():
    """A single quiet year inside an active history is not dormancy."""
    y = np.concatenate([np.full(6, 100.0), np.zeros(14), np.full(40, 90.0)])
    panel = pd.DataFrame(
        {"title": "E", "month_index": range(len(y)), "edits": y, "event_class": "t"}
    )
    events = survival.dormancy_events(panel)
    assert not events["observed"].iloc[0]

    y2 = np.concatenate([np.full(6, 100.0), np.zeros(60)])
    panel2 = pd.DataFrame(
        {"title": "E", "month_index": range(len(y2)), "edits": y2, "event_class": "t"}
    )
    assert survival.dormancy_events(panel2)["observed"].iloc[0]


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------


def test_benjamini_hochberg_is_monotone_and_conservative():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    adj = analysis.benjamini_hochberg(p)
    assert np.all(np.diff(adj) >= -1e-12)
    assert np.all(adj >= p - 1e-12)
    assert adj[-1] <= 1.0


def test_cliffs_delta_signs():
    a = np.arange(50.0)
    b = np.arange(50.0) + 100
    assert analysis.cliffs_delta(a, b) == pytest.approx(-1.0)
    assert analysis.cliffs_delta(b, a) == pytest.approx(1.0)
    assert abs(analysis.cliffs_delta(a, a)) < 1e-9


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    values = rng.lognormal(3, 1, 300)
    point, lo, hi = analysis.bootstrap_ci(values, n_boot=500, seed=3)
    assert lo <= point <= hi
    assert analysis.bootstrap_ci([])[0] != analysis.bootstrap_ci([])[0]  # NaN


def test_resistance_index_treats_infinity_as_maximal():
    frame = pd.DataFrame(
        {
            "title": ["a", "b", "c", "d", "e", "f"],
            "empirical_half_life": [1.0, 2.0, 3.0, 4.0, 5.0, np.inf],
            "share_year1": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        }
    )
    out = analysis.resistance_index(frame)
    assert out.iloc[0]["title"] == "f"
    assert out["resistance_index"].is_monotonic_decreasing


# --------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


def test_cache_prevents_a_second_request(tmp_path):
    session = _FakeSession([_FakeResponse({"query": {"pages": []}})])
    client = wikiapi.WikiClient(
        session=session, cache_path=tmp_path / "cache.sqlite"
    )
    a = client._request("https://example.test/api", {"x": 1})
    b = client._request("https://example.test/api", {"x": 1})
    assert a == b
    assert session.calls == 1
    assert client.stats == {"hits": 1, "misses": 1, "errors": 0}


def test_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(wikiapi.time, "sleep", lambda *_: None)
    session = _FakeSession(
        [_FakeResponse({}, 503), _FakeResponse({}, 429), _FakeResponse({"ok": True})]
    )
    client = wikiapi.WikiClient(session=session, cache_path=tmp_path / "c.sqlite")
    assert client._request("https://example.test/api")["ok"] is True
    assert session.calls == 3


def test_404_is_cached_as_missing(tmp_path):
    session = _FakeSession([_FakeResponse({}, 404)])
    client = wikiapi.WikiClient(session=session, cache_path=tmp_path / "c.sqlite")
    assert client._request("https://example.test/x")["__missing__"] is True


def test_token_bucket_limits_rate(monkeypatch):
    slept = []
    monkeypatch.setattr(wikiapi.time, "sleep", lambda s: slept.append(s))
    bucket = wikiapi.TokenBucket(rate_per_second=2.0, burst=1)
    bucket.acquire()
    bucket.acquire()
    assert slept, "second immediate acquire should have waited"


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------


def test_simulator_produces_a_coherent_dataset():
    data = simulate.simulate_corpus(n_articles=24, seed=1)
    assert not data["corpus"].empty
    assert set(data["revisions"]["title"]) <= set(data["corpus"]["title"])
    assert set(data["snapshot_links"]["target"]) <= set(data["corpus"]["title"])
    # Ground truth must line up one-to-one with the corpus.
    assert len(data["ground_truth"]) == len(data["corpus"])


def test_simulated_class_differences_are_real():
    """The simulator must actually encode the effect the analysis looks for,
    otherwise a null result on synthetic data would be uninterpretable."""
    data = simulate.simulate_corpus(n_articles=120, seed=4)
    truth = data["ground_truth"]
    fast = truth[truth["event_class"] == "culture_award"]["true_exponent"].median()
    slow = truth[truth["event_class"] == "armed_conflict"]["true_exponent"].median()
    assert fast > slow
