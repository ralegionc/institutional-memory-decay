"""Tests for wikitext feature extraction."""

from __future__ import annotations

import pytest

from imd import wikitext

SAMPLE = """
'''The Example Incident''' was an event.<ref name="a">{{cite news |url=http://news.example.com/x |title=X |url-status=dead |archive-url=https://web.archive.org/web/2020/http://news.example.com/x}}</ref>
It had consequences.<ref name="a" /> Later analysis disagreed.<ref>{{cite web|url=https://gov.example.org/report.pdf|title=Report}}</ref>

== Background ==
See [[Another Event]] and [[Some Place|there]].{{citation needed}}
[[File:Photo.jpg|thumb|A photo]]

=== Detail ===
{{dead link|date=January 2020}}
<!-- [[Hidden Link]] should not be counted -->

== References ==
{{reflist}}
[[Category:2011 disasters]]
"""


@pytest.fixture(scope="module")
def feats():
    return wikitext.extract_features(SAMPLE)


def test_reference_counting_excludes_reuse(feats):
    """A ``<ref name="a" />`` pointer is not a second source."""
    assert feats.n_reused_refs == 1
    assert feats.n_refs == 2


def test_citation_templates(feats):
    assert feats.n_cite_templates == 2


def test_rot_and_archive_signals(feats):
    assert feats.n_dead_link_tags == 1
    assert feats.n_url_status_dead == 1
    assert feats.n_archive_urls >= 1
    assert feats.n_citation_needed == 1


def test_structure_counts(feats):
    assert feats.n_sections == 3
    assert feats.n_images == 1
    assert feats.n_categories == 1


def test_wikilinks_exclude_namespaces_and_comments(feats):
    assert "Another Event" in feats.wikilinks
    assert "Some Place" in feats.wikilinks
    # File/Category links are not article links; commented-out links are gone.
    assert not any(w.startswith(("File", "Category")) for w in feats.wikilinks)
    assert "Hidden Link" not in feats.wikilinks


def test_domains_are_normalised():
    f = wikitext.extract_features("http://www.Example.COM/a https://example.com/b")
    assert f.domains == ["example.com"]
    assert f.n_urls == 2
    assert f.n_unique_domains == 1


def test_derived_rates(feats):
    assert 0 <= feats.rot_rate <= 1
    assert 0 <= feats.archive_rate <= 1
    assert feats.refs_per_kb > 0


def test_empty_input_is_safe():
    for value in (None, "", "   "):
        f = wikitext.extract_features(value)
        assert f.n_refs == 0
        assert f.rot_rate == 0.0
        assert f.to_row()["n_bytes"] >= 0


def test_normalise_title():
    assert wikitext.normalise_title("some_page#section") == "Some page"
    assert wikitext.normalise_title("  another page  ") == "Another page"
    assert wikitext.normalise_title("") == ""


def test_to_row_is_flat(feats):
    row = feats.to_row()
    assert "wikilinks" not in row and "domains" not in row
    assert all(isinstance(v, (int, float)) for v in row.values())
