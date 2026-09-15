"""Workplace and locations, per platform, from the captured fixtures -- and the
honesty rule they all follow (`jfl_intake.workplace`). No test opens a socket.

Fixtures were captured live on 2026-09-10; `greenhouse_anthropic_workplace_jobs.json`
on 2026-09-15 (seven Anthropic jobs chosen to cover every `Location Type` value
shape, trimmed of `data_compliance`). Where a fixture lacks a value a rule needs,
the test copies a real fixture record and changes only that field, saying so.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest
from jfl_intake.adapters import (
    ashby,
    breezy,
    greenhouse,
    lever,
    personio,
    pinpoint,
    recruitee,
    rippling,
    teamtailor,
)
from jfl_intake.adapters.smartrecruiters import _SmartRecruitersCheck
from jfl_intake.adapters.workable import _WorkableCheck
from jfl_intake.adapters.workday import _WorkdayCheck
from jfl_intake.http import Transport
from jfl_intake.normalise import fingerprint
from jfl_intake.workplace import (
    PLATFORMS_WITHOUT_WORKPLACE_FIELD,
    effective_include_unstated,
    from_location_text,
    greenhouse_metadata,
    include_unstated_by_default,
    says_remote_friendly,
    split_location_text,
)

FIXTURES = Path(__file__).parent / "fixtures"
NO_TRANSPORT = cast(Transport, None)  # `_record` never touches the transport


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def text_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


# -- the text rule ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["United Kingdom", "London", "Germany", "EMEA", "United States", "Remotely, Idaho"],
)
def test_a_place_name_alone_is_never_remote(text: str) -> None:
    """Cohere's live board: `United Kingdom` is Hybrid there, `London` jobs Remote."""
    assert from_location_text([text]) == "unknown"


def test_the_literal_word_remote_or_hybrid_decides_and_nothing_else() -> None:
    assert from_location_text(["Remote - UK"]) == "remote"
    assert from_location_text(["London (Hybrid)"]) == "hybrid"
    assert (
        from_location_text(["Remote-Friendly (Travel-Required)", "San Francisco, CA"]) == "remote"
    )


def test_text_naming_both_or_negating_is_unknown() -> None:
    assert from_location_text(["Remote, US", "Hybrid, London"]) == "unknown"
    assert from_location_text(["Non-remote, London"]) == "unknown"
    assert from_location_text(["Not remote"]) == "unknown"


def test_text_never_yields_onsite() -> None:
    assert from_location_text(["On-site, London office"]) == "unknown"


def test_greenhouse_location_text_is_split_on_semicolons_and_pipes_only() -> None:
    assert split_location_text("New York City, NY; San Francisco, CA | New York City, NY") == (
        "New York City, NY",
        "San Francisco, CA",
    )


# -- Greenhouse: employer metadata, then text --------------------------------------


def test_greenhouse_anthropic_location_type_shapes_map_and_keep_the_label() -> None:
    jobs = greenhouse.parse(fixture("greenhouse_anthropic_workplace_jobs.json")).jobs
    got = [(j.title, j.workplace, j.workplace_label) for j in jobs]
    assert got == [
        ("Applied AI Architect, Beneficial Deployments (Life Sciences)", "onsite", "On-Site"),
        ("Anthropic Fellows Program, AI Safety & Security", "onsite", "On-Site"),
        ("Business Systems Analyst", "remote", "Remote"),
        ("Applied AI Architect, Industries", "hybrid", "Hybrid (Travel-Required)"),
        # `Location Type: null`, location "Sydney, Australia": nothing stated.
        ("Applied AI Architect", "unknown", None),
        # `Location Type: null`, location text "Remote-Friendly (...)": the text rule.
        ("Staff+ Software Engineer, Data Infrastructure", "remote", None),
        # `On-Site` although the text says "Remote-Friendly": metadata wins.
        ("Compute Country Lead, Canada", "onsite", "On-Site"),
    ]


def test_greenhouse_locations_are_every_listed_place_and_location_is_unchanged() -> None:
    jobs = greenhouse.parse(fixture("greenhouse_anthropic_workplace_jobs.json")).jobs
    fellows = jobs[1]
    assert fellows.location == (
        "London, UK; Ontario, CAN; Remote-Friendly, United States; San Francisco, CA"
    )
    assert fellows.locations == (
        "London, UK",
        "Ontario, CAN",
        "Remote-Friendly, United States",
        "San Francisco, CA",
    )
    assert fellows.fingerprint == fingerprint(fellows.title, fellows.location)


def test_greenhouse_literal_remote_in_location_text_is_remote_without_metadata() -> None:
    body = fixture("greenhouse_anthropic_jobs.json")
    job = copy.deepcopy(body["jobs"][1])  # Singapore, On-Site
    job["metadata"] = []
    job["location"] = {"name": "Remote, Singapore"}
    parsed = greenhouse.parse({"jobs": [job], "meta": {"total": 1}}).jobs[0]
    assert parsed.workplace == "remote"
    assert parsed.workplace_label is None


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("Location Type", "On-Site", ("onsite", "On-Site")),
        ("location type", "Remote", ("remote", "Remote")),
        ("Workplace Type", "Hybrid (Travel-Required)", ("hybrid", "Hybrid (Travel-Required)")),
        ("Work Type", "In Office", ("onsite", "In Office")),
        ("Remote", "Onsite", ("onsite", "Onsite")),
        ("Location Type", None, None),
        ("Location Type", "Flexible", None),
        ("Location Type", "Remote or Hybrid", None),  # contradicts itself: fall through
        ("Department", "Remote", None),  # not a workplace field name
    ],
)
def test_greenhouse_metadata_names_and_values(
    name: str, value: str | None, expected: tuple[str, str] | None
) -> None:
    metadata = [{"id": 1, "name": name, "value": value, "value_type": "single_select"}]
    assert greenhouse_metadata(metadata) == expected


# -- the platforms with a structured field ----------------------------------------------


def test_ashby_uses_workplace_type_and_ignores_is_remote() -> None:
    """The OpenAI fixture has `workplaceType: null` everywhere; one record is
    copied with Cohere's observed combination -- `Hybrid`, yet `isRemote: true`.
    """
    body = fixture("ashby_openai_job_board.json")
    assert [j.workplace for j in ashby.parse(body).jobs] == ["unknown"] * 3

    hybrid = copy.deepcopy(body["jobs"][0])
    hybrid.update(workplaceType="Hybrid", isRemote=True, location="United Kingdom")
    onsite = copy.deepcopy(body["jobs"][1])
    onsite.update(workplaceType="OnSite", isRemote=True)
    remote = copy.deepcopy(body["jobs"][2])
    remote.update(workplaceType="Remote", isRemote=False, location="London")
    jobs = ashby.parse({"jobs": [hybrid, onsite, remote]}).jobs
    assert [j.workplace for j in jobs] == ["hybrid", "onsite", "remote"]


def test_ashby_locations_include_secondary_locations() -> None:
    body = fixture("ashby_openai_job_board.json")
    job = copy.deepcopy(body["jobs"][2])
    job["secondaryLocations"] = [{"location": "Seoul, South Korea"}, {"location": "Tokyo, Japan"}]
    parsed = ashby.parse({"jobs": [job]}).jobs[0]
    assert parsed.locations == ("Tokyo, Japan", "Seoul, South Korea")
    assert parsed.location == "Tokyo, Japan"


def test_lever_workplace_type_and_all_locations() -> None:
    jobs = lever.parse(fixture("lever_palantir_postings.json")).jobs
    assert [j.workplace for j in jobs] == ["hybrid", "hybrid", "hybrid"]
    assert jobs[0].locations == ("London, United Kingdom",)


def test_lever_unspecified_falls_back_to_the_text_and_all_locations_to_location() -> None:
    posting = copy.deepcopy(fixture("lever_palantir_postings.json")[0])
    posting["workplaceType"] = "unspecified"
    del posting["categories"]["allLocations"]
    parsed = lever.parse([posting]).jobs[0]
    assert parsed.workplace == "unknown"
    assert parsed.locations == ("London, United Kingdom",)


def test_workable_workplace_and_locations() -> None:
    check = _WorkableCheck(NO_TRANSPORT, "huggingface")
    jobs = [check._record(j)[1] for j in fixture("workable_huggingface_jobs.json")["results"]]
    assert [j.workplace for j in jobs if j] == ["remote", "remote", "remote"]
    assert jobs[0] is not None
    assert jobs[0].locations == ("Paris, Île-de-France, France",)
    assert jobs[0].location == "Paris, France"  # unchanged: the fingerprint's half


def test_workable_on_site_and_the_remote_boolean_fallback() -> None:
    check = _WorkableCheck(NO_TRANSPORT, "huggingface")
    base = fixture("workable_huggingface_jobs.json")["results"][0]
    on_site = {**base, "workplace": "on_site", "remote": False}  # Devsinc's live shape
    no_field_false = {k: v for k, v in base.items() if k != "workplace"} | {"remote": False}
    no_field_true = {k: v for k, v in base.items() if k != "workplace"} | {"remote": True}
    got = [check._record(j)[1] for j in (on_site, no_field_false, no_field_true)]
    assert [j.workplace for j in got if j] == ["onsite", "unknown", "remote"]


def test_pinpoint_workplace_type_and_location_name() -> None:
    jobs = pinpoint.parse(fixture("pinpoint_sunking_postings.json")).jobs
    assert {j.workplace for j in jobs} == {"onsite"}
    assert jobs[0].locations == ("South Africa",)


def test_teamtailor_remote_status_and_every_tt_location() -> None:
    jobs = teamtailor.parse(text_fixture("teamtailor_jobs.rss")).jobs
    assert [j.workplace for j in jobs] == ["hybrid", "hybrid", "hybrid"]
    assert jobs[2].locations == ("Toronto",)  # blank tt:name, tt:city fallback


@pytest.mark.parametrize(
    ("status", "expected"),
    [("hybrid", "hybrid"), ("fully", "remote"), ("none", "unknown"), ("temporary", "unknown")],
)
def test_teamtailor_remote_status_values(status: str, expected: str) -> None:
    feed = text_fixture("teamtailor_jobs.rss").replace(
        "<remoteStatus>hybrid</remoteStatus>", f"<remoteStatus>{status}</remoteStatus>"
    )
    assert {j.workplace for j in teamtailor.parse(feed).jobs} == {expected}


def test_teamtailor_none_is_never_read_past_to_the_location_text() -> None:
    feed = (
        text_fixture("teamtailor_jobs.rss")
        .replace("<remoteStatus>hybrid</remoteStatus>", "<remoteStatus>none</remoteStatus>")
        .replace("<tt:name>Stockholm</tt:name>", "<tt:name>Remote, Sweden</tt:name>")
    )
    assert teamtailor.parse(feed).jobs[0].workplace == "unknown"


def test_teamtailor_lists_every_location_of_a_multi_location_item() -> None:
    feed = text_fixture("teamtailor_jobs.rss").replace(
        "</tt:location>\n      </tt:locations>",
        "</tt:location>\n        <tt:location><tt:name>Gothenburg</tt:name></tt:location>\n"
        "      </tt:locations>",
        1,
    )
    first = teamtailor.parse(feed).jobs[0]
    assert first.locations == ("Stockholm", "Gothenburg")
    assert first.location == "Stockholm"


def test_recruitee_booleans_and_location_names() -> None:
    jobs = recruitee.parse(fixture("recruitee_offers.json")).jobs
    assert [j.workplace for j in jobs] == ["remote", "remote", "remote"]
    assert jobs[0].locations == ("Work from Anywhere in the US",)
    assert jobs[0].location == "Remote job"


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"remote": False, "hybrid": True, "on_site": False}, "hybrid"),
        ({"remote": False, "hybrid": False, "on_site": True}, "onsite"),
        ({"remote": False, "hybrid": False, "on_site": False}, "unknown"),
        ({"remote": True, "hybrid": True, "on_site": False}, "unknown"),
    ],
)
def test_recruitee_exactly_one_true_boolean(flags: dict[str, bool], expected: str) -> None:
    offer = {**fixture("recruitee_offers.json")["offers"][0], **flags}
    assert recruitee.parse({"offers": [offer]}).jobs[0].workplace == expected


def test_smartrecruiters_booleans_and_full_location() -> None:
    check = _SmartRecruitersCheck(NO_TRANSPORT, "BoschGroup")
    postings = fixture("smartrecruiters_bosch_postings.json")["content"]
    jobs = [check._record(p)[1] for p in postings]
    # Three `remote: false, hybrid: false`, one `hybrid: true`.
    assert [j.workplace for j in jobs if j] == ["unknown", "unknown", "unknown", "hybrid"]
    assert jobs[3] is not None
    assert jobs[3].locations == ("Charleston, SC, United States",)
    assert jobs[3].location == "Charleston, SC, us"  # unchanged


def test_smartrecruiters_remote_true_is_remote() -> None:
    check = _SmartRecruitersCheck(NO_TRANSPORT, "BoschGroup")
    posting = copy.deepcopy(fixture("smartrecruiters_bosch_postings.json")["content"][0])
    posting["location"]["remote"] = True
    record = check._record(posting)[1]
    assert record is not None and record.workplace == "remote"


def test_breezy_is_remote_false_is_unknown_and_true_is_remote() -> None:
    body = fixture("breezy_jobs.json")
    jobs = breezy.parse(body).jobs
    # First has `is_remote: false`; the other two carry no `is_remote` at all.
    assert [j.workplace for j in jobs] == ["unknown", "unknown", "unknown"]
    assert jobs[0].locations == ("Chaos, FL",)
    assert jobs[1].locations == ("New York, NY",)  # empty `locations[]`: falls back

    remote = copy.deepcopy(body[0])
    remote["location"]["is_remote"] = True
    assert breezy.parse([remote]).jobs[0].workplace == "remote"


def test_breezy_false_is_not_read_past_to_the_location_text() -> None:
    job = copy.deepcopy(fixture("breezy_jobs.json")[0])
    job["location"]["name"] = "Remote, FL"
    job["locations"][0]["name"] = "Remote, FL"
    assert breezy.parse([job]).jobs[0].workplace == "unknown"


# -- the platforms with no structured field ------------------------------------------


def test_rippling_merged_labels_and_text_rule_only() -> None:
    jobs = rippling.parse(fixture("rippling_jobs.json")).jobs
    assert [j.workplace for j in jobs] == ["unknown"] * 4
    assert jobs[0].locations == ("Austin, TX",)

    row = fixture("rippling_jobs.json")[1]
    rows = [row, {**row, "workLocation": {"label": "Remote (US)", "id": "Remote (US)"}}]
    merged = rippling.parse(rows).jobs[0]
    assert merged.locations == ("Cleveland, OH", "Remote (US)")
    assert merged.workplace == "remote"


def test_workday_locations_text_is_stored_as_given() -> None:
    check = _WorkdayCheck(NO_TRANSPORT, "adobe", "wd5", "external_experienced")
    postings = fixture("workday_adobe_page0.json")["jobPostings"]
    records = [check._record(p)[1] for p in postings]
    texts = {r.locations for r in records if r}
    assert ("2 Locations",) in texts  # never expanded into invented places
    assert {r.workplace for r in records if r} == {"unknown"}


def test_personio_office_and_additional_offices() -> None:
    job = personio.parse(text_fixture("personio_position.xml")).jobs[0]
    assert job.locations == ("Munich", "Berlin")
    assert job.location == "Munich"
    assert job.workplace == "unknown"


# -- identity is untouched ---------------------------------------------------------------


def test_workplace_and_locations_never_change_the_fingerprint() -> None:
    jobs = greenhouse.parse(fixture("greenhouse_anthropic_jobs.json")).jobs
    for job in jobs:
        assert job.fingerprint == fingerprint(job.title, job.location)


# -- the per-board include-unstated default ----------------------------------------------


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("greenhouse", True),
        ("rippling", True),
        ("workday", True),
        ("personio", True),
        ("ashby", False),
        ("lever", False),
        ("workable", False),
        ("pinpoint", False),
        ("teamtailor", False),
        ("recruitee", False),
        ("smartrecruiters", False),
        ("breezy", False),
    ],
)
def test_include_unstated_defaults_by_platform(platform: Any, expected: bool) -> None:
    assert include_unstated_by_default(platform) is expected
    assert effective_include_unstated(platform, None) is expected
    assert effective_include_unstated(platform, not expected) is (not expected)


def test_the_no_field_set_is_exactly_four_platforms() -> None:
    assert {"greenhouse", "rippling", "workday", "personio"} == PLATFORMS_WITHOUT_WORKPLACE_FIELD


# -- remote-friendly evidence ------------------------------------------------------------


def _evidence(
    locations: tuple[str, ...] = (), location: str | None = None, label: str | None = None
) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(locations=locations, location=location, workplace_label=label)


@pytest.mark.parametrize(
    "text",
    [
        "Remote-Friendly (Travel-Required)",
        "Remote-Friendly (Travel Required) | Canada",
        "remote friendly, United States",
        "REMOTE—FRIENDLY",
    ],
)
def test_says_remote_friendly_reads_the_observed_phrase_however_punctuated(text: str) -> None:
    assert says_remote_friendly(_evidence(locations=("London, UK", text)))
    assert says_remote_friendly(_evidence(location=text))  # a row with no `locations`
    assert says_remote_friendly(_evidence(label=text))


@pytest.mark.parametrize(
    "text",
    [
        "Remote",
        "Hybrid (Travel-Required)",
        "Remote first",  # the owner's example of STRICT remote -- deliberately not evidence
        "Remote-first, UK",
        "Friendly Remote Team",  # the words, but not consecutive in that order
        "Not remote-friendly",
        "non remote friendly",
        "London, UK",
        "",
    ],
)
def test_says_remote_friendly_is_only_that_phrase_un_negated(text: str) -> None:
    assert not says_remote_friendly(_evidence(locations=(text,), location=text, label=text))


def test_says_remote_friendly_on_the_captured_anthropic_jobs() -> None:
    jobs = greenhouse.parse(fixture("greenhouse_anthropic_workplace_jobs.json")).jobs
    assert [(j.title, says_remote_friendly(j)) for j in jobs] == [
        ("Applied AI Architect, Beneficial Deployments (Life Sciences)", False),
        ("Anthropic Fellows Program, AI Safety & Security", True),
        ("Business Systems Analyst", True),
        ("Applied AI Architect, Industries", False),
        ("Applied AI Architect", False),
        ("Staff+ Software Engineer, Data Infrastructure", True),
        ("Compute Country Lead, Canada", True),
    ]
    # Evidence is read, never stored: the structured answer is unchanged.
    assert jobs[6].workplace == "onsite"
