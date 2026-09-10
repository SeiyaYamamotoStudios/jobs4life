"""URL -> platform detection: deterministic, by pattern, and never a guess."""

from __future__ import annotations

import pytest
from jfl_intake.detect import (
    FORBIDDEN_SOURCE_MESSAGE,
    ForbiddenSourceError,
    MalformedBoardUrlError,
    UnsupportedBoardError,
    detect_board,
)

NVIDIA = {"tenant": "nvidia", "wd": "wd5", "site": "NVIDIAExternalCareerSite"}


@pytest.mark.parametrize(
    ("url", "platform", "key"),
    [
        ("https://boards.greenhouse.io/anthropic", "greenhouse", {"token": "anthropic"}),
        (
            "https://job-boards.greenhouse.io/anthropic/jobs/4461450008",
            "greenhouse",
            {"token": "anthropic"},
        ),
        ("boards.greenhouse.io/Anthropic?gh_src=abc", "greenhouse", {"token": "anthropic"}),
        (
            "https://boards.greenhouse.io/embed/job_board?for=anthropic",
            "greenhouse",
            {"token": "anthropic"},
        ),
        ("https://jobs.ashbyhq.com/openai", "ashby", {"name": "openai"}),
        (
            "https://jobs.ashbyhq.com/openai/8fb1615c-34bf-47c4-a1d1-b7b2f836bbd3",
            "ashby",
            {"name": "openai"},
        ),
        ("https://jobs.lever.co/palantir", "lever", {"company": "palantir"}),
        (
            "https://jobs.lever.co/palantir/ac978161-6f46-4f6b-ad9e-a258e642751c",
            "lever",
            {"company": "palantir"},
        ),
        ("https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite", "workday", NVIDIA),
        (
            "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/"
            "US-CA-Santa-Clara/Senior-Engineer_JR2015858",
            "workday",
            NVIDIA,
        ),
        (
            "https://adobe.wd5.myworkdayjobs.com/external_experienced",
            "workday",
            {"tenant": "adobe", "wd": "wd5", "site": "external_experienced"},
        ),
        (
            "https://jobs.smartrecruiters.com/BoschGroup",
            "smartrecruiters",
            {"company_id": "BoschGroup"},
        ),
        (
            "https://careers.smartrecruiters.com/BoschGroup",
            "smartrecruiters",
            {"company_id": "BoschGroup"},
        ),
        ("https://ats.rippling.com/rippling", "rippling", {"slug": "rippling"}),
        ("https://ats.rippling.com/rippling/jobs", "rippling", {"slug": "rippling"}),
        ("https://breezy.breezy.hr", "breezy", {"company": "breezy"}),
        ("https://Breezy.breezy.hr", "breezy", {"company": "breezy"}),
        (
            "https://career.teamtailor.com/jobs.rss",
            "teamtailor",
            {"site": "career.teamtailor.com"},
        ),
        (
            "https://personio.jobs.personio.de/xml",
            "personio",
            {"company": "personio", "tld": "de"},
        ),
        (
            "https://personio.jobs.personio.com/xml",
            "personio",
            {"company": "personio", "tld": "com"},
        ),
        ("https://make.recruitee.com/o/delivery-lead-3", "recruitee", {"company": "make"}),
        (
            "https://sunking.pinpointhq.com/en/postings/d73a7468",
            "pinpoint",
            {"company": "sunking"},
        ),
        ("https://apply.workable.com/huggingface", "workable", {"subdomain": "huggingface"}),
        (
            "https://apply.workable.com/huggingface/j/9E2A4C02C7",
            "workable",
            {"subdomain": "huggingface"},
        ),
    ],
)
def test_each_supported_platform_is_detected(url: str, platform: str, key: dict[str, str]) -> None:
    ref = detect_board(url)
    assert ref.platform == platform
    assert ref.board_key == key
    assert ref.board_url == url.strip()


@pytest.mark.parametrize(
    "url",
    [
        "https://www.linkedin.com/jobs/view/4012345678",
        "linkedin.com/company/anthropic/jobs",
        "https://uk.linkedin.com/jobs/search?keywords=engineering",
        "https://lnkd.in/eAbCdEf",
        "https://www.indeed.com/viewjob?jk=abc123",
        "https://uk.indeed.com/jobs?q=engineering+manager",
        "https://www.indeed.co.uk/cmp/Anthropic/jobs",
        # An ATS URL smuggled inside a LinkedIn redirect is still LinkedIn.
        "https://www.linkedin.com/redir/redirect?url=https://boards.greenhouse.io/anthropic",
    ],
)
def test_linkedin_and_indeed_are_rejected_with_a_clear_message(url: str) -> None:
    with pytest.raises(ForbiddenSourceError) as caught:
        detect_board(url)
    assert str(caught.value) == FORBIDDEN_SOURCE_MESSAGE
    assert "LinkedIn and Indeed are not supported" in str(caught.value)
    assert "Greenhouse" in str(caught.value)  # it says what to paste instead


@pytest.mark.parametrize(
    "url",
    [
        "https://careers.example.com/jobs",
        "https://notlinkedin.com/jobs",  # the suffix match is on a label boundary
        "https://job-boards.eu.greenhouse.io/anthropic",  # unverified, so unsupported
        "https://boards.greenhouse.io/",
        "https://jobs.lever.co/",
        "https://jobs.ashbyhq.com",
        "https://nvidia.wd5.myworkdayjobs.com/",
        "https://nvidia.wd5.myworkdayjobs.com/en-US/",
        "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/x/jobs",
        # A Teamtailor board on the employer's own custom domain cannot be
        # recognised by pattern -- nothing in the URL says "Teamtailor".
        "https://careers.example.com/jobs.rss",
        "https://jobs.smartrecruiters.com/",
        "https://careers.smartrecruiters.com",
        "https://ats.rippling.com/",
        "https://breezy.hr",  # no subdomain
        "https://.breezy.hr",
        "https://teamtailor.com",  # no subdomain
        "https://personio.jobs.personio.net/xml",  # tld must be de or com
        "https://jobs.personio.de/xml",  # no company subdomain
        "https://recruitee.com/api/offers/",
        "https://pinpointhq.com/postings.json",
        "https://apply.workable.com/",
        "https://apply.workable.com",
    ],
)
def test_unsupported_urls_are_rejected_rather_than_guessed(url: str) -> None:
    with pytest.raises(UnsupportedBoardError):
        detect_board(url)


@pytest.mark.parametrize(
    "url", ["", "   ", "ftp://boards.greenhouse.io/x", "not a url", "https://"]
)
def test_malformed_input_is_rejected(url: str) -> None:
    with pytest.raises(MalformedBoardUrlError):
        detect_board(url)
