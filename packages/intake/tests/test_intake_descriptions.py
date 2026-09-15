"""Slice C7: `fetch_description`, one posting at a time.

Every JSON/XML fixture here is real content captured live on 2026-09-15 (see
`jfl_intake.descriptions`'s module docstring and each `_fetch_*` function for
the endpoint and what was verified against), trimmed to a few hundred
characters -- same convention as the board-check fixtures. No test opens a
socket: every transport here is a fake.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_args

from jfl_core.models import BoardPlatform
from jfl_intake.descriptions import (
    MAX_CHARS,
    PLATFORM_HANDLERS,
    DescriptionResult,
    fetch_description,
    html_to_text,
)
from jfl_intake.http import HttpResponse, RequestBudgetExceeded, TransportError

FIXTURES = Path(__file__).parent / "fixtures"


def json_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def text_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


class FakeTransport:
    """Replays one response (or raises one error) per URL; asserts a handler
    never calls a method it has no business calling.
    """

    def __init__(self, responses: Mapping[str, HttpResponse | Exception] | None = None) -> None:
        self._responses = dict(responses or {})
        self.calls: list[str] = []

    def get_json(self, url: str) -> HttpResponse:
        self.calls.append(url)
        response = self._responses[url]
        if isinstance(response, Exception):
            raise response
        return response

    def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
        raise AssertionError("no description fetch here POSTs")

    def get_text(self, url: str) -> HttpResponse:
        self.calls.append(url)
        response = self._responses[url]
        if isinstance(response, Exception):
            raise response
        return response


class NeverCalledTransport:
    def get_json(self, url: str) -> HttpResponse:
        raise AssertionError("must not make a request")

    def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
        raise AssertionError("must not make a request")

    def get_text(self, url: str) -> HttpResponse:
        raise AssertionError("must not make a request")


def ok(body: Any) -> HttpResponse:
    return HttpResponse(status=200, body=body)


def status(code: int) -> HttpResponse:
    return HttpResponse(status=code, body=None)


# -- the registry agrees with BoardPlatform --------------------------------


def test_every_board_platform_has_a_description_handler() -> None:
    assert set(PLATFORM_HANDLERS) == set(get_args(BoardPlatform))


# -- Greenhouse --------------------------------------------------------


GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/anthropic/jobs/4461450008"


def test_greenhouse_double_decodes_and_extracts_text() -> None:
    transport = FakeTransport(
        {GREENHOUSE_URL: ok(json_fixture("descriptions_greenhouse_job.json"))}
    )
    result = fetch_description("greenhouse", {"token": "anthropic"}, "4461450008", None, transport)
    assert result.error_code is None
    assert result.requests == 1
    assert result.text is not None
    assert "About Anthropic" in result.text
    assert "&lt;" not in result.text  # the double-encoding is gone, not just re-escaped
    assert "&gt;" not in result.text


def test_greenhouse_404_is_not_found() -> None:
    transport = FakeTransport({GREENHOUSE_URL: status(404)})
    result = fetch_description("greenhouse", {"token": "anthropic"}, "4461450008", None, transport)
    assert result == DescriptionResult(text=None, error_code="not_found", requests=1)
    assert result.is_transient is False


def test_greenhouse_rate_limited_is_transient() -> None:
    transport = FakeTransport({GREENHOUSE_URL: status(429)})
    result = fetch_description("greenhouse", {"token": "anthropic"}, "4461450008", None, transport)
    assert result.error_code == "unreachable"
    assert result.is_transient is True


def test_greenhouse_server_error_is_transient() -> None:
    transport = FakeTransport({GREENHOUSE_URL: status(503)})
    result = fetch_description("greenhouse", {"token": "anthropic"}, "4461450008", None, transport)
    assert result.error_code == "unreachable"


def test_greenhouse_malformed_body_is_bad_response() -> None:
    transport = FakeTransport({GREENHOUSE_URL: ok({"title": "no content field"})})
    result = fetch_description("greenhouse", {"token": "anthropic"}, "4461450008", None, transport)
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=1)


def test_greenhouse_blank_content_is_empty() -> None:
    transport = FakeTransport({GREENHOUSE_URL: ok({"content": "&lt;p&gt;&amp;nbsp;&lt;/p&gt;"})})
    result = fetch_description("greenhouse", {"token": "anthropic"}, "4461450008", None, transport)
    assert result == DescriptionResult(text=None, error_code="empty", requests=1)


def test_greenhouse_invalid_board_key_is_bad_response_with_no_request() -> None:
    transport = NeverCalledTransport()
    result = fetch_description("greenhouse", {}, "4461450008", None, transport)
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=0)


def test_greenhouse_timeout_is_unreachable() -> None:
    transport = FakeTransport({GREENHOUSE_URL: TransportError("timeout")})
    result = fetch_description("greenhouse", {"token": "anthropic"}, "4461450008", None, transport)
    assert result == DescriptionResult(text=None, error_code="unreachable", requests=1)
    assert result.is_transient is True


def test_greenhouse_request_budget_exceeded_is_unreachable() -> None:
    transport = FakeTransport({GREENHOUSE_URL: RequestBudgetExceeded("deadline_exceeded")})
    result = fetch_description("greenhouse", {"token": "anthropic"}, "4461450008", None, transport)
    assert result.error_code == "unreachable"


# -- Ashby ---------------------------------------------------------------

ASHBY_URL = "https://api.ashbyhq.com/posting-api/job-board/openai"


def test_ashby_refetches_listing_and_matches_by_id() -> None:
    transport = FakeTransport({ASHBY_URL: ok(json_fixture("descriptions_ashby_job_board.json"))})
    result = fetch_description(
        "ashby", {"name": "openai"}, "8fb1615c-34bf-47c4-a1d1-b7b2f836bbd3", None, transport
    )
    assert result.error_code is None
    assert result.text is not None
    assert "About the Team" in result.text
    assert result.requests == 1


def test_ashby_id_not_in_listing_is_not_found() -> None:
    transport = FakeTransport({ASHBY_URL: ok(json_fixture("descriptions_ashby_job_board.json"))})
    result = fetch_description("ashby", {"name": "openai"}, "no-such-id", None, transport)
    assert result == DescriptionResult(text=None, error_code="not_found", requests=1)


def test_ashby_falls_back_to_plain_description() -> None:
    body = {"jobs": [{"id": "x", "descriptionPlain": "Plain text only.\n\nSecond paragraph."}]}
    transport = FakeTransport({ASHBY_URL: ok(body)})
    result = fetch_description("ashby", {"name": "openai"}, "x", None, transport)
    assert result.error_code is None
    # `descriptionPlain` has no tags to mark a paragraph boundary, so its own
    # blank lines are trusted for that instead (`_plain_text`).
    assert result.text == "Plain text only.\nSecond paragraph."


def test_ashby_malformed_listing_is_bad_response() -> None:
    transport = FakeTransport({ASHBY_URL: ok(["not", "the", "right", "shape"])})
    result = fetch_description("ashby", {"name": "openai"}, "x", None, transport)
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=1)


# -- Lever -----------------------------------------------------------------

LEVER_URL = (
    "https://api.lever.co/v0/postings/palantir/ac978161-6f46-4f6b-ad9e-a258e642751c?mode=json"
)


def test_lever_combines_description_lists_and_additional() -> None:
    transport = FakeTransport({LEVER_URL: ok(json_fixture("descriptions_lever_posting.json"))})
    result = fetch_description(
        "lever", {"company": "palantir"}, "ac978161-6f46-4f6b-ad9e-a258e642751c", None, transport
    )
    assert result.error_code is None
    assert result.text is not None
    assert "A World-Changing Company" in result.text
    assert "Administrative Business Partner (Foundry)" in result.text  # the list's heading
    assert "- Provide administrative support" in result.text  # a bare <li>, no enclosing <ul>


def test_lever_not_found_posting_is_not_found() -> None:
    transport = FakeTransport(
        {LEVER_URL: HttpResponse(status=404, body={"ok": False, "error": "Document not found"})}
    )
    result = fetch_description(
        "lever", {"company": "palantir"}, "ac978161-6f46-4f6b-ad9e-a258e642751c", None, transport
    )
    assert result == DescriptionResult(text=None, error_code="not_found", requests=1)


def test_lever_missing_description_field_is_bad_response() -> None:
    transport = FakeTransport({LEVER_URL: ok({"id": "x"})})
    result = fetch_description(
        "lever", {"company": "palantir"}, "ac978161-6f46-4f6b-ad9e-a258e642751c", None, transport
    )
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=1)


# -- Workday -----------------------------------------------------------

NVIDIA_KEY = {"tenant": "nvidia", "wd": "wd5", "site": "NVIDIAExternalCareerSite"}
WORKDAY_PATH = "/job/US-CA-Santa-Clara/Lead-Safety-Architect---Autonomous-Vehicles_JR2014137"
WORKDAY_POSTING_URL = f"https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite{WORKDAY_PATH}"
WORKDAY_ENDPOINT = (
    f"https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite{WORKDAY_PATH}"
)


def test_workday_derives_the_cxs_endpoint_from_the_stored_url() -> None:
    transport = FakeTransport({WORKDAY_ENDPOINT: ok(json_fixture("descriptions_workday_job.json"))})
    result = fetch_description("workday", NVIDIA_KEY, "JR2014137", WORKDAY_POSTING_URL, transport)
    assert result.error_code is None
    assert result.text is not None
    assert "NVIDIA has been transforming" in result.text
    assert transport.calls == [WORKDAY_ENDPOINT]


def test_workday_without_a_stored_url_is_bad_response_with_no_request() -> None:
    transport = NeverCalledTransport()
    result = fetch_description("workday", NVIDIA_KEY, "JR2014137", None, transport)
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=0)


def test_workday_url_from_a_different_tenant_is_bad_response_with_no_request() -> None:
    transport = NeverCalledTransport()
    wrong_url = "https://someone-else.wd5.myworkdayjobs.com/OtherSite/job/x_JR1"
    result = fetch_description("workday", NVIDIA_KEY, "JR2014137", wrong_url, transport)
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=0)


def test_workday_bad_request_status_is_bad_response_not_empty() -> None:
    # Workday answers a malformed request with 400 -- fail loudly, never treat
    # it as an empty board (see adapters.workday's docstring).
    transport = FakeTransport({WORKDAY_ENDPOINT: status(400)})
    result = fetch_description("workday", NVIDIA_KEY, "JR2014137", WORKDAY_POSTING_URL, transport)
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=1)


def test_workday_unexpected_shape_is_bad_response() -> None:
    transport = FakeTransport(
        {WORKDAY_ENDPOINT: ok({"jobPostingInfo": {"title": "no description key"}})}
    )
    result = fetch_description("workday", NVIDIA_KEY, "JR2014137", WORKDAY_POSTING_URL, transport)
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=1)


def test_workday_invalid_wd_shape_is_bad_response_with_no_request() -> None:
    transport = NeverCalledTransport()
    bad_key = {"tenant": "nvidia", "wd": "notwd", "site": "NVIDIAExternalCareerSite"}
    result = fetch_description("workday", bad_key, "JR2014137", WORKDAY_POSTING_URL, transport)
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=0)


# -- SmartRecruiters --------------------------------------------------------

SMARTRECRUITERS_URL = (
    "https://api.smartrecruiters.com/v1/companies/BoschGroup/postings/744000149723720"
)


def test_smartrecruiters_concatenates_known_sections_in_order() -> None:
    transport = FakeTransport(
        {SMARTRECRUITERS_URL: ok(json_fixture("descriptions_smartrecruiters_posting.json"))}
    )
    result = fetch_description(
        "smartrecruiters", {"company_id": "BoschGroup"}, "744000149723720", None, transport
    )
    assert result.error_code is None
    assert result.text is not None
    company_idx = result.text.index("Bosch was founded")
    job_idx = result.text.index("Technical Project Manager acts")
    quals_idx = result.text.index("Required skills")
    assert company_idx < job_idx < quals_idx
    assert "- Project management" in result.text  # a <li> inside <ul>


def test_smartrecruiters_missing_sections_is_bad_response() -> None:
    transport = FakeTransport({SMARTRECRUITERS_URL: ok({"jobAd": {}})})
    result = fetch_description(
        "smartrecruiters", {"company_id": "BoschGroup"}, "744000149723720", None, transport
    )
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=1)


# -- Workable ----------------------------------------------------------

WORKABLE_URL = "https://apply.workable.com/api/v1/widget/accounts/huggingface?details=true"
WORKABLE_POSTING_URL = "https://apply.workable.com/huggingface/j/9E2A4C02C7"


def test_workable_resolves_shortcode_from_url_and_matches_widget_listing() -> None:
    transport = FakeTransport({WORKABLE_URL: ok(json_fixture("descriptions_workable_widget.json"))})
    result = fetch_description(
        "workable", {"subdomain": "huggingface"}, "6074222", WORKABLE_POSTING_URL, transport
    )
    assert result.error_code is None
    assert result.text is not None
    assert "Hugging Face" in result.text


def test_workable_without_a_stored_url_is_bad_response_with_no_request() -> None:
    transport = NeverCalledTransport()
    result = fetch_description("workable", {"subdomain": "huggingface"}, "6074222", None, transport)
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=0)


def test_workable_shortcode_not_in_widget_listing_is_not_found() -> None:
    transport = FakeTransport({WORKABLE_URL: ok(json_fixture("descriptions_workable_widget.json"))})
    missing_url = "https://apply.workable.com/huggingface/j/NOPE"
    result = fetch_description(
        "workable", {"subdomain": "huggingface"}, "1", missing_url, transport
    )
    assert result == DescriptionResult(text=None, error_code="not_found", requests=1)


# -- Pinpoint ------------------------------------------------------------

PINPOINT_URL = "https://sunking.pinpointhq.com/postings.json"


def test_pinpoint_refetches_listing_and_combines_named_sections() -> None:
    transport = FakeTransport(
        {PINPOINT_URL: ok(json_fixture("descriptions_pinpoint_postings.json"))}
    )
    result = fetch_description("pinpoint", {"company": "sunking"}, "332887", None, transport)
    assert result.error_code is None
    assert result.text is not None
    assert "Job Location" in result.text
    assert "What you would be expected to do" in result.text  # key_responsibilities_header


def test_pinpoint_id_not_in_listing_is_not_found() -> None:
    transport = FakeTransport(
        {PINPOINT_URL: ok(json_fixture("descriptions_pinpoint_postings.json"))}
    )
    result = fetch_description("pinpoint", {"company": "sunking"}, "no-such-id", None, transport)
    assert result == DescriptionResult(text=None, error_code="not_found", requests=1)


# -- Recruitee ---------------------------------------------------------

RECRUITEE_URL = "https://make.recruitee.com/api/offers/2695896"


def test_recruitee_combines_description_and_requirements() -> None:
    transport = FakeTransport(
        {RECRUITEE_URL: ok(json_fixture("descriptions_recruitee_offer.json"))}
    )
    result = fetch_description("recruitee", {"company": "make"}, "2695896", None, transport)
    assert result.error_code is None
    assert result.text is not None
    assert "Location:" in result.text
    assert "What You'll Do" in result.text


def test_recruitee_404_is_not_found() -> None:
    transport = FakeTransport({RECRUITEE_URL: status(404)})
    result = fetch_description("recruitee", {"company": "make"}, "2695896", None, transport)
    assert result == DescriptionResult(text=None, error_code="not_found", requests=1)


def test_recruitee_blank_fields_is_empty() -> None:
    transport = FakeTransport(
        {RECRUITEE_URL: ok({"offer": {"id": 2695896, "description": "", "requirements": None}})}
    )
    result = fetch_description("recruitee", {"company": "make"}, "2695896", None, transport)
    assert result == DescriptionResult(text=None, error_code="empty", requests=1)


# -- Teamtailor ----------------------------------------------------------

TEAMTAILOR_URL = "https://career.teamtailor.com/jobs.rss"


def text_response(body: str) -> HttpResponse:
    return HttpResponse(status=200, body=body)


def test_teamtailor_refetches_the_feed_and_matches_by_guid() -> None:
    transport = FakeTransport(
        {TEAMTAILOR_URL: text_response(text_fixture("descriptions_teamtailor.rss"))}
    )
    result = fetch_description(
        "teamtailor",
        {"site": "career.teamtailor.com"},
        "a79a10f6-9f69-49b0-9f6d-8bed3d6e3a4f",
        None,
        transport,
    )
    assert result.error_code is None
    assert result.text is not None
    assert "Let's build the future together!" in result.text
    assert "Teamtailor is an Employer Branding" in result.text


def test_teamtailor_item_with_no_description_is_empty() -> None:
    transport = FakeTransport(
        {TEAMTAILOR_URL: text_response(text_fixture("descriptions_teamtailor.rss"))}
    )
    result = fetch_description(
        "teamtailor",
        {"site": "career.teamtailor.com"},
        "00000000-0000-0000-0000-000000000000",
        None,
        transport,
    )
    assert result == DescriptionResult(text=None, error_code="empty", requests=1)


def test_teamtailor_guid_not_in_feed_is_not_found() -> None:
    transport = FakeTransport(
        {TEAMTAILOR_URL: text_response(text_fixture("descriptions_teamtailor.rss"))}
    )
    result = fetch_description(
        "teamtailor", {"site": "career.teamtailor.com"}, "no-such-guid", None, transport
    )
    assert result == DescriptionResult(text=None, error_code="not_found", requests=1)


def test_teamtailor_malformed_feed_is_bad_response() -> None:
    transport = FakeTransport({TEAMTAILOR_URL: text_response("not xml at all <<<")})
    result = fetch_description(
        "teamtailor", {"site": "career.teamtailor.com"}, "any-guid", None, transport
    )
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=1)


# -- Personio ------------------------------------------------------------

PERSONIO_URL = "https://personio.jobs.personio.de/xml"


def test_personio_refetches_the_feed_and_matches_by_id() -> None:
    transport = FakeTransport(
        {PERSONIO_URL: text_response(text_fixture("descriptions_personio.xml"))}
    )
    result = fetch_description(
        "personio", {"company": "personio", "tld": "de"}, "1834171", None, transport
    )
    assert result.error_code is None
    assert result.text is not None
    assert "The Role" in result.text
    assert "- Own pipelines" in result.text
    assert "What you bring" in result.text


def test_personio_present_but_empty_job_descriptions_is_empty() -> None:
    # The one live shape actually observed 2026-09-15 (see the module
    # docstring): the element exists, has no children.
    transport = FakeTransport(
        {PERSONIO_URL: text_response(text_fixture("descriptions_personio.xml"))}
    )
    result = fetch_description(
        "personio", {"company": "personio", "tld": "de"}, "1834172", None, transport
    )
    assert result == DescriptionResult(text=None, error_code="empty", requests=1)


def test_personio_missing_job_descriptions_element_is_bad_response() -> None:
    xml = """<?xml version="1.0"?><workzag-jobs><position><id>1</id></position></workzag-jobs>"""
    transport = FakeTransport({PERSONIO_URL: text_response(xml)})
    result = fetch_description(
        "personio", {"company": "personio", "tld": "de"}, "1", None, transport
    )
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=1)


def test_personio_invalid_tld_is_bad_response_with_no_request() -> None:
    transport = NeverCalledTransport()
    result = fetch_description(
        "personio", {"company": "personio", "tld": "co.uk"}, "1834171", None, transport
    )
    assert result == DescriptionResult(text=None, error_code="bad_response", requests=0)


# -- Rippling --------------------------------------------------------------

RIPPLING_URL = (
    "https://api.rippling.com/platform/api/ats/v1/board/rippling"
    "/jobs/75ad50c6-778f-42ee-9c63-70d1cd687202"
)


def test_rippling_combines_role_then_company() -> None:
    transport = FakeTransport(
        {RIPPLING_URL: ok(json_fixture("descriptions_rippling_posting.json"))}
    )
    result = fetch_description(
        "rippling", {"slug": "rippling"}, "75ad50c6-778f-42ee-9c63-70d1cd687202", None, transport
    )
    assert result.error_code is None
    assert result.text is not None
    role_idx = result.text.index("About the role")
    company_idx = result.text.index("About Rippling")
    assert role_idx < company_idx


def test_rippling_404_is_not_found() -> None:
    transport = FakeTransport({RIPPLING_URL: status(404)})
    result = fetch_description(
        "rippling", {"slug": "rippling"}, "75ad50c6-778f-42ee-9c63-70d1cd687202", None, transport
    )
    assert result == DescriptionResult(text=None, error_code="not_found", requests=1)


# -- Breezy: unsupported, always -------------------------------------------


def test_breezy_is_unsupported_with_no_request() -> None:
    transport = NeverCalledTransport()
    result = fetch_description("breezy", {"company": "breezy"}, "98323abf2296", None, transport)
    assert result == DescriptionResult(text=None, error_code="unsupported_platform", requests=0)
    assert result.is_transient is False


# -- html_to_text --------------------------------------------------------


def test_html_to_text_headings_paragraphs_and_list_items_on_own_lines() -> None:
    markup = "<h2>A Heading</h2><p>A paragraph.</p><ul><li>First</li><li>Second</li></ul>"
    assert html_to_text(markup) == "A Heading\nA paragraph.\n- First\n- Second"


def test_html_to_text_decodes_entities() -> None:
    assert html_to_text("<p>Tom &amp; Jerry &mdash; a &lt;team&gt;</p>") == "Tom & Jerry — a <team>"


def test_html_to_text_nested_markup_li_wrapping_p() -> None:
    markup = "<ul><li><p>Wrapped in a paragraph.</p></li><li>Plain.</li></ul>"
    assert html_to_text(markup) == "- Wrapped in a paragraph.\n- Plain."


def test_html_to_text_inline_tags_do_not_break_a_line() -> None:
    markup = "<p>Some <strong>bold</strong> and <em>italic</em> <a href='#'>link</a> text.</p>"
    assert html_to_text(markup) == "Some bold and italic link text."


def test_html_to_text_drops_scripts_styles_and_comments() -> None:
    # Dropped, but not glued together: script/style content is invisible,
    # not a line break, so the words on either side end up space-separated.
    markup = "<div>Before<script>alert(1)</script><style>.x{}</style><!-- a comment -->After</div>"
    assert html_to_text(markup) == "Before After"


def test_html_to_text_br_is_a_line_break() -> None:
    assert html_to_text("<p>Line one<br>Line two</p>") == "Line one\nLine two"


def test_html_to_text_collapses_whitespace_including_nbsp() -> None:
    assert html_to_text("<p>Too   much\n\nspace &nbsp; here</p>") == "Too much space here"


def test_html_to_text_blank_input_is_blank() -> None:
    assert html_to_text("<div><p></p><p>   </p></div>") == ""


def test_finish_truncates_and_marks_it() -> None:
    transport = FakeTransport(
        {GREENHOUSE_URL: ok({"content": f"&lt;p&gt;{'x' * (MAX_CHARS + 500)}&lt;/p&gt;"})}
    )
    result = fetch_description("greenhouse", {"token": "anthropic"}, "4461450008", None, transport)
    assert result.error_code is None
    assert result.text is not None
    assert len(result.text) <= MAX_CHARS + len(f"\n\n[truncated at {MAX_CHARS} characters]")
    assert result.text.endswith(f"[truncated at {MAX_CHARS} characters]")
