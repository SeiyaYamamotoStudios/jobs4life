"""Orchestration for slice 2a: job intake, requirement extraction, coverage,
and the gap-answer write-back.

Fixed control flow throughout -- see CLAUDE.md, "What is agentic, and what is
not." Each function here is one arrow of the pipeline the slice brief
describes:

    job ad text -> [extract_requirements] -> job + job_requirements rows
    requirements + corpus -> [check_coverage] -> coverage rows + gap questions
    gap question + answer -> (no model call) -> adjudicated span

Nothing in this module makes a model call directly -- that is `extract.py` and
`coverage.py`'s job. This module only sequences those calls with id derivation
and repository writes.
"""

from __future__ import annotations

import uuid

from jfl_core.context import RequestContext
from jfl_core.ids import adjudicated_span_id_from_answer, content_hash
from jfl_core.ids import gap_question_id as derive_gap_question_id
from jfl_core.ids import job_id as derive_job_id
from jfl_core.ids import requirement_id as derive_requirement_id
from jfl_core.models import GapQuestion, Job, JobRequirement, JobSource, RequirementCoverage, Span
from jfl_core.repositories import GroundingRepository, JobRepository, RunRepository

from jfl_generate.coverage import check_coverage
from jfl_generate.errors import GenerateError
from jfl_generate.extract import extract_requirements

# Statuses that trigger a gap question -- the corpus does not (yet) evidence the
# requirement, as opposed to `evidenced` (nothing to ask) or `contradicted`
# (asking would prompt the user to argue with their own corpus).
_GAP_STATUSES = ("absent", "partial")


def add_job(
    ctx: RequestContext,
    run_repo: RunRepository,
    job_repo: JobRepository,
    raw_text: str,
    source: JobSource = "paste",
) -> tuple[Job, list[JobRequirement]]:
    """Extract requirements from a pasted job ad and store the job + requirement
    rows. Re-adding the same ad text is idempotent: the job id is deterministic
    from its content hash, and `replace_requirements` overwrites rather than
    duplicates.
    """
    extracted = extract_requirements(ctx, run_repo, raw_text)

    jid = derive_job_id(ctx.user_id, raw_text)
    job = Job(
        id=jid,
        user_id=ctx.user_id,
        source=source,
        employer=extracted.employer,
        title=extracted.title,
        location=extracted.location,
        raw_text=raw_text,
        content_hash=content_hash(raw_text),
    )
    job_repo.upsert_job(job)

    requirements = [
        JobRequirement(
            id=derive_requirement_id(jid, item.text),
            user_id=ctx.user_id,
            job_id=jid,
            ordinal=i,
            text=item.text,
            necessity=item.necessity,
        )
        for i, item in enumerate(extracted.requirements)
    ]
    job_repo.replace_requirements(ctx.user_id, jid, requirements)
    return job, requirements


def run_coverage(
    ctx: RequestContext,
    grounding_repo: GroundingRepository,
    run_repo: RunRepository,
    job_repo: JobRepository,
    job_id: uuid.UUID,
) -> list[RequirementCoverage]:
    """Check corpus coverage for every requirement on a job, write one
    append-only `requirement_coverage` row per requirement for this run, and
    upsert a gap question for anything `absent` or `partial`. Re-running this
    after a gap question is answered is the whole demonstration: the same
    requirement should move off `absent`/`partial` on the next pass.
    """
    found = job_repo.get_job(ctx.user_id, job_id)
    if found is None:
        raise GenerateError(f"no job {job_id} for this user")
    _job, requirements = found
    if not requirements:
        raise GenerateError("job has no requirements to check")

    output = check_coverage(ctx, grounding_repo, run_repo, [r.text for r in requirements])

    coverage_rows: list[RequirementCoverage] = []
    for requirement, item in zip(requirements, output.results, strict=True):
        coverage = RequirementCoverage(
            user_id=ctx.user_id,
            requirement_id=requirement.id,
            trace_id=ctx.trace_id,
            status=item.status,
            cited_span_ids=item.cited_span_ids,
            reason=item.reason,
        )
        job_repo.record_coverage(coverage)
        coverage_rows.append(coverage)

        if item.status in _GAP_STATUSES and item.question:
            job_repo.upsert_gap_question(
                GapQuestion(
                    id=derive_gap_question_id(requirement.id),
                    user_id=ctx.user_id,
                    requirement_id=requirement.id,
                    question=item.question,
                )
            )

    return coverage_rows


def answer_question(
    ctx: RequestContext,
    grounding_repo: GroundingRepository,
    job_repo: JobRepository,
    question_id: uuid.UUID,
    answer_text: str,
) -> uuid.UUID:
    """Write the user's answer back to the corpus verbatim, as an adjudicated
    span. NO model call anywhere in this path -- see CLAUDE.md's decisions log:
    a model tidying the answer into a neater corpus fact is the ratchet in
    miniature.
    """
    question = job_repo.get_question(ctx.user_id, question_id)
    if question is None:
        raise GenerateError(f"no gap question {question_id} for this user")
    if question.status != "open":
        raise GenerateError(f"question {question_id} is already {question.status}")

    span_id = adjudicated_span_id_from_answer(ctx.user_id, answer_text, question_id)
    span = Span(
        id=span_id,
        user_id=ctx.user_id,
        document_id=None,
        provenance="adjudicated",
        kind="paragraph",
        section_path="Answered questions",
        text=answer_text,
        content_hash=content_hash(answer_text),
    )
    grounding_repo.add_adjudicated_span(ctx.user_id, span)
    job_repo.mark_question_answered(ctx.user_id, question_id, answer_text, span_id)
    return span_id
