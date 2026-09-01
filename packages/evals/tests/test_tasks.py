"""Unit tests for the Inspect task wiring (jfl_evals.tasks) that don't need the
network or a live model: sample construction, the metric functions against
hand-built SampleScore lists, and the task's own `limit` default. The solver and
scorer's real behaviour end-to-end (a real check_text call) is exercised only by
the eval itself, run manually -- see README.md -- never by the default test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from inspect_ai.scorer import Metric, MetricProtocol, SampleScore, Score, Value
from jfl_evals.dataset import GoldenItem
from jfl_evals.tasks import (
    DEFAULT_DATASET_PATH,
    _sample_for,
    claim_gate_fever,
    confusion_matrix,
    framing_rate,
    over_claim_rate,
    over_flag_rate,
)


def _call(m: Metric, scores: list[SampleScore]) -> Value:
    # `Metric` is `MetricProtocol | MetricDeprecated` (inspect_ai's own signature
    # migration) -- the two protocols disagree on the list element type, so a bare
    # call through the union type does not check. Every metric in jfl_evals.tasks
    # is written against MetricProtocol; this cast just says so once instead of at
    # every call site below.
    return cast(MetricProtocol, m)(scores)


def _item() -> GoldenItem:
    return GoldenItem(
        id="fever-1",
        claim="The sky is blue.",
        expected_verdict="supported",
        evidence=["The sky appears blue due to Rayleigh scattering."],
        fever_label="SUPPORTS",
    )


class TestSampleFor:
    def test_input_is_the_claim_text(self) -> None:
        sample = _sample_for(_item())
        assert sample.input == "The sky is blue."

    def test_target_is_the_expected_verdict(self) -> None:
        sample = _sample_for(_item())
        assert sample.target == "supported"

    def test_id_round_trips_the_item_id(self) -> None:
        sample = _sample_for(_item())
        assert sample.id == "fever-1"

    def test_metadata_carries_the_full_item_for_the_solver_to_rebuild(self) -> None:
        sample = _sample_for(_item())
        assert sample.metadata is not None
        rebuilt = GoldenItem.model_validate(sample.metadata["item"])
        assert rebuilt == _item()


def _sample_score(expected: str, actual: str | None, kind: str | None = "claim") -> SampleScore:
    return SampleScore(
        score=Score(
            value={"expected": expected, "actual": actual, "kind": kind},
            metadata={"expected": expected, "actual": actual, "kind": kind, "error": None},
        ),
        sample_id="x",
    )


class TestMetrics:
    """Each metric is a thin adapter over jfl_evals.scoring.aggregate -- these just
    confirm the wiring (right field, right shape), not the arithmetic itself,
    which is covered exhaustively in test_scoring.py.
    """

    def test_over_claim_rate_reports_rate_count_and_denominator(self) -> None:
        scores = [
            _sample_score("unsupported", "supported"),  # over-claim
            _sample_score("supported", "supported"),
        ]
        result = _call(over_claim_rate(), scores)
        assert result == {"rate": 1.0, "count": 1, "denominator": 1}

    def test_over_flag_rate_reports_rate_count_and_denominator(self) -> None:
        scores = [
            _sample_score("supported", "unsupported"),  # over-flag
            _sample_score("supported", "supported"),
        ]
        result = _call(over_flag_rate(), scores)
        assert result == {"rate": 0.5, "count": 1, "denominator": 2}

    def test_confusion_matrix_flattens_pairs_and_includes_item_counts(self) -> None:
        scores = [_sample_score("supported", "supported")]
        result = _call(confusion_matrix(), scores)
        assert isinstance(result, dict)
        assert result["supported->supported"] == 1
        assert result["n_items"] == 1
        assert result["n_scored"] == 1
        assert result["n_errors"] == 0

    def test_framing_rate_reports_rate_and_count(self) -> None:
        scores = [
            _sample_score("supported", "supported", kind="claim"),
            _sample_score("supported", "supported", kind="framing"),
        ]
        result = _call(framing_rate(), scores)
        # This framing item's expected label is "supported", so it is not an
        # over-claim -- the joint count/rate stay at zero alongside the plain one.
        assert result == {
            "rate": 0.5,
            "count": 1,
            "over_claim_rate": 0.0,
            "over_claim_count": 0,
        }

    def test_framing_rate_reports_the_joint_over_claim_count_when_framing_over_claims(
        self,
    ) -> None:
        """The failure jfl_evals.scoring's module docstring is about: a framing
        misclassification on an item that should NOT have been "supported" --
        the gate's prompt contract forces framing straight to "supported" with no
        corpus check, so this is the one path with no defence in depth anywhere.
        """
        scores = [
            _sample_score("supported", "supported", kind="claim"),  # ordinary match
            _sample_score("review", "supported", kind="framing"),  # framing + over-claim
        ]
        result = _call(framing_rate(), scores)
        assert result == {
            "rate": 0.5,
            "count": 1,
            "over_claim_rate": 0.5,
            "over_claim_count": 1,
        }

    def test_empty_scores_do_not_raise(self) -> None:
        for metric in (over_claim_rate(), over_flag_rate(), framing_rate()):
            result = _call(metric, [])
            assert isinstance(result, dict)
            assert result["rate"] == 0.0


class TestClaimGateFeverTask:
    def test_default_limit_truncates_the_dataset_to_five_items(self) -> None:
        task = claim_gate_fever()
        assert len(task.dataset) == 5

    def test_limit_is_respected(self) -> None:
        task = claim_gate_fever(limit=12)
        assert len(task.dataset) == 12

    def test_default_dataset_path_points_at_the_committed_cache(self) -> None:
        assert Path(DEFAULT_DATASET_PATH).exists()

    def test_model_is_none_none_since_the_solver_never_calls_generate(self) -> None:
        task = claim_gate_fever()
        assert str(task.model) == "none/none"
